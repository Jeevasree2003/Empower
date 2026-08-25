"""Orchestrate live knowledge retrieval: entities → queries → search → OpenIE."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ktc.cleaning import (
    clean_knowledge_text,
    drop_strict_substring_texts,
    normalize_clause_key,
    strip_legal_citations,
)
from ktc.coreference import resolve_coreferences
from ktc.entity_extraction import extract_entities
from ktc.extraction import extract_triplets
from ktc.filtering import filter_triplets
from ktc.knowledge_item import KnowledgeCandidate
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_retrieval import LiveDeadline, run_io_tasks, search_allowlisted
from ktc.live_summarize import (
    LiveKnowledgeSentence,
    LivePageStats,
    is_scraped_boilerplate,
    summarize_search_results,
)
from ktc.query_builder import SearchQuery, build_queries
from ktc.ranking import MIN_COSINE, encode_texts_cached, select_sentences_by_cosine

logger = logging.getLogger(__name__)


_SPEAKER_SPLIT = re.compile(
    r"\s+(?=agent:|bot:|victim:|user:|seeker:|client:)",
    re.IGNORECASE,
)
_VICTIM_PREFIXES = ("victim:", "user:", "seeker:", "client:")


@dataclass
class LiveFunnel:
    live_sentences: int = 0
    live_triplets: int = 0
    live_sentence_relevance: int = 0
    live_sentences_used_directly: int = 0
    pages: List[Dict] = field(default_factory=list)


def victim_utterances_from_history(dialog_history: str) -> List[str]:
    """Return victim/user utterance texts in order from formatted dialog history."""
    parts = _SPEAKER_SPLIT.split(dialog_history.strip())
    texts: List[str] = []
    for part in parts:
        if part.lower().startswith(_VICTIM_PREFIXES):
            texts.append(part.split(":", 1)[-1].strip())
    return [t for t in texts if t]


def victim_utterance_from_history(dialog_history: str) -> str:
    """Return the most recent victim utterance from formatted dialog history.

    KARE raw roles are ``user``/``bot``; preprocess maps them to ``victim``/``agent``.
    Accept both so inspect scripts on raw JSONL still trigger live retrieval.
    """
    utterances = victim_utterances_from_history(dialog_history)
    return utterances[-1] if utterances else ""


def _norm_key(text: str) -> str:
    return normalize_clause_key(text)


def _live_text_without_citations(text: str) -> str:
    return strip_legal_citations(text or "")


_DIRECT_MIN_LEN = 40
_DIRECT_MAX_LEN = 420


def _usable_direct_sentence(text: str) -> str:
    """Citation-stripped live sentence that is not empty, boilerplate, or the wrong length."""
    cleaned = _live_text_without_citations(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or is_scraped_boilerplate(cleaned):
        return ""
    if len(cleaned) < _DIRECT_MIN_LEN or len(cleaned) > _DIRECT_MAX_LEN:
        return ""
    return cleaned


def dedup_candidates(candidates: Sequence[KnowledgeCandidate]) -> List[KnowledgeCandidate]:
    """Keep the first candidate for each normalized sentence; drop strict substring clauses."""
    unique: List[KnowledgeCandidate] = []
    seen = set()
    for item in candidates:
        text = _live_text_without_citations(item.text)
        if not text:
            continue
        if text != (item.text or "").strip():
            item = replace(item, text=text)
        key = _norm_key(item.text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    kept_keys = {
        _norm_key(text) for text in drop_strict_substring_texts([item.text for item in unique])
    }
    return [item for item in unique if _norm_key(item.text) in kept_keys]


def _unique_live_sentences(sentences: Sequence[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for sentence in sentences:
        key = re.sub(r"\s+", " ", (sentence or "").strip().lower()).rstrip(".!?")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(sentence.strip())
    return unique


def sentence_relevance_candidates(
    pages: Sequence[LivePageStats],
    ranker,
    *,
    min_cosine: float = MIN_COSINE,
    top_k_per_page: int = 8,
) -> List[KnowledgeCandidate]:
    """SBERT-selected live sentences that do not depend on OpenIE succeeding.

    All pages in a turn are scored in one encoder batch, then split per page.
    """
    if ranker is None:
        return []
    started = time.monotonic()
    page_uniques = [_unique_live_sentences(page.sentences) for page in pages]
    total_sentences = sum(len(group) for group in page_uniques)
    selected: List[KnowledgeCandidate] = []
    model = getattr(ranker, "model", None)
    encode = getattr(model, "encode", None) if model is not None else None
    page_kept: List[List[Tuple[str, float]]] = []
    if callable(encode):
        queries: List[str] = []
        query_index: Dict[str, int] = {}
        for page in pages:
            query = (page.query or "").strip() or " "
            if query not in query_index:
                query_index[query] = len(queries)
                queries.append(query)
        flat: List[str] = []
        spans: List[Tuple[int, int]] = []
        for group in page_uniques:
            spans.append((len(flat), len(group)))
            flat.extend(group)
        batch = queries + flat
        embs = encode_texts_cached(model, batch) if batch else np.zeros((0, 0))
        n_queries = len(queries)
        sent_embs = embs[n_queries:]
        for page, group, (start, count) in zip(pages, page_uniques, spans):
            if count == 0:
                page_kept.append([])
                continue
            query = (page.query or "").strip() or " "
            query_emb = embs[query_index[query]]
            scores = np.dot(sent_embs[start : start + count], query_emb)
            ranked = sorted(zip(group, scores), key=lambda item: float(item[1]), reverse=True)
            page_kept.append(
                [(text, float(score)) for text, score in ranked if float(score) >= min_cosine][:top_k_per_page]
            )
    else:
        for page, group in zip(pages, page_uniques):
            page_kept.append(
                select_sentences_by_cosine(
                    page.query,
                    group,
                    ranker,
                    min_cosine=min_cosine,
                    top_k=top_k_per_page,
                )
            )
    for page, kept in zip(pages, page_kept):
        for sentence, _score in kept:
            cleaned = _live_text_without_citations(sentence)
            if not cleaned or is_scraped_boilerplate(cleaned):
                continue
            selected.append(
                KnowledgeCandidate(
                    text=cleaned,
                    source="live_api",
                    url=page.url,
                    query=page.query,
                    extraction_method="sentence_relevance",
                )
            )
    logger.info(
        "sentence_relevance_elapsed elapsed=%.3fs pages=%s total_sentences=%s",
        time.monotonic() - started,
        len(pages),
        total_sentences,
    )
    return selected


def direct_sentence_candidates(
    pages: Sequence[LivePageStats],
    openie: Sequence[KnowledgeCandidate],
    ranker,
    nlp=None,
    *,
    min_cosine: float = MIN_COSINE,
    top_k_per_page: int = 8,
    relevance: Optional[Sequence[KnowledgeCandidate]] = None,
) -> Tuple[List[KnowledgeCandidate], List[KnowledgeCandidate]]:
    """SBERT-relevant live sentences: OpenIE when it works, else the fluent sentence itself."""
    if relevance is None:
        relevance = sentence_relevance_candidates(
            pages,
            ranker,
            min_cosine=min_cosine,
            top_k_per_page=top_k_per_page,
        )
    openie_by_url: Dict[str, List[KnowledgeCandidate]] = {}
    for item in openie:
        openie_by_url.setdefault(item.url or "", []).append(item)

    extra_openie: List[KnowledgeCandidate] = []
    direct: List[KnowledgeCandidate] = []
    for candidate in relevance:
        cleaned = _usable_direct_sentence(candidate.text)
        if not cleaned:
            continue
        if _openie_covers_sentence(cleaned, openie_by_url.get(candidate.url or "", [])):
            continue
        triplets = _triplets_from_cleaned_sentence(
            cleaned, url=candidate.url, query=candidate.query, nlp=nlp
        )
        if triplets:
            extra_openie.extend(triplets)
            continue
        direct.append(
            KnowledgeCandidate(
                text=cleaned,
                source="live_sentence_direct",
                url=candidate.url,
                query=candidate.query,
                extraction_method="sentence_direct",
            )
        )
    return extra_openie, direct


def _openie_covers_sentence(sentence: str, openie_for_url: Sequence[KnowledgeCandidate]) -> bool:
    key = _norm_key(sentence)
    if not key:
        return False
    for item in openie_for_url:
        other = _norm_key(item.text)
        if not other:
            continue
        if other == key or other in key or key in other:
            return True
    return False


def _triplets_from_cleaned_sentence(
    cleaned: str,
    *,
    url: Optional[str],
    query: Optional[str],
    nlp=None,
) -> List[KnowledgeCandidate]:
    raw = extract_triplets(cleaned, backend="spacy", nlp=nlp)
    resolved = resolve_coreferences(raw, cleaned, nlp=nlp, backend="heuristic")
    out: List[KnowledgeCandidate] = []
    for triplet in filter_triplets(resolved):
        text = _live_text_without_citations(triplet.as_text())
        if not text:
            continue
        out.append(
            KnowledgeCandidate(
                text=text,
                source="live_api",
                url=url,
                query=query,
                triplet=triplet,
                extraction_method="openie",
            )
        )
    return out


def fetch_live_knowledge(
    victim_utterance: str,
    config: LiveRetrievalConfig,
    budget: ApiCallBudget,
    nlp=None,
    enabled: Optional[bool] = None,
    ranker=None,
    min_cosine: float = MIN_COSINE,
    queries: Optional[List[SearchQuery]] = None,
    situations: Optional[Sequence[str]] = None,
) -> Tuple[List[KnowledgeCandidate], List[SearchQuery], List[LiveKnowledgeSentence], LiveFunnel]:
    """Run stages 0–0.9 for one turn. Returns candidates, queries, raw live sentences, funnel."""
    live_on = config.enable_live_retrieval if enabled is None else enabled
    if not live_on:
        return [], [], [], LiveFunnel()

    logger.info("live utterance=%r", victim_utterance)
    entities = extract_entities(victim_utterance, nlp=nlp)
    logger.info("live entities=%s", [e.get("text") for e in entities])
    if queries is None:
        queries = build_queries(
            entities,
            max_queries=config.max_live_queries_per_dialogue,
            victim_text=victim_utterance,
            ranker=ranker,
            situations=situations,
        )
    logger.info("live queries=%s", [q.text for q in queries])

    deadline = LiveDeadline(config.max_live_retrieval_seconds)

    def _run_query(
        query: SearchQuery,
    ) -> Tuple[List[LiveKnowledgeSentence], List[LivePageStats]]:
        if deadline.expired():
            deadline.warn_if_expired()
            return [], []
        local_pages: List[LivePageStats] = []
        results = search_allowlisted(query.text, config, budget=budget, deadline=deadline)
        logger.info(
            "live hits query=%r allowlisted=%s",
            query.text,
            [r.url for r in results],
        )
        sentences = summarize_search_results(
            query.text,
            results,
            config,
            budget=budget,
            deadline=deadline,
            page_stats=local_pages,
        )
        return sentences, local_pages

    query_batches = run_io_tasks(
        [lambda q=query: _run_query(q) for query in queries],
        max_workers=config.max_concurrent_queries,
        deadline=deadline,
    )

    all_live_sentences: List[LiveKnowledgeSentence] = []
    page_stats: List[LivePageStats] = []
    for batch in query_batches:
        if not batch:
            continue
        sentences, pages = batch
        all_live_sentences.extend(sentences)
        page_stats.extend(pages)

    openie_candidates = _triplet_candidates_from_live(all_live_sentences, nlp=nlp)
    per_page = config.live_sentence_candidates_per_page or config.live_sentence_top_k
    relevance_candidates = sentence_relevance_candidates(
        page_stats,
        ranker,
        min_cosine=min_cosine,
        top_k_per_page=per_page,
    )
    extra_openie, direct_candidates = direct_sentence_candidates(
        page_stats,
        openie_candidates,
        ranker,
        nlp=nlp,
        min_cosine=min_cosine,
        top_k_per_page=per_page,
        relevance=relevance_candidates,
    )
    openie_candidates = list(openie_candidates) + list(extra_openie)
    candidates = dedup_candidates(list(openie_candidates) + list(direct_candidates))

    funnel = LiveFunnel(
        live_sentences=sum(page.sentences_extracted for page in page_stats) or len(all_live_sentences),
        live_triplets=len(openie_candidates),
        live_sentence_relevance=len(relevance_candidates),
        live_sentences_used_directly=len(direct_candidates),
        pages=[
            _page_funnel_row(page, openie_candidates, relevance_candidates, direct_candidates)
            for page in page_stats
        ],
    )
    logger.info(
        "live_retrieval_elapsed elapsed=%.1fs queries=%d sentences=%d candidates=%d",
        deadline.elapsed(),
        len(queries),
        funnel.live_sentences,
        len(candidates),
    )
    return candidates, queries, all_live_sentences, funnel


def _page_funnel_row(
    page: LivePageStats,
    openie: Sequence[KnowledgeCandidate],
    relevance: Sequence[KnowledgeCandidate],
    direct: Sequence[KnowledgeCandidate] = (),
) -> Dict:
    openie_here = [c for c in openie if c.url == page.url]
    rel_here = [c for c in relevance if c.url == page.url]
    direct_here = [c for c in direct if c.url == page.url]
    return {
        "url": page.url,
        "query": page.query,
        "sentences_extracted": page.sentences_extracted,
        "triplets_extracted": len(openie_here),
        "sentence_relevance_candidates": len(rel_here),
        "sentences_used_directly": len(direct_here),
        "openie_texts": [c.text for c in openie_here],
        "sentence_relevance_texts": [c.text for c in rel_here],
        "direct_texts": [c.text for c in direct_here],
    }


def _triplet_candidates_from_live(
    sentences: List[LiveKnowledgeSentence],
    nlp=None,
) -> List[KnowledgeCandidate]:
    """Stage 2a–2b on live excerpts so verbalization is triplet-based, not footer text."""
    candidates: List[KnowledgeCandidate] = []
    for item in sentences:
        if is_scraped_boilerplate(item.sentence):
            logger.info("live_skip_boilerplate url=%s", item.source_url)
            continue
        cleaned = strip_legal_citations(clean_knowledge_text(item.sentence))
        if not cleaned or is_scraped_boilerplate(cleaned):
            continue
        raw = extract_triplets(cleaned, backend="spacy", nlp=nlp)
        resolved = resolve_coreferences(raw, cleaned, nlp=nlp, backend="heuristic")
        for triplet in filter_triplets(resolved):
            text = _live_text_without_citations(triplet.as_text())
            if not text:
                continue
            candidates.append(
                KnowledgeCandidate(
                    text=text,
                    source="live_api",
                    url=item.source_url,
                    query=item.query,
                    triplet=triplet,
                    extraction_method="openie",
                )
            )
    return candidates


def static_candidates_from_triplets(triplets) -> List[KnowledgeCandidate]:
    return [
        KnowledgeCandidate(
            text=t.as_text(),
            source="static_dataset",
            triplet=t,
            extraction_method="openie",
        )
        for t in triplets
    ]
