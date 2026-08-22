"""Orchestrate live knowledge retrieval: entities → queries → search → OpenIE."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ktc.cleaning import clean_knowledge_text
from ktc.coreference import resolve_coreferences
from ktc.entity_extraction import extract_entities
from ktc.extraction import extract_triplets
from ktc.filtering import filter_triplets
from ktc.knowledge_item import KnowledgeCandidate
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_retrieval import LiveDeadline, run_io_tasks, search_allowlisted
from ktc.live_summarize import LiveKnowledgeSentence, is_scraped_boilerplate, summarize_search_results
from ktc.query_builder import SearchQuery, build_queries

logger = logging.getLogger(__name__)


_SPEAKER_SPLIT = re.compile(
    r"\s+(?=agent:|bot:|victim:|user:|seeker:|client:)",
    re.IGNORECASE,
)
_VICTIM_PREFIXES = ("victim:", "user:", "seeker:", "client:")


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


def fetch_live_knowledge(
    victim_utterance: str,
    config: LiveRetrievalConfig,
    budget: ApiCallBudget,
    nlp=None,
    enabled: Optional[bool] = None,
) -> Tuple[List[KnowledgeCandidate], List[SearchQuery], List[LiveKnowledgeSentence]]:
    """Run stages 0–0.9 for one turn. Returns candidates, queries, raw live sentences."""
    live_on = config.enable_live_retrieval if enabled is None else enabled
    if not live_on:
        return [], [], []

    logger.info("live utterance=%r", victim_utterance)
    entities = extract_entities(victim_utterance, nlp=nlp)
    logger.info("live entities=%s", [e.get("text") for e in entities])
    queries = build_queries(
        entities,
        max_queries=config.max_live_queries_per_dialogue,
        victim_text=victim_utterance,
    )
    logger.info("live queries=%s", [q.text for q in queries])

    deadline = LiveDeadline(config.max_live_retrieval_seconds)

    def _run_query(query: SearchQuery) -> List[LiveKnowledgeSentence]:
        if deadline.expired():
            deadline.warn_if_expired()
            return []
        results = search_allowlisted(query.text, config, budget=budget, deadline=deadline)
        logger.info(
            "live hits query=%r allowlisted=%s",
            query.text,
            [r.url for r in results],
        )
        return summarize_search_results(
            query.text, results, config, budget=budget, deadline=deadline
        )

    sentence_batches = run_io_tasks(
        [lambda q=query: _run_query(q) for query in queries],
        max_workers=config.max_concurrent_queries,
        deadline=deadline,
    )

    candidates: List[KnowledgeCandidate] = []
    all_live_sentences: List[LiveKnowledgeSentence] = []
    for batch in sentence_batches:
        if not batch:
            continue
        all_live_sentences.extend(batch)
        candidates.extend(_triplet_candidates_from_live(batch, nlp=nlp))

    logger.info(
        "live_retrieval_elapsed elapsed=%.1fs queries=%d sentences=%d candidates=%d",
        deadline.elapsed(),
        len(queries),
        len(all_live_sentences),
        len(candidates),
    )
    return candidates, queries, all_live_sentences


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
        cleaned = clean_knowledge_text(item.sentence)
        if not cleaned or is_scraped_boilerplate(cleaned):
            continue
        raw = extract_triplets(cleaned, backend="spacy", nlp=nlp)
        resolved = resolve_coreferences(raw, cleaned, nlp=nlp, backend="heuristic")
        for triplet in filter_triplets(resolved):
            candidates.append(
                KnowledgeCandidate(
                    text=triplet.as_text(),
                    source="live_api",
                    url=item.source_url,
                    query=item.query,
                    triplet=triplet,
                )
            )
    return candidates


def static_candidates_from_triplets(triplets) -> List[KnowledgeCandidate]:
    return [
        KnowledgeCandidate(text=t.as_text(), source="static_dataset", triplet=t)
        for t in triplets
    ]
