"""Stage 0.9 — LLM summarization of live search results with source attribution."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ktc.cleaning import strip_legal_citations
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_retrieval import SearchResult, enrich_search_results

logger = logging.getLogger(__name__)

NO_RELEVANT_INFO = "NO_RELEVANT_INFO"

SUMMARIZE_SYSTEM_PROMPT = """You convert trusted web source excerpts into factual knowledge sentences for a victim-support chatbot knowledge base.

Hard rules (violations are unacceptable):
1. Only state facts explicitly present in the provided source text. Never infer, extrapolate, or add outside knowledge.
2. Output short, single-fact sentences in plain, conversational English — phrased the way a trained support agent would share information in chat (e.g. "There is a helpline number listed: 9152987821" or "The National Mental Health Programme provides care at primary health centres"), NOT encyclopedic or statistical prose (avoid "WHO estimates that...", "DALYs per 100 000 population", or "economic loss is estimated at").
3. If the source contains NO facts relevant to the query, output exactly one line: NO_RELEVANT_INFO
   If the source contains some relevant facts but not the full answer, output only the relevant facts and do NOT write NO_RELEVANT_INFO.
4. Never phrase output as advice or commands. Do not write "you should", "you must", or "contact X immediately".
   Instead write factual statements: "Section X states that...", "The helpline number listed is...", "The portal name is..."
5. Do not include URLs in the sentence text; source URLs are tracked separately.
6. WHO and government pages often use general language. Treat programme names, helpline numbers, legal definitions, portal names, and reporting options as valid facts when present in the source.
7. Never output meta-commentary about the source (e.g. "the source does not provide", "this page does not mention").
"""

_SHORT_FACT_MIN_LEN = 8
_PHONE_PATTERN = re.compile(r"\d{3,}")
_META_COMMENTARY_PATTERN = re.compile(
    r"(the source does not|this page does not|no information is provided|the text does not mention)",
    re.IGNORECASE,
)
_NON_COUNSELOR_PROSE = re.compile(
    r"policy makers|member states|\bdaly\b|economic loss|who estimates|"
    r"100\s*000 population|disability-adjusted",
    re.IGNORECASE,
)
_NAV_FOOTER = re.compile(
    r"email us at|mon\s*[-–]\s*sat|tue\s*[-–]\s*sat|\bcopyright\b|all rights reserved|"
    r"icall@tiss\.edu|subscribe|follow us|we mental health\s*&|"
    r"psychosocial support.{0,20}icall",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_QUERY_STOPWORDS = frozenset(
    {
        "how",
        "to",
        "the",
        "a",
        "an",
        "in",
        "of",
        "for",
        "and",
        "or",
        "on",
        "at",
        "is",
        "what",
        "india",
        "official",
        "current",
        "latest",
        "under",
    }
)


def _query_tokens(query: str) -> set:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", query.lower())
        if len(tok) > 2 and tok not in _QUERY_STOPWORDS
    }


def extractive_sentences(query: str, text: str, max_n: int = 3) -> List[str]:
    """Select query-overlapping sentences from source text. No LLM, no quota."""
    if not text or not text.strip():
        return []
    tokens = _query_tokens(query)
    ranked: List[tuple] = []
    for line in split_live_sentences(text):
        stoks = set(re.findall(r"[a-z0-9]+", line.lower()))
        overlap = len(tokens & stoks)
        if overlap == 0 and not _PHONE_PATTERN.search(line):
            continue
        ranked.append((overlap, -abs(180 - len(line)), line))
    ranked.sort(reverse=True)
    selected: List[str] = []
    seen = set()
    for _, _, line in ranked:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(line)
        if len(selected) >= max_n:
            break
    return selected


def split_live_sentences(text: str) -> List[str]:
    """Sentence-split live page text, dropping nav/footer and policy prose."""
    if not text or not text.strip():
        return []
    selected: List[str] = []
    seen = set()
    for raw in _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text.strip())):
        line = strip_legal_citations(raw.strip())
        if len(line) < 40 or len(line) > 420:
            continue
        if _NAV_FOOTER.search(line) or _is_broken_live_sentence(line) or _is_meta_commentary(line):
            continue
        if _NON_COUNSELOR_PROSE.search(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(line)
    return selected


@dataclass
class LiveKnowledgeSentence:
    sentence: str
    source_url: str
    query: str

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "source_url": self.source_url,
            "query": self.query,
        }


@dataclass
class LivePageStats:
    url: str
    query: str
    sentences_extracted: int
    sentences: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "query": self.query,
            "sentences_extracted": self.sentences_extracted,
            "sentences": self.sentences,
        }


def is_scraped_boilerplate(text: str) -> bool:
    return bool(_NAV_FOOTER.search(text or ""))


def _is_broken_live_sentence(line: str) -> bool:
    text = line.strip()
    if text.startswith("--") or "[...]" in text:
        return True
    if re.search(r"\bis not rape\b", text, re.I):
        return True
    if text.lower().endswith("constituting a") or text.endswith(","):
        return True
    if not re.search(r"[.!?]$", text) and len(text) > 120:
        return True
    return False


def _is_meta_commentary(line: str) -> bool:
    return bool(_META_COMMENTARY_PATTERN.search(line))


def _is_short_fact(line: str) -> bool:
    return len(line) >= _SHORT_FACT_MIN_LEN and bool(_PHONE_PATTERN.search(line))


def _parse_sentences(raw: str) -> List[str]:
    text = raw.strip()
    if not text:
        return []

    normalized = re.sub(r"\s+", "_", text.upper())
    if normalized in {NO_RELEVANT_INFO, f"{NO_RELEVANT_INFO}."}:
        return []

    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*\d.]+\s*", "", line)
        if not line:
            continue
        line_norm = line.upper().replace(" ", "_")
        if line_norm == NO_RELEVANT_INFO or line_norm.startswith(f"{NO_RELEVANT_INFO}_"):
            continue
        if _is_meta_commentary(line) or _is_broken_live_sentence(line) or is_scraped_boilerplate(line):
            continue
        if _NON_COUNSELOR_PROSE.search(line):
            continue
        line = strip_legal_citations(line)
        if not line:
            continue
        if len(line) > 20 or _is_short_fact(line):
            lines.append(line)
    return lines


def _make_llm_client(config: LiveRetrievalConfig):
    """Build an OpenAI-compatible client (OpenAI, Groq, etc.)."""
    from openai import OpenAI

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_API_BASE", "").strip() or (config.llm_api_base or "").strip()
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _summarize_one_source(
    query: str,
    result: SearchResult,
    config: LiveRetrievalConfig,
    client,
) -> List[str]:
    """Summarize a single source; returns sentence strings (no URL attribution here)."""
    user_prompt = (
        f"Search query: {query}\n\n"
        "Summarize only what this source states that helps answer the query.\n\n"
        f"Source ({result.domain}):\n"
        f"Title: {result.title}\n"
        f"Text: {result.snippet}"
    )
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    return _parse_sentences((response.choices[0].message.content or "").strip())


def summarize_search_results(
    query: str,
    results: List[SearchResult],
    config: LiveRetrievalConfig,
    budget: Optional[ApiCallBudget] = None,
    deadline=None,
    page_stats: Optional[List[LivePageStats]] = None,
) -> List[LiveKnowledgeSentence]:
    """Turn allowlisted search hits into factual sentences. Never raises.

    Default path is extractive (no Groq). ``summarize_backend: llm`` uses the API
    when a key is present and falls back to extractive on errors/429.
    """
    if not results:
        return []

    backend = (config.summarize_backend or "extractive").strip().lower()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    client = None
    use_llm = backend == "llm" and bool(api_key)
    if use_llm:
        try:
            from openai import OpenAI  # noqa: F401

            client = _make_llm_client(config)
        except Exception as exc:
            logger.warning("llm_client_unavailable; using extractive summarization error=%s", exc)
            use_llm = False
            client = None

    attributed: List[LiveKnowledgeSentence] = []
    sliced = results[: config.results_per_query]
    enriched_results = enrich_search_results(sliced, config, deadline=deadline)

    for result, enriched in zip(sliced, enriched_results):
        if deadline is not None and deadline.expired():
            break
        try:
            sentences: List[str] = []
            if use_llm and client is not None:
                if budget is not None and not budget.record(1):
                    logger.warning(
                        "API call budget exhausted; extractive fallback for query=%r", query
                    )
                    sentences = extractive_sentences(query, enriched.snippet)
                else:
                    try:
                        sentences = _summarize_one_source(query, enriched, config, client)
                    except Exception as exc:
                        logger.warning(
                            "live_summarize_failed query=%r url=%s error=%s; extractive fallback",
                            query,
                            result.url,
                            exc,
                        )
                        sentences = extractive_sentences(query, enriched.snippet)
            else:
                sentences = extractive_sentences(query, enriched.snippet)

            split_sentences = split_live_sentences(enriched.snippet)
            if page_stats is not None:
                page_stats.append(
                    LivePageStats(
                        url=result.url,
                        query=query,
                        sentences_extracted=len(split_sentences),
                        sentences=split_sentences,
                    )
                )

            for sentence in sentences:
                attributed.append(
                    LiveKnowledgeSentence(sentence=sentence, source_url=result.url, query=query)
                )
        except Exception as exc:
            logger.warning(
                "live_summarize_failed query=%r url=%s error=%s", query, result.url, exc
            )

    return attributed
