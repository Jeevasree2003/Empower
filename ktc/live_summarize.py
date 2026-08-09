"""Stage 0.9 — LLM summarization of live search results with source attribution."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_retrieval import SearchResult, enrich_search_result

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
        if _is_meta_commentary(line):
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
) -> List[LiveKnowledgeSentence]:
    """Summarize allowlisted search hits into factual sentences. Never raises."""
    if not results:
        return []

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        logger.warning("LLM_API_KEY not set; skipping summarization for query=%r", query)
        return []

    try:
        from openai import OpenAI  # noqa: F401 — checked by _make_llm_client
    except ImportError:
        logger.warning("openai package not installed; skipping summarization for query=%r", query)
        return []

    client = _make_llm_client(config)
    attributed: List[LiveKnowledgeSentence] = []

    for result in results[: config.results_per_query]:
        if budget is not None and not budget.can_call():
            logger.warning("API call budget exhausted; skipping remaining summarization for query=%r", query)
            break
        try:
            if budget is not None:
                budget.record(1)
            enriched = enrich_search_result(result, cache_ttl_days=config.cache_ttl_days)
            sentences = _summarize_one_source(query, enriched, config, client)
            for sentence in sentences:
                attributed.append(
                    LiveKnowledgeSentence(sentence=sentence, source_url=result.url, query=query)
                )
        except Exception as exc:
            logger.warning(
                "live_summarize_failed query=%r url=%s error=%s", query, result.url, exc
            )

    return attributed
