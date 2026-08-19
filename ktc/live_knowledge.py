"""Orchestrate live knowledge retrieval: entities → queries → search → summarize."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from ktc.entity_extraction import extract_entities
from ktc.knowledge_item import KnowledgeCandidate
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_retrieval import search_allowlisted
from ktc.live_summarize import LiveKnowledgeSentence, summarize_search_results
from ktc.query_builder import SearchQuery, build_queries

logger = logging.getLogger(__name__)


_SPEAKER_SPLIT = re.compile(
    r"\s+(?=agent:|bot:|victim:|user:|seeker:|client:)",
    re.IGNORECASE,
)
_VICTIM_PREFIXES = ("victim:", "user:", "seeker:", "client:")


def victim_utterances_from_history(dialog_history: str) -> List[str]:
    """Return victim/user utterances in order from formatted dialog history."""
    if not (dialog_history or "").strip():
        return []
    parts = _SPEAKER_SPLIT.split(dialog_history.strip())
    utterances: List[str] = []
    for part in parts:
        lowered = part.lower()
        if not lowered.startswith(_VICTIM_PREFIXES):
            continue
        text = part.split(":", 1)[-1].strip()
        if text:
            utterances.append(text)
    return utterances


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
) -> Tuple[List[KnowledgeCandidate], List[SearchQuery], List[LiveKnowledgeSentence]]:
    """Run stages 0–0.9 for one turn. Returns candidates, queries, raw live sentences."""
    if not config.enable_live_retrieval:
        return [], [], []

    logger.info("live_victim_utterance=%r", victim_utterance[:300] if victim_utterance else "")
    entities = extract_entities(victim_utterance, nlp=nlp)
    logger.info("live_entities=%s", [(e.get("text"), e.get("category")) for e in entities])
    queries = build_queries(entities, max_queries=config.max_live_queries_per_dialogue)
    logger.info("live_queries=%s", [q.text for q in queries])

    candidates: List[KnowledgeCandidate] = []
    all_live_sentences: List[LiveKnowledgeSentence] = []

    for query in queries:
        results = search_allowlisted(query.text, config, budget=budget)
        live_sentences = summarize_search_results(query.text, results, config, budget=budget)
        logger.info(
            "live_search query=%r allowlisted=%d summarized=%d",
            query.text,
            len(results),
            len(live_sentences),
        )
        all_live_sentences.extend(live_sentences)
        for item in live_sentences:
            candidates.append(
                KnowledgeCandidate(
                    text=item.sentence,
                    source="live_api",
                    url=item.source_url,
                    query=item.query,
                )
            )

    logger.info("live_candidates=%d", len(candidates))
    return candidates, queries, all_live_sentences


def static_candidates_from_triplets(triplets) -> List[KnowledgeCandidate]:
    return [
        KnowledgeCandidate(text=t.as_text(), source="static_dataset", triplet=t)
        for t in triplets
    ]
