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


def victim_utterance_from_history(dialog_history: str) -> str:
    """Return the most recent victim utterance from formatted dialog history."""
    parts = re.split(r"\s+(?=agent:|victim:)", dialog_history.strip())
    victim_parts = [p for p in parts if p.startswith("victim:")]
    if not victim_parts:
        return ""
    last = victim_parts[-1]
    return last.split(":", 1)[-1].strip()


def fetch_live_knowledge(
    victim_utterance: str,
    config: LiveRetrievalConfig,
    budget: ApiCallBudget,
    nlp=None,
) -> Tuple[List[KnowledgeCandidate], List[SearchQuery], List[LiveKnowledgeSentence]]:
    """Run stages 0–0.9 for one turn. Returns candidates, queries, raw live sentences."""
    if not config.enable_live_retrieval:
        return [], [], []

    entities = extract_entities(victim_utterance, nlp=nlp)
    queries = build_queries(entities, max_queries=config.max_live_queries_per_dialogue)

    candidates: List[KnowledgeCandidate] = []
    all_live_sentences: List[LiveKnowledgeSentence] = []

    for query in queries:
        results = search_allowlisted(query.text, config, budget=budget)
        live_sentences = summarize_search_results(query.text, results, config, budget=budget)
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

    return candidates, queries, all_live_sentences


def static_candidates_from_triplets(triplets) -> List[KnowledgeCandidate]:
    return [
        KnowledgeCandidate(text=t.as_text(), source="static_dataset", triplet=t)
        for t in triplets
    ]
