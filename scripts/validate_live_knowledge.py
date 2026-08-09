#!/usr/bin/env python
"""Validate hybrid live knowledge pipeline on a small dialogue sample."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import logging
import os
from pathlib import Path

from ktc.entity_extraction import extract_entities
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_knowledge import fetch_live_knowledge, victim_utterance_from_history
from ktc.pipeline import KnowledgeTripletPipeline
from ktc.query_builder import build_queries

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROLE_MAP = {"bot": "agent", "agent": "agent", "user": "victim", "victim": "victim"}


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(dialogue_id):
                return record
    raise SystemExit(f"Dialogue {dialogue_id} not found")


def first_turn_context(dialogue: dict) -> tuple[str, str, str]:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history: list[str] = []
    victim_text = ""
    agent_response = ""
    for utterance in utterances:
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        text = utterance["utterance"].strip()
        if role == "victim":
            victim_text = text
        if role == "agent" and history:
            agent_response = text
            return victim_text, " ".join(history), agent_response
        history.append(f"{role}: {text}")
    raise SystemExit(f"No agent turn in dialogue {dialogue['dialogue_id']}")


def stage_entities(dialogue_ids: list[str], input_path: Path, nlp) -> None:
    print("=" * 72)
    print("STAGE 0 — ENTITY EXTRACTION")
    print("=" * 72)
    for did in dialogue_ids:
        dialogue = load_dialogue(input_path, did)
        victim, history, _agent = first_turn_context(dialogue)
        entities = extract_entities(victim, nlp=nlp)
        print(f"\nDialogue {did}")
        print(f"  Victim utterance: {victim!r}")
        if entities:
            for ent in entities:
                print(f"    - [{ent['category']}] {ent['text']}")
        else:
            print("    (no entities extracted)")


def stage_queries(dialogue_ids: list[str], input_path: Path, config: LiveRetrievalConfig, nlp) -> None:
    print("\n" + "=" * 72)
    print("STAGE 0.5 — QUERY CONSTRUCTION")
    print("=" * 72)
    for did in dialogue_ids:
        dialogue = load_dialogue(input_path, did)
        victim, _history, _agent = first_turn_context(dialogue)
        entities = extract_entities(victim, nlp=nlp)
        queries = build_queries(entities, max_queries=config.max_live_queries_per_dialogue)
        print(f"\nDialogue {did}")
        if not queries:
            print("  (no queries built)")
            continue
        for i, q in enumerate(queries, 1):
            print(f"  {i}. [{q.entity_category}|{q.template}] {q.text}")


def stage_live_retrieval(dialogue_ids: list[str], input_path: Path, config: LiveRetrievalConfig, nlp) -> None:
    print("\n" + "=" * 72)
    print("STAGE 0.75–0.9 — LIVE RETRIEVAL + SUMMARIZATION")
    print("=" * 72)

    if not os.environ.get("LIVE_SEARCH_API_KEY"):
        print("\nSKIPPED: LIVE_SEARCH_API_KEY not set.")
        print("Set LIVE_SEARCH_API_KEY and LLM_API_KEY to run live retrieval tests.")
        return
    if not os.environ.get("LLM_API_KEY"):
        print("\nSKIPPED: LLM_API_KEY not set.")
        return

    budget = ApiCallBudget(config.max_api_calls_per_run)
    for did in dialogue_ids:
        dialogue = load_dialogue(input_path, did)
        victim, _history, _agent = first_turn_context(dialogue)
        print(f"\nDialogue {did} — victim: {victim!r}")
        entities = extract_entities(victim, nlp=nlp)
        queries = build_queries(entities, max_queries=config.max_live_queries_per_dialogue)
        for query in queries:
            from ktc.live_retrieval import search_allowlisted
            from ktc.live_summarize import summarize_search_results

            results = search_allowlisted(query.text, config, budget=budget)
            if not results:
                print(f"  Query: {query.text}")
                print("    → (no allowlisted results)")
                continue
            summaries = summarize_search_results(query.text, results, config, budget=budget)
            for result in results[:3]:
                print(f"  Query: {query.text}")
                print(f"    Source: {result.url}")
            for s in summaries:
                print(f"    Summary: {s.sentence}")


def stage_merged_turn(dialogue_id: str, input_path: Path, pipeline: KnowledgeTripletPipeline) -> None:
    print("\n" + "=" * 72)
    print(f"STAGE 1 — MERGED STATIC+LIVE OUTPUT (dialogue {dialogue_id}, turn 0)")
    print("=" * 72)

    dialogue = load_dialogue(input_path, dialogue_id)
    victim, history, agent_response = first_turn_context(dialogue)
    knowledge = dialogue.get("knowledge", "") or ""

    static_only = pipeline.run_hybrid(knowledge, history, enable_live=False)
    hybrid = pipeline.run_hybrid(knowledge, history, enable_live=True)

    print(f"\nVictim: {victim!r}")
    print(f"Agent (gold): {agent_response!r}")
    print(f"\nTop-1 similarity — static only: {static_only.top1_similarity_score:.4f}")
    print(f"Top-1 similarity — hybrid:      {hybrid.top1_similarity_score:.4f}")

    print("\n--- STATIC ONLY (top 8) ---")
    for i, c in enumerate(static_only.ranked_candidates[:8], 1):
        print(f"  {i}. [{c.source}] {c.text[:100]}")

    print("\n--- HYBRID (top 12, source tags) ---")
    for i, c in enumerate(hybrid.ranked_candidates[:12], 1):
        url = f" url={c.url}" if c.url else ""
        print(f"  {i}. [{c.source}]{url} {c.text[:100]}")

    print("\n--- VERBALIZED HYBRID (training-style) ---")
    for i, sent in enumerate(hybrid.verbalized[:10], 1):
        print(f"  {i}. {sent}")


def main():
    parser = argparse.ArgumentParser(description="Validate hybrid live knowledge on a small sample.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--dialogue_ids",
        nargs="+",
        default=["1", "100", "1000", "3000", "4500"],
    )
    parser.add_argument(
        "--merge_dialogue_id",
        default="3000",
        help="Dialogue for full merged static+live comparison (default: 3000 DV case)",
    )
    parser.add_argument("--stage", choices=["all", "entities", "queries", "live", "merge"], default="all")
    args = parser.parse_args()

    config = LiveRetrievalConfig.load()
    import spacy

    nlp = spacy.load("en_core_web_sm")
    pipeline = KnowledgeTripletPipeline(live_config=config)

    if args.stage in {"all", "entities"}:
        stage_entities(args.dialogue_ids, args.input, nlp)
    if args.stage in {"all", "queries"}:
        stage_queries(args.dialogue_ids, args.input, config, nlp)
    if args.stage in {"all", "live"}:
        stage_live_retrieval(args.dialogue_ids, args.input, config, nlp)
    if args.stage in {"all", "merge"}:
        stage_merged_turn(args.merge_dialogue_id, args.input, pipeline)

    print("\n" + "=" * 72)
    print("COST ESTIMATE (at default config limits)")
    print("=" * 72)
    max_q = config.max_live_queries_per_dialogue
    print(f"  max queries/dialogue: {max_q}")
    print(f"  ~{max_q} search calls + ~{max_q} LLM summarize calls per dialogue (when live enabled)")
    print(f"  configured estimate: ${config.estimated_cost_per_dialogue_usd:.3f} USD / dialogue")
    print(f"  full 4999 dialogues (if all live): ~${4999 * config.estimated_cost_per_dialogue_usd:.2f} USD upper bound")


if __name__ == "__main__":
    main()
