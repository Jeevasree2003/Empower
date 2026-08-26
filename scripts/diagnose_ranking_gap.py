#!/usr/bin/env python
"""Step 1: per-sentence cosine similarity between dialog history and live vs static candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np

from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_knowledge import fetch_live_knowledge, static_candidates_from_triplets
from ktc.pipeline import KnowledgeTripletPipeline
from ktc.ranking import SentenceBertRanker

ROLE_MAP = {"bot": "agent", "agent": "agent", "user": "victim", "victim": "victim"}


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(dialogue_id):
                return record
    raise SystemExit(f"Dialogue {dialogue_id} not found")


def first_turn_context(dialogue: dict) -> tuple[str, str]:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history: list[str] = []
    victim_text = ""
    for utterance in utterances:
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        text = utterance["utterance"].strip()
        if role == "victim":
            victim_text = text
        if role == "agent" and history:
            return victim_text, " ".join(history)
        history.append(f"{role}: {text}")
    raise SystemExit(f"No agent turn in dialogue {dialogue['dialogue_id']}")


def score_texts(ranker: SentenceBertRanker, history: str, texts: list[str]) -> list[float]:
    history_emb = ranker.model.encode(history.strip() or " ", normalize_embeddings=True)
    if not texts:
        return []
    embs = ranker.model.encode(texts, normalize_embeddings=True)
    return [float(np.dot(emb, history_emb)) for emb in embs]


def diagnose_dialogue(
    dialogue_id: str,
    input_path: Path,
    pipeline: KnowledgeTripletPipeline,
    ranker: SentenceBertRanker,
    config: LiveRetrievalConfig,
) -> None:
    dialogue = load_dialogue(input_path, dialogue_id)
    victim, history = first_turn_context(dialogue)
    knowledge = dialogue.get("knowledge", "") or ""

    filtered = pipeline.get_filtered_triplets(knowledge)
    static_pool = static_candidates_from_triplets(filtered)
    budget = ApiCallBudget(config.max_api_calls_per_run)
    live_candidates, queries, _raw, _funnel = fetch_live_knowledge(victim, config, budget, nlp=pipeline._get_nlp())

    static_result = ranker.rank_candidates_with_scores(history, static_pool, top_k=len(static_pool))
    hybrid_pool = static_pool + live_candidates
    hybrid_result = ranker.rank_candidates_with_scores(history, hybrid_pool, top_k=len(hybrid_pool))

    live_texts = [c.text for c in live_candidates]
    live_scores = score_texts(ranker, history, live_texts)

    top_static = static_result.candidates[:3]
    top_static_scores = static_result.scores[:3]

    print("\n" + "=" * 80)
    print(f"DIALOGUE {dialogue_id}")
    print("=" * 80)
    print(f"Victim: {victim!r}")
    print(f"History: {history!r}")
    print(f"\nQueries built ({len(queries)}):")
    for q in queries:
        print(f"  - [{q.template}] {q.text}")

    print(f"\n--- LIVE_API sentences ({len(live_candidates)}) ---")
    for i, (c, score) in enumerate(zip(live_candidates, live_scores), 1):
        print(f"\n  [{i}] cosine={score:.4f}")
        print(f"      {c.text}")

    print(f"\n--- TOP 3 STATIC_DATASET (outranking live in hybrid pool) ---")
    for i, (c, score) in enumerate(zip(top_static, top_static_scores), 1):
        print(f"\n  [{i}] cosine={score:.4f}")
        print(f"      {c.text}")

    if live_scores and top_static_scores:
        best_live = max(live_scores)
        best_static = top_static_scores[0]
        print(f"\n--- COMPARISON ---")
        print(f"  Best live score:    {best_live:.4f}")
        print(f"  Top static score:   {best_static:.4f}")
        print(f"  Gap (static-live):  {best_static - best_live:+.4f}")

    print(f"\n--- HYBRID TOP-5 (with scores) ---")
    for i, (c, score) in enumerate(zip(hybrid_result.candidates[:5], hybrid_result.scores[:5]), 1):
        print(f"  {i}. [{c.source}] {score:.4f} | {c.text[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dialogue_ids", nargs="+", default=["1", "100", "1000"])
    args = parser.parse_args()

    config = LiveRetrievalConfig.load()
    pipeline = KnowledgeTripletPipeline(live_config=config)
    ranker = SentenceBertRanker()

    for did in args.dialogue_ids:
        diagnose_dialogue(did, args.input, pipeline, ranker, config)


if __name__ == "__main__":
    main()
