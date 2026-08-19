#!/usr/bin/env python
"""Print KTC stages for one KARE dialogue turn (static by default)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from ktc.pipeline import KnowledgeTripletPipeline
from ktc.triplet import Triplet

ROLE_MAP = {
    "bot": "agent",
    "agent": "agent",
    "user": "victim",
    "victim": "victim",
}


def _history_for_turn(dialogue: dict, turn: int) -> str:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history = []
    bot_turn = 0
    for utterance in utterances:
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        text = f"{role}: {utterance['utterance'].strip()}"
        if role == "agent" and history:
            if bot_turn == turn:
                return " ".join(history)
            bot_turn += 1
        history.append(text)
    raise SystemExit(f"Bot turn {turn} not found")


def _preview(text: str, n: int = 280) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../../KARE-data/KARE/Data/KARE.jsonl"))
    parser.add_argument("--dialogue_id", default="100")
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument("--enable-live", action="store_true", default=False)
    parser.add_argument("--coref-backend", choices=["none", "heuristic", "model"], default="none")
    parser.add_argument("--verbalization-backend", choices=["template", "llm"], default="template")
    args = parser.parse_args()

    dialogue = None
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(args.dialogue_id):
                dialogue = record
                break
    if dialogue is None:
        raise SystemExit(f"Dialogue {args.dialogue_id} not found")

    history = _history_for_turn(dialogue, args.turn)
    knowledge = dialogue.get("knowledge", "") or ""
    print("=" * 78)
    print(f"DIALOGUE {args.dialogue_id}, turn {args.turn}")
    print("=" * 78)
    print(f"knowledge_text ({len(knowledge)} chars):")
    print(_preview(knowledge, 400))
    print()
    print("dialog_history:")
    print(_preview(history, 400))
    print()

    pipeline = KnowledgeTripletPipeline(
        coref_backend=args.coref_backend,
        verbalization_backend=args.verbalization_backend,
    )
    result = pipeline.inspect(knowledge, history, enable_live=args.enable_live)

    print("=" * 78)
    print("STAGE 1 — CLEANING")
    print("=" * 78)
    print(f"preview: {result['cleaned_knowledge_preview'][:400]}")
    print()
    print(f"ranking_query: {result.get('ranking_query')!r}")
    print(f"selected_passages: {len(result.get('selected_passages') or [])}")
    for i, (passage, score) in enumerate(
        zip(result.get("selected_passages") or [], result.get("passage_scores") or []), start=1
    ):
        print(f"  passage {i} score={score:.3f}: {_preview(passage, 160)}")
    print()

    print("=" * 78)
    print("STAGE 2 — EXTRACTION (from selected passages only)")
    print("=" * 78)
    raw = [Triplet(**t) if isinstance(t, dict) else t for t in result["raw_triplets"]]
    print(f"count: {len(raw)}")
    for trip in raw[:15]:
        print(f"  {(trip.head, trip.relation, trip.tail)}")
    if len(raw) > 15:
        print(f"  ... and {len(raw) - 15} more")
    print()

    print("=" * 78)
    print("STAGE 3 — COREFERENCE")
    print("=" * 78)
    resolved = result["resolved_triplets"]
    changed = 0
    for before, after in zip(result["raw_triplets"], resolved):
        if before != after:
            changed += 1
            if changed <= 8:
                print(f"  {before['head']!r} -> {after['head']!r}")
    print(f"triplets changed by coreference: {changed} / {len(raw)}")
    print()

    print("=" * 78)
    print("STAGE 4 — FILTERING")
    print("=" * 78)
    filtered = result["filtered_triplets"]
    print(f"kept: {len(filtered)} / {len(raw)}")
    for trip in filtered[:12]:
        print(f"  KEPT {(trip['head'], trip['relation'], trip['tail'])}")
    print()

    print("=" * 78)
    print("STAGE 5/6 — RANKING + VERBALIZATION")
    print("=" * 78)
    print(f"top1_score: {result['top1_similarity_score']:.3f}")
    print(f"ranked_candidates: {len(result['ranked_candidates'])}")
    for sentence in result["verbalized"]:
        print(f"  - {sentence}")
    if not result["verbalized"]:
        print("  (none — no_passages_used)")
    print()

    print("=" * 78)
    print("STAGE 7/8 — FULL PIPELINE")
    print("=" * 78)
    sources = Counter(c.get("source") for c in result["ranked_candidates"])
    print(f"live_retrieval_enabled: {result['live_retrieval_enabled']}")
    print(f"sources: {dict(sources)}")
    print("Done.")


if __name__ == "__main__":
    main()
