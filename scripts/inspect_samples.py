#!/usr/bin/env python
"""Inspect static KTC output on a few KARE knowledge passages (live retrieval off).

For hybrid static+live fetching, use scripts/inspect_ktc.py or
scripts/run_ktc_stages.py with --enable-live (off by default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ktc.pipeline import KnowledgeTripletPipeline

DEFAULT_KARE = Path(__file__).resolve().parents[2].parent / "KARE-data" / "KARE" / "Data" / "KARE.jsonl"
KEYWORDS = ("Cyber Cell", "NCW", "Section 376", "FIR")


def _history_at_first_bot_turn(dialogue: dict) -> str:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history: list[str] = []
    for utterance in utterances:
        role = utterance["author_role"]
        text = f"{role}: {utterance['utterance'].strip()}"
        if role in {"bot", "agent"} and history:
            return " ".join(history)
        history.append(text)
    return " ".join(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect static KTC on sample KARE passages.")
    parser.add_argument("--input", type=Path, default=DEFAULT_KARE, help="KARE.jsonl path")
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"KARE file not found: {args.input}")

    pipeline = KnowledgeTripletPipeline(coref_backend="heuristic")

    samples: list[dict] = []
    priority: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            knowledge = record.get("knowledge", "")
            if "Cyber Cell" in knowledge:
                priority.append(record)
            elif any(keyword in knowledge for keyword in KEYWORDS):
                samples.append(record)
            if len(priority) >= 1 and len(samples) >= max(0, args.max_samples - 1):
                break

    samples = (priority + samples)[: args.max_samples]
    if not samples:
        raise SystemExit(f"No matching dialogues in {args.input}")

    for record in samples:
        dialogue_id = record["dialogue_id"]
        knowledge = record["knowledge"]
        history = _history_at_first_bot_turn(record)

        print("=" * 80)
        print(f"dialogue_id: {dialogue_id}")
        print(f"knowledge preview: {knowledge[:220].replace(chr(10), ' ')}...")
        print(f"history preview: {history[:160]}{'...' if len(history) > 160 else ''}")

        result = pipeline.inspect(knowledge, history, enable_live=False)
        print(f"raw_triplets: {len(result['raw_triplets'])}")
        print(f"filtered_triplets: {len(result['filtered_triplets'])}")
        print("--- raw (first 12) ---")
        for triplet in result["raw_triplets"][:12]:
            print(f"  {triplet}")
        print("--- verbalized top-ranked (first 12) ---")
        for sentence in result["verbalized"][:12]:
            print(f"  {sentence}")
        print()


if __name__ == "__main__":
    main()
