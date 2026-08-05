#!/usr/bin/env python
"""Flag dialogues with weak knowledge–dialog relevance (low top-1 SBERT score)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ktc.pipeline import KnowledgeTripletPipeline

ROLE_MAP = {"bot": "agent", "agent": "agent", "user": "victim", "victim": "victim"}
WEAK_THRESHOLD = 0.3


def load_dialogues(path: Path, max_dialogues: int | None = None) -> list[dict]:
    dialogues = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
                if max_dialogues is not None and len(dialogues) >= max_dialogues:
                    break
    return dialogues


def first_agent_history(dialogue: dict) -> str | None:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history: list[str] = []
    for utterance in utterances:
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        text = f"{role}: {utterance['utterance'].strip()}"
        if role == "agent" and history:
            return " ".join(history)
        history.append(text)
    return None


def score_bucket(score: float) -> str:
    if score < 0.15:
        return "<0.15"
    if score < 0.3:
        return "0.15-0.3"
    if score < 0.5:
        return "0.3-0.5"
    return ">0.5"


def main():
    parser = argparse.ArgumentParser(description="Report per-dialogue knowledge relevance scores.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/knowledge_quality_report.csv"),
    )
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--weak_threshold", type=float, default=WEAK_THRESHOLD)
    args = parser.parse_args()

    dialogues = load_dialogues(args.input, max_dialogues=args.max_dialogues)
    pipeline = KnowledgeTripletPipeline()

    rows: list[dict] = []
    bucket_counts: dict[str, int] = defaultdict(int)
    weak_count = 0

    for dialogue in dialogues:
        did = str(dialogue.get("dialogue_id", ""))
        knowledge = dialogue.get("knowledge", "") or ""
        history = first_agent_history(dialogue)
        if history is None:
            top1 = 0.0
        else:
            filtered = pipeline.get_filtered_triplets(knowledge)
            _, top1 = pipeline.run_with_score(knowledge, history, filtered=filtered)

        weak = top1 < args.weak_threshold
        if weak:
            weak_count += 1
        bucket = score_bucket(top1)
        bucket_counts[bucket] += 1
        rows.append(
            {
                "dialogue_id": did,
                "top1_similarity_score": f"{top1:.4f}",
                "weak_knowledge_match": weak,
                "score_bucket": bucket,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dialogue_id", "top1_similarity_score", "weak_knowledge_match", "score_bucket"],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    pct = 100.0 * weak_count / total if total else 0.0
    print(f"total_dialogues: {total}")
    print(f"weak_knowledge_match (top1 < {args.weak_threshold}): {weak_count} ({pct:.1f}%)")
    print("score_bucket distribution:")
    for bucket in ["<0.15", "0.15-0.3", "0.3-0.5", ">0.5"]:
        count = bucket_counts.get(bucket, 0)
        b_pct = 100.0 * count / total if total else 0.0
        print(f"  {bucket:8s}: {count:5d} ({b_pct:5.1f}%)")
    print(f"CSV written to {args.output}")


if __name__ == "__main__":
    main()
