#!/usr/bin/env python
"""Inspect KTC output for a single dialogue turn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ktc.pipeline import KnowledgeTripletPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="KARE.jsonl path")
    parser.add_argument("--dialogue_id", default="1")
    parser.add_argument("--turn", type=int, default=8, help="Bot turn index to inspect")
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

    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history = []
    bot_turn = 0
    target_history = None

    for utterance in utterances:
        role = utterance["author_role"]
        text = f"{role}: {utterance['utterance'].strip()}"
        if role in {"bot", "agent"} and history:
            if bot_turn == args.turn:
                target_history = " ".join(history)
                break
            bot_turn += 1
        history.append(text)

    if target_history is None:
        raise SystemExit(f"Bot turn {args.turn} not found")

    pipeline = KnowledgeTripletPipeline()
    result = pipeline.inspect(dialogue["knowledge"], target_history)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
