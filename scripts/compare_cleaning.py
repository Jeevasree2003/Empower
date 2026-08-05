#!/usr/bin/env python
"""Compare raw OpenIE triplets on uncleaned vs cleaned knowledge text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ktc.cleaning import clean_knowledge_text
from ktc.extraction import extract_triplets


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(dialogue_id):
                return record
    raise SystemExit(f"Dialogue {dialogue_id} not found")


def get_history_at_bot_turn(dialogue: dict, turn: int) -> str:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history = []
    bot_turn = 0
    for utterance in utterances:
        role = utterance["author_role"]
        text = f"{role}: {utterance['utterance'].strip()}"
        if role in {"bot", "agent"} and history:
            if bot_turn == turn:
                return " ".join(history)
            bot_turn += 1
        history.append(text)
    raise SystemExit(f"Bot turn {turn} not found in dialogue {dialogue['dialogue_id']}")


def format_triplets(triplets) -> str:
    lines = []
    for i, t in enumerate(triplets, 1):
        lines.append(f"  {i}. ({t.head}) | {t.relation} | ({t.tail})")
    return "\n".join(lines) if lines else "  (none)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dialogue_id", required=True)
    parser.add_argument("--turn", type=int, default=8)
    args = parser.parse_args()

    dialogue = load_dialogue(args.input, args.dialogue_id)
    knowledge = dialogue["knowledge"]
    cleaned = clean_knowledge_text(knowledge)

    import spacy

    nlp = spacy.load("en_core_web_sm")
    raw_uncleaned = extract_triplets(knowledge, nlp=nlp)
    raw_cleaned = extract_triplets(cleaned, nlp=nlp)

    print(f"=== dialogue_id={args.dialogue_id} bot_turn={args.turn} ===")
    print()
    print("--- KNOWLEDGE (first 500 chars, UNCLEANED) ---")
    print(knowledge[:500])
    print()
    print("--- KNOWLEDGE (first 500 chars, CLEANED) ---")
    print(cleaned[:500])
    print()
    print(f"--- RAW TRIPLETS WITHOUT CLEANING ({len(raw_uncleaned)} total) ---")
    print(format_triplets(raw_uncleaned))
    print()
    print(f"--- RAW TRIPLETS WITH CLEANING ({len(raw_cleaned)} total) ---")
    print(format_triplets(raw_cleaned))


if __name__ == "__main__":
    main()
