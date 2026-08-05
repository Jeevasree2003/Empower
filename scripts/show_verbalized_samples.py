#!/usr/bin/env python
"""Print human-readable verbalized knowledge samples for selected dialogue turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ktc.pipeline import KnowledgeTripletPipeline

ROLE_MAP = {"bot": "agent", "agent": "agent", "user": "victim", "victim": "victim"}


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(dialogue_id):
                return record
    raise SystemExit(f"Dialogue {dialogue_id} not found")


def get_turn_context(dialogue: dict, bot_turn: int | None = None):
    """Return (bot_turn_index, victim_utterance, agent_response, dialog_history)."""
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history: list[str] = []
    current_bot_turn = 0
    last_victim_text = None

    for utterance in utterances:
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        text = utterance["utterance"].strip()

        if role == "victim":
            last_victim_text = text

        if role == "agent" and history:
            if bot_turn is None or current_bot_turn == bot_turn:
                return current_bot_turn, last_victim_text, text, " ".join(history)
            current_bot_turn += 1

        history.append(f"{role}: {text}")

    raise SystemExit(f"Bot turn {bot_turn} not found in dialogue {dialogue['dialogue_id']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["1:8", "100:", "1000:", "3000:", "4500:"],
        help="dialogue_id:turn (omit turn for first bot turn with history)",
    )
    args = parser.parse_args()

    pipeline = KnowledgeTripletPipeline()
    divider = "=" * 72

    for spec in args.samples:
        if ":" in spec:
            did, turn_str = spec.split(":", 1)
            bot_turn = int(turn_str) if turn_str else None
        else:
            did, bot_turn = spec, None

        dialogue = load_dialogue(args.input, did)
        turn_idx, victim_text, agent_response, history = get_turn_context(dialogue, bot_turn)
        filtered = pipeline.get_filtered_triplets(dialogue.get("knowledge", "") or "")
        verbalized = pipeline.run(dialogue.get("knowledge", "") or "", history, filtered=filtered)

        print(divider)
        print(f"DIALOGUE {did}  |  BOT TURN {turn_idx}")
        print(divider)
        print()
        print("VICTIM (triggering utterance):")
        print(f'  "{victim_text}"')
        print()
        print("VERBALIZED KNOWLEDGE (training input):")
        if verbalized:
            for i, sent in enumerate(verbalized, 1):
                print(f"  {i}. {sent}")
        else:
            print("  (none — would use no_passages_used)")
        print()
        print("AGENT RESPONSE (gold label):")
        print(f'  "{agent_response}"')
        print()


if __name__ == "__main__":
    main()
