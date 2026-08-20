#!/usr/bin/env python
"""Print KTC stages for one KARE dialogue turn (static by default)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ktc.pipeline import KnowledgeTripletPipeline

ROLE_MAP = {
    "bot": "agent",
    "agent": "agent",
    "user": "victim",
    "victim": "victim",
}


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record.get("dialogue_id")) == str(dialogue_id):
                return record
    raise SystemExit(f"Dialogue {dialogue_id} not found in {path}")


def history_at_turn(dialogue: dict, turn: int) -> str:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history: list[str] = []
    bot_turn = 0
    for utterance in utterances:
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        text = f"{role}: {utterance['utterance'].strip()}"
        if role in {"bot", "agent"} and history:
            if bot_turn == turn:
                return " ".join(history)
            bot_turn += 1
        history.append(text)
    raise SystemExit(f"Bot turn {turn} not found in dialogue {dialogue.get('dialogue_id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect gated KTC stages for one turn.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("KARE-data/KARE/Data/KARE.jsonl"),
        help="KARE.jsonl path",
    )
    parser.add_argument("--dialogue_id", required=True)
    parser.add_argument("--turn", type=int, default=0, help="Bot turn index (0 = first agent reply with history)")
    parser.add_argument(
        "--enable-live",
        action="store_true",
        default=False,
        help="Force allowlisted live retrieval for this turn.",
    )
    parser.add_argument(
        "--verbalization-backend",
        choices=["template", "llm"],
        default="template",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    dialogue = load_dialogue(args.input, args.dialogue_id)
    history = history_at_turn(dialogue, args.turn)
    pipeline = KnowledgeTripletPipeline(
        verbalization_backend=args.verbalization_backend,
        coref_backend="heuristic",
    )
    result = pipeline.inspect(
        dialogue.get("knowledge", "") or "",
        history,
        enable_live=args.enable_live,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
