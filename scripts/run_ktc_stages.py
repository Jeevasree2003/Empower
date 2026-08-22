#!/usr/bin/env python
"""Print KTC stages for one KARE dialogue turn (static by default).

Default output is a demo-friendly stage dump. Pass --json for the raw object.
"""

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


def _print_stage(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_demo(dialogue: dict, history: str, result: dict, turn: int) -> None:
    _print_stage("STAGE 0  Dialogue turn")
    print(f"dialogue_id: {dialogue.get('dialogue_id')}")
    print(f"bot_turn: {turn}")
    print(f"knowledge_chars: {len(dialogue.get('knowledge') or '')}")
    print(f"history:\n  {history}")
    print(f"victim_span:\n  {result.get('victim_span')}")

    _print_stage("STAGE 0.5  Entities, situation, constructed queries")
    print(f"situations: {result.get('situations')}")
    print(f"entities: {result.get('entities')}")
    print("constructed_queries (what Tavily would search with --enable-live):")
    queries = result.get("constructed_queries") or []
    if not queries:
        print("  (none)")
    for item in queries:
        print(f"  [{item.get('template')}] {item.get('query')}")
    print()
    print(result.get("query_field_note"))

    _print_stage("STAGE 1  Static KARE knowledge (gated passages + OpenIE)")
    static = result.get("static_knowledge") or {}
    print(f"no_passages_used: {static.get('no_passages_used')}")
    print(f"passages_used: {len(static.get('passages_used') or [])}")
    for i, passage in enumerate(static.get("passages_used") or [], 1):
        print(f"  P{i}: {passage[:220].replace(chr(10), ' ')}")
    print("filtered_triplets:")
    trips = static.get("filtered_triplets") or []
    if not trips:
        print("  (none — blob off-topic or gated out)")
    for trip in trips[:12]:
        print(f"  ({trip.get('head')}) -[{trip.get('relation')}]-> ({trip.get('tail')})")
    print("static_verbalized:")
    for sentence in static.get("verbalized") or []:
        print(f"  - {sentence}")

    _print_stage("STAGE 2  Live retrieval → OpenIE triplets → verbalize")
    live = result.get("live_knowledge") or {}
    print(f"live_enabled: {live.get('enabled')}")
    elapsed = live.get("elapsed_seconds")
    if elapsed is not None:
        print(f"live_retrieval_elapsed_seconds: {elapsed:.2f}")
    live_sents = live.get("verbalized") or []
    if not live.get("enabled"):
        print("  live off (default). Re-run with --enable-live to fetch allowlisted pages.")
    elif not live_sents:
        print("  live on, but no OpenIE triplets survived filtering (nav/footer dropped).")
    for sentence in live_sents:
        print(f"  - {sentence}")

    _print_stage("STAGE 3  Supplemental counseling facts (NOT Stage 2e; trigger-matched only)")
    extra = result.get("supplemental_counseling") or []
    print(f"count: {len(extra)}")
    if not extra:
        print("  (none — no crisis/crime triggers in this utterance)")
    for cand in extra:
        print(f"  [{cand.get('domain')}] {cand.get('text')}")

    _print_stage("STAGE 4  verbalized = Stage 2e KTC only")
    print(result.get("query_field_note"))
    print("verbalized:")
    sents = result.get("verbalized") or []
    if not sents:
        print("  (empty — no gated KARE/live triplets for this turn)")
    for sentence in sents:
        print(f"  - {sentence}")

    _print_stage("STAGE 5  synthesized_knowledge (LLM-3; --synthesize)")
    synthesized = result.get("synthesized_knowledge")
    if synthesized is None:
        print("  (not requested). Re-run with --synthesize to merge candidates into one passage.")
    elif not synthesized.strip():
        print("  (empty — no ranked/supplemental candidates to synthesize)")
    else:
        print(synthesized)

    _print_stage("STAGE 6  final_knowledge_text (KT for training / response generation)")
    sources = result.get("final_knowledge_sources") or []
    if not sources:
        print("sources: (none — empty-knowledge turn)")
    else:
        print("sources: " + " + ".join(sources))
    final_text = result.get("final_knowledge_text") or ""
    if not final_text.strip():
        print("  (empty)")
    else:
        print(final_text)


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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw inspect JSON instead of the demo stage dump.",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        default=False,
        help="Run LLM-3 evidence synthesis and print it next to per-candidate verbalized output.",
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
        synthesize=args.synthesize,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print_demo(dialogue, history, result, args.turn)


if __name__ == "__main__":
    main()
