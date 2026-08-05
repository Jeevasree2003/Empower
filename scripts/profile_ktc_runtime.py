#!/usr/bin/env python
"""Profile per-stage KTC runtime for one dialogue."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from ktc.coreference import resolve_coreferences
from ktc.extraction import extract_triplets
from ktc.filtering import filter_triplets
from ktc.pipeline import KnowledgeTripletPipeline
from ktc.ranking import SentenceBertRanker, rank_triplets
from ktc.verbalization import verbalize_triplets


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(dialogue_id):
                return record
    raise SystemExit(f"Dialogue {dialogue_id} not found")


def agent_turns(dialogue: dict) -> list[tuple[int, str]]:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    history: list[str] = []
    turns: list[tuple[int, str]] = []
    bot_turn = 0
    for utterance in utterances:
        role = utterance["author_role"]
        text = f"{role}: {utterance['utterance'].strip()}"
        if role in {"bot", "agent"} and history:
            turns.append((bot_turn, " ".join(history)))
            bot_turn += 1
        history.append(text)
    return turns


def profile_turn(
    knowledge_text: str,
    dialog_history: str,
    nlp,
    ranker_holder: dict,
    top_k: int = 26,
) -> dict[str, float]:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    raw = extract_triplets(knowledge_text, nlp=nlp)
    timings["extraction"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    resolved = resolve_coreferences(raw, knowledge_text, nlp=nlp)
    timings["coreference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    filtered = filter_triplets(resolved)
    timings["filtering"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if ranker_holder["ranker"] is None:
        ranker_holder["ranker"] = SentenceBertRanker()
        ranker_holder["init_time"] += time.perf_counter() - t0
        t0 = time.perf_counter()
    ranked, _top1 = rank_triplets(dialog_history, filtered, top_k=top_k, ranker=ranker_holder["ranker"])
    timings["ranking_encode"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verbalize_triplets(ranked, backend="template")
    timings["verbalization"] = time.perf_counter() - t0

    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dialogue_id", default="1")
    args = parser.parse_args()

    dialogue = load_dialogue(args.input, args.dialogue_id)
    knowledge = dialogue["knowledge"]
    turns = agent_turns(dialogue)

    import spacy

    nlp = spacy.load("en_core_web_sm")
    ranker_holder = {"ranker": None, "init_time": 0.0}

    totals = defaultdict(float)
    print(f"dialogue_id={args.dialogue_id} agent_turns={len(turns)} knowledge_chars={len(knowledge)}")
    print()

    for turn_idx, history in turns:
        timings = profile_turn(knowledge, history, nlp, ranker_holder)
        turn_total = sum(timings.values())
        print(f"--- bot_turn={turn_idx} total={turn_total:.2f}s ---")
        for stage, seconds in timings.items():
            print(f"  {stage:18s} {seconds:7.2f}s")
            totals[stage] += seconds
        print()

    dialogue_total = sum(totals.values()) + ranker_holder["init_time"]
    print(f"SBERT model load (once per pipeline if ranker=None): {ranker_holder['init_time']:.2f}s")
    print()
    print("=== DIALOGUE TOTALS ===")
    for stage in ["extraction", "coreference", "filtering", "ranking_encode", "verbalization"]:
        sec = totals[stage]
        pct = 100.0 * sec / dialogue_total if dialogue_total else 0
        print(f"  {stage:18s} {sec:8.2f}s  ({pct:5.1f}%)")
    print(f"  {'sbert_load':18s} {ranker_holder['init_time']:8.2f}s  ({100*ranker_holder['init_time']/dialogue_total:5.1f}%)")
    print(f"  {'TOTAL':18s} {dialogue_total:8.2f}s")

    # Current preprocess behavior: new ranker every turn when pipeline.ranker is None
    print()
    print("=== CURRENT preprocess_kare.py behavior (ranker=None => reload each turn) ===")
    reload_totals = defaultdict(float)
    for _turn_idx, history in turns:
        rh = {"ranker": None, "init_time": 0.0}
        timings = profile_turn(knowledge, history, nlp, rh)
        reload_totals["sbert_load"] += rh["init_time"]
        for k, v in timings.items():
            reload_totals[k] += v
    reload_total = sum(reload_totals.values())
    print(f"  agent_turns={len(turns)}  TOTAL={reload_total:.2f}s  (sbert_load={reload_totals['sbert_load']:.2f}s across turns)")


if __name__ == "__main__":
    main()
