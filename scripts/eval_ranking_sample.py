#!/usr/bin/env python
"""Compare bi-encoder vs cross-encoder rerank on a random dialogue sample.

Experimental evaluation harness — CrossEncoderReranker was tested here and not adopted
as the production default (see CrossEncoderReranker docstring in ktc/ranking.py:
8/25 vs 5/25 hybrid top-1 improvements, 8/25 vs 4/25 live@top-1, seed=42 n=25).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_knowledge import fetch_live_knowledge, static_candidates_from_triplets
from ktc.pipeline import KnowledgeTripletPipeline
from ktc.ranking import CrossEncoderReranker, SentenceBertRanker

ROLE_MAP = {"bot": "agent", "agent": "agent", "user": "victim", "victim": "victim"}


@dataclass
class DialogueScores:
    dialogue_id: str
    static_bi: float
    hybrid_bi: float
    static_ce: float
    hybrid_ce: float
    live_in_hybrid_top1_bi: bool
    live_in_hybrid_top1_ce: bool

    @property
    def delta_bi(self) -> float:
        return self.hybrid_bi - self.static_bi

    @property
    def delta_ce(self) -> float:
        return self.hybrid_ce - self.static_ce

    @property
    def improved_bi(self) -> bool:
        return self.delta_bi > 0

    @property
    def improved_ce(self) -> bool:
        return self.delta_ce > 0


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(dialogue_id):
                return record
    raise KeyError(f"Dialogue {dialogue_id} not found")


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
    raise ValueError(f"No agent turn in dialogue {dialogue['dialogue_id']}")


def sample_dialogue_ids(input_path: Path, n: int, seed: int) -> list[str]:
    with input_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    picked = rng.sample(records, min(n, len(records)))
    return [str(r["dialogue_id"]) for r in picked]


def evaluate_dialogue(
    dialogue_id: str,
    input_path: Path,
    pipeline: KnowledgeTripletPipeline,
    config: LiveRetrievalConfig,
    bi_ranker: SentenceBertRanker,
    ce_ranker: CrossEncoderReranker,
) -> DialogueScores:
    dialogue = load_dialogue(input_path, dialogue_id)
    victim, history = first_turn_context(dialogue)
    knowledge = dialogue.get("knowledge", "") or ""

    filtered = pipeline.get_filtered_triplets(knowledge)
    static_pool = static_candidates_from_triplets(filtered)

    pipeline.api_budget = ApiCallBudget(config.max_api_calls_per_run)
    live_candidates, _queries, _raw, _funnel = fetch_live_knowledge(
        victim, config, pipeline.api_budget, nlp=pipeline._get_nlp()
    )
    hybrid_pool = static_pool + live_candidates

    static_bi = bi_ranker.rank_candidates_with_scores(history, static_pool, top_k=1)
    hybrid_bi = bi_ranker.rank_candidates_with_scores(history, hybrid_pool, top_k=1)
    static_ce = ce_ranker.rank_candidates_with_scores(history, static_pool, top_k=1)
    hybrid_ce = ce_ranker.rank_candidates_with_scores(history, hybrid_pool, top_k=1)

    live_top1_bi = hybrid_bi.candidates[0].source == "live_api" if hybrid_bi.candidates else False
    live_top1_ce = hybrid_ce.candidates[0].source == "live_api" if hybrid_ce.candidates else False

    return DialogueScores(
        dialogue_id=dialogue_id,
        static_bi=static_bi.top1_score,
        hybrid_bi=hybrid_bi.top1_score,
        static_ce=static_ce.top1_score,
        hybrid_ce=hybrid_ce.top1_score,
        live_in_hybrid_top1_bi=live_top1_bi,
        live_in_hybrid_top1_ce=live_top1_ce,
    )


def print_table(rows: list[DialogueScores]) -> None:
    header = (
        f"{'Dialogue':>8} | {'Static':>7} {'Hybrid':>7} {'d_bi':>7} | "
        f"{'Static':>7} {'Hybrid':>7} {'d_ce':>7} | {'Impr bi':>7} {'Impr ce':>7} | live@1"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        impr_bi = "yes" if row.improved_bi else "no"
        impr_ce = "yes" if row.improved_ce else "no"
        live_tag = f"bi={int(row.live_in_hybrid_top1_bi)}/ce={int(row.live_in_hybrid_top1_ce)}"
        print(
            f"{row.dialogue_id:>8} | "
            f"{row.static_bi:7.4f} {row.hybrid_bi:7.4f} {row.delta_bi:+7.4f} | "
            f"{row.static_ce:7.4f} {row.hybrid_ce:7.4f} {row.delta_ce:+7.4f} | "
            f"{impr_bi:>7} {impr_ce:>7} | {live_tag}"
        )

    n = len(rows)
    if n == 0:
        return

    mean_delta_bi = sum(r.delta_bi for r in rows) / n
    mean_delta_ce = sum(r.delta_ce for r in rows) / n
    improved_bi = sum(1 for r in rows if r.improved_bi)
    improved_ce = sum(1 for r in rows if r.improved_ce)
    live_top1_bi = sum(1 for r in rows if r.live_in_hybrid_top1_bi)
    live_top1_ce = sum(1 for r in rows if r.live_in_hybrid_top1_ce)

    print("\n" + "=" * len(header))
    print(f"Sample size: {n}")
    print(f"Top-1 improved (hybrid > static): bi-encoder {improved_bi}/{n}, cross-encoder {improved_ce}/{n}")
    print(f"Mean hybrid-static delta: bi {mean_delta_bi:+.4f}, ce {mean_delta_ce:+.4f}")
    print(f"Live_api at hybrid top-1: bi {live_top1_bi}/{n}, ce {live_top1_ce}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval bi-encoder vs cross-encoder on random sample.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-file",
        type=Path,
        help="JSON file with dialogue_ids list; if missing, sample is written here.",
    )
    parser.add_argument("--dialogue-ids", nargs="*", help="Override sample with explicit IDs.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/eval/ranking_sample_results.json"),
        help="Write per-dialogue scores JSON for reproducibility.",
    )
    args = parser.parse_args()

    if args.dialogue_ids:
        dialogue_ids = [str(d) for d in args.dialogue_ids]
    elif args.sample_file and args.sample_file.exists():
        dialogue_ids = json.loads(args.sample_file.read_text(encoding="utf-8"))["dialogue_ids"]
    else:
        dialogue_ids = sample_dialogue_ids(args.input, args.sample_size, args.seed)
        if args.sample_file:
            args.sample_file.parent.mkdir(parents=True, exist_ok=True)
            args.sample_file.write_text(
                json.dumps({"seed": args.seed, "sample_size": args.sample_size, "dialogue_ids": dialogue_ids}, indent=2),
                encoding="utf-8",
            )

    config = LiveRetrievalConfig.load()
    pipeline = KnowledgeTripletPipeline(live_config=config)
    bi_ranker = SentenceBertRanker()
    ce_ranker = CrossEncoderReranker()

    print("=" * 80)
    print("RANKING EVAL — random sample (bi-encoder vs cross-encoder rerank)")
    print("=" * 80)
    print(f"Dialogues ({len(dialogue_ids)}): {', '.join(dialogue_ids)}")
    print(f"Columns: bi-encoder scores | cross-encoder scores | hybrid top-1 improved vs static")
    print()

    rows: list[DialogueScores] = []
    for i, did in enumerate(dialogue_ids, 1):
        print(f"[{i}/{len(dialogue_ids)}] Evaluating dialogue {did}...", flush=True)
        row = evaluate_dialogue(did, args.input, pipeline, config, bi_ranker, ce_ranker)
        rows.append(row)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(
                    [
                        {
                            "dialogue_id": r.dialogue_id,
                            "static_bi": r.static_bi,
                            "hybrid_bi": r.hybrid_bi,
                            "static_ce": r.static_ce,
                            "hybrid_ce": r.hybrid_ce,
                            "live_in_hybrid_top1_bi": r.live_in_hybrid_top1_bi,
                            "live_in_hybrid_top1_ce": r.live_in_hybrid_top1_ce,
                        }
                        for r in rows
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )

    print()
    print_table(rows)


if __name__ == "__main__":
    main()
