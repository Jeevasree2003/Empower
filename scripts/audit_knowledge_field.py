"""Audit script: quantify how many KDPL/RDPL training rows end up as the
NO_PASSAGE_USED sentinel, and confirm the preprocess_kare.py fix is working.

Two ways to run it:

1. Against an already-written train/valid/test.json (fast, checks the fix landed):

    python scripts/audit_knowledge_field.py --data_dir data/preprocessed

2. Against raw KARE.jsonl, comparing OLD (verbalized-only) vs NEW
   (final_knowledge_text) coverage on a sample, without writing any files
   (useful before committing to a full 4999-dialogue preprocessing run):

    python scripts/audit_knowledge_field.py \\
        --input KARE-data\\\\KARE\\\\Data\\\\KARE.jsonl \\
        --sample_dialogues 100 --enable-live

Prints, per split/sample: total turns, NO_PASSAGE_USED count/%, and (in
compare mode) the delta between the old and new field choice.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

NO_PASSAGE_USED = "no_passages_used"
KNOWLEDGE_SEP = "__knowledge__"


def audit_written_split(path: Path) -> Dict[str, float]:
    total = 0
    empty = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            knowledge_field = row["knowledge"][0]
            training_part = knowledge_field.split(KNOWLEDGE_SEP, 1)[0].strip()
            if training_part == NO_PASSAGE_USED:
                empty += 1
    pct = (100.0 * empty / total) if total else 0.0
    return {"total": total, "empty": empty, "empty_pct": round(pct, 2)}


def audit_written_dir(data_dir: Path) -> None:
    print(f"Auditing {data_dir}\n" + "-" * 60)
    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.json"
        if not path.exists():
            print(f"{split:6s}: (missing) {path}")
            continue
        stats = audit_written_split(path)
        print(
            f"{split:6s}: n={stats['total']:6d}  "
            f"NO_PASSAGE_USED={stats['empty']:6d} ({stats['empty_pct']}%)"
        )
        if stats["empty_pct"] >= 40:
            print(
                f"         ^ still high. If this is post-fix, either coverage genuinely "
                f"caps here for this data, or a knowledge_mode/enable-live mismatch is in play."
            )


def audit_compare_from_source(
    input_path: Path,
    sample_dialogues: int,
    enable_live: bool,
    verbalization_backend: str,
    coref_backend: str,
    top_k: int,
    seed: int,
    samples_out: Optional[Path] = None,
    disable_context_fallback: bool = False,
) -> None:
    import random

    from ktc.live_config import LiveRetrievalConfig
    from ktc.pipeline import KnowledgeTripletPipeline
    from scripts.preprocess_kare import ROLE_MAP, format_utterance, load_dialogues

    dialogues = load_dialogues(input_path)
    rng = random.Random(seed)
    rng.shuffle(dialogues)
    dialogues = dialogues[:sample_dialogues]

    live_config = LiveRetrievalConfig.load()
    pipeline = KnowledgeTripletPipeline(
        top_k=top_k,
        verbalization_backend=verbalization_backend,
        coref_backend=coref_backend,
        live_config=live_config,
    )

    total = 0
    old_empty = 0
    new_empty = 0
    all_texts: List[str] = []
    source_counts: Counter = Counter()
    fallback_count = 0

    if samples_out is None:
        samples_out = Path("reports") / "audit_compare_100_samples.jsonl"
    samples_out.parent.mkdir(parents=True, exist_ok=True)

    with samples_out.open("w", encoding="utf-8") as sample_fh:
        for dialogue in dialogues:
            utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
            knowledge_text = dialogue.get("knowledge", "") or ""
            dialogue_id = str(dialogue.get("dialogue_id") or "")
            history: List[str] = []
            bot_turn = 0

            for utterance in utterances:
                formatted = format_utterance(utterance)
                role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])

                if role == "agent" and history:
                    dialog_history = " ".join(history)
                    result = pipeline.run_hybrid(
                        knowledge_text,
                        dialog_history,
                        enable_live=enable_live,
                        dialogue_id=dialogue_id,
                        turn=bot_turn,
                        context_fallback=False if disable_context_fallback else None,
                    )
                    bot_turn += 1
                    total += 1
                    final_text = (result.final_knowledge_text or "").strip()
                    if not result.verbalized:
                        old_empty += 1
                    if not final_text:
                        new_empty += 1
                    all_texts.append(final_text)
                    sources = list(result.final_knowledge_sources or [])
                    for src in sources:
                        source_counts[src] += 1
                    funnel = result.knowledge_funnel or {}
                    if funnel.get("context_fallback_used"):
                        fallback_count += 1
                    sample_fh.write(
                        json.dumps(
                            {
                                "dialogue_id": dialogue_id,
                                "turn_id": bot_turn - 1,
                                "final_knowledge_sources": sources,
                                "counseling_bank_used": getattr(
                                    result, "counseling_bank_used", None
                                ),
                                "context_fallback_used": bool(
                                    funnel.get("context_fallback_used")
                                ),
                                "final_knowledge_text_head": final_text[:150],
                                "final_knowledge_text_full": final_text,
                                "dialog_history_tail": dialog_history[-300:],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                history.append(formatted)

    def pct(n: int) -> float:
        return round(100.0 * n / total, 2) if total else 0.0

    unique_n = len(set(all_texts))
    text_counts = Counter(all_texts)

    print(
        f"Compare mode: {sample_dialogues} dialogues, {total} turns, "
        f"enable_live={enable_live}, context_fallback={not disable_context_fallback}"
    )
    print("-" * 60)
    print(f"OLD (verbalized only, current preprocess_kare.py bug):")
    print(f"  NO_PASSAGE_USED: {old_empty} / {total} ({pct(old_empty)}%)")
    print(f"NEW (final_knowledge_text, the fix):")
    print(f"  NO_PASSAGE_USED: {new_empty} / {total} ({pct(new_empty)}%)")
    print(f"Turns rescued by the fix: {old_empty - new_empty} ({pct(old_empty - new_empty)}pp)")
    print(f"Wrote per-turn samples: {samples_out}")
    print(f"Text diversity: unique={unique_n} vs total={len(all_texts)}")
    print(f"Source histogram (turns containing each source): {dict(source_counts)}")
    print(f"context_fallback_used turns: {fallback_count}")
    print("Top 10 repeated exact final_knowledge_text strings:")
    for i, (text, count) in enumerate(text_counts.most_common(10), start=1):
        preview = (text[:200] + "…") if len(text) > 200 else text
        print(f"  {i:2d}. n={count:4d}  {preview!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None,
        help="Audit an already-written preprocessed dir (train/valid/test.json).")
    parser.add_argument("--input", type=Path, default=None,
        help="Raw KARE.jsonl to sample and compare old vs new field choice (no files written).")
    parser.add_argument("--sample_dialogues", type=int, default=100)
    parser.add_argument("--enable-live", action="store_true", default=False)
    parser.add_argument("--verbalization_backend", choices=["template", "llm"], default="llm")
    parser.add_argument("--coref_backend", choices=["heuristic", "model"], default="heuristic")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples_out",
        type=Path,
        default=Path("reports") / "audit_compare_100_samples.jsonl",
        help="Per-turn diagnostic dump (compare mode only).",
    )
    parser.add_argument(
        "--disable_context_fallback",
        action="store_true",
        default=False,
        help="Turn off LLM context-fallback so empty gated turns stay empty (training-data audit).",
    )
    args = parser.parse_args()

    if args.data_dir is not None:
        audit_written_dir(args.data_dir)
    elif args.input is not None:
        audit_compare_from_source(
            args.input, args.sample_dialogues, args.enable_live,
            args.verbalization_backend, args.coref_backend, args.top_k, args.seed,
            samples_out=args.samples_out,
            disable_context_fallback=args.disable_context_fallback,
        )
    else:
        parser.error("Pass either --data_dir (audit written files) or --input (compare from source).")


if __name__ == "__main__":
    main()
