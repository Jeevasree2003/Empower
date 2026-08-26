#!/usr/bin/env python
"""Aggregate per-turn KTC knowledge_funnel stats across a KARE.jsonl corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_kare import (  # noqa: E402
    ROLE_MAP,
    format_utterance,
    load_dialogues,
    process_dialogue_turn,
)
from ktc.pipeline import KnowledgeTripletPipeline  # noqa: E402


def _source_bucket(sources: Optional[List[str]]) -> str:
    labels = list(sources or [])
    if labels == ["concatenation_fallback"]:
        return "concatenation_fallback"
    if labels == ["llm_synthesis"]:
        return "llm_synthesis"
    if labels == ["verbalized"]:
        return "verbalized"
    if not labels:
        return "empty"
    return "+".join(labels)


def accumulate_turn(stats: Dict, result) -> None:
    stats["total_turns"] += 1
    if getattr(result, "no_passages_used", False):
        stats["no_passages_used"] += 1
    funnel = getattr(result, "knowledge_funnel", None) or {}
    stats["gate_passed"].append(int(funnel.get("gate_passed") or 0))
    stats["final_verbalized_count"].append(int(funnel.get("final_verbalized_count") or 0))
    bucket = _source_bucket(getattr(result, "final_knowledge_sources", None))
    stats["sources"][bucket] = stats["sources"].get(bucket, 0) + 1


def _dist(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": round(statistics.mean(values), 3),
        "max": max(values),
    }


def summarize(stats: Dict) -> Dict:
    total = stats["total_turns"] or 1
    unused = stats["no_passages_used"]
    return {
        "total_turns": stats["total_turns"],
        "no_passages_used": unused,
        "no_passages_used_pct": round(100.0 * unused / total, 2),
        "gate_passed": _dist(stats["gate_passed"]),
        "final_verbalized_count": _dist(stats["final_verbalized_count"]),
        "final_knowledge_sources": dict(sorted(stats["sources"].items())),
    }


def format_summary_table(summary: Dict) -> str:
    lines = [
        "KTC knowledge funnel report",
        f"total_turns: {summary['total_turns']}",
        f"no_passages_used: {summary['no_passages_used']} ({summary['no_passages_used_pct']}%)",
        "gate_passed min/mean/max: "
        f"{summary['gate_passed']['min']}/{summary['gate_passed']['mean']}/{summary['gate_passed']['max']}",
        "final_verbalized_count min/mean/max: "
        f"{summary['final_verbalized_count']['min']}/"
        f"{summary['final_verbalized_count']['mean']}/"
        f"{summary['final_verbalized_count']['max']}",
        "final_knowledge_sources:",
    ]
    for name, count in summary["final_knowledge_sources"].items():
        lines.append(f"  {name}: {count}")
    return "\n".join(lines) + "\n"


def iter_agent_turns(dialogue: dict) -> Iterable[tuple]:
    utterances = sorted(dialogue.get("utterances") or [], key=lambda u: int(u["utterance_no"]))
    history: List[str] = []
    bot_turn = 0
    for utterance in utterances:
        formatted = format_utterance(utterance)
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        if role == "agent" and history:
            yield bot_turn, " ".join(history)
            bot_turn += 1
        history.append(formatted)


def run_report(
    input_path: Path,
    output_dir: Path,
    max_dialogues: Optional[int] = None,
    enable_live: bool = False,
) -> Dict:
    dialogues = load_dialogues(input_path)
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]
    pipeline = KnowledgeTripletPipeline(
        verbalization_backend="template",
        coref_backend="heuristic",
    )
    stats = {
        "total_turns": 0,
        "no_passages_used": 0,
        "gate_passed": [],
        "final_verbalized_count": [],
        "sources": {},
    }
    for dialogue in dialogues:
        knowledge_text = dialogue.get("knowledge", "") or ""
        dialogue_id = str(dialogue.get("dialogue_id") or "")
        for turn, history in iter_agent_turns(dialogue):
            result, _verbalized = process_dialogue_turn(
                pipeline,
                knowledge_text,
                history,
                knowledge_mode="ktc",
                enable_live=enable_live,
                dialogue_id=dialogue_id,
                turn=turn,
            )
            if result is not None:
                accumulate_turn(stats, result)
    summary = summarize(stats)
    summary["input"] = str(input_path)
    summary["dialogues_scanned"] = len(dialogues)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"funnel_report_{stamp}.json"
    txt_path = output_dir / f"funnel_report_{stamp}.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(format_summary_table(summary), encoding="utf-8")
    summary["json_path"] = str(json_path)
    summary["txt_path"] = str(txt_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate KTC funnel stats over KARE.jsonl.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("KARE-data/KARE/Data/KARE.jsonl"),
    )
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    parser.add_argument("--enable-live", action="store_true", default=False)
    args = parser.parse_args()
    summary = run_report(args.input, args.output_dir, args.max_dialogues, args.enable_live)
    print(format_summary_table(summary))
    print(f"wrote {summary['json_path']}")
    print(f"wrote {summary['txt_path']}")


if __name__ == "__main__":
    main()
