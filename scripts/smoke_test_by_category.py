#!/usr/bin/env python
"""Corpus-level KTC smoke test: one representative dialogue per paper crime category."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ktc.pipeline import KnowledgeTripletPipeline  # noqa: E402

# Paper Section III-A crime types (16). Records have no category field; infer from text.
PAPER_CRIME_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "domestic_abuse": (r"\bdomestic (?:violence|abuse)\b", r"\b498a\b", r"\bpwdva\b", r"\bin-laws beat"),
    "sexual_assault": (r"\bsexual assault\b", r"\bmolested\b", r"\bipc\s*354\b"),
    "acid_attacks": (r"\bacid attack\b", r"\bthrew acid\b"),
    "stalking": (r"\bstalking\b", r"\bcyberstalk\w*\b", r"\bipc\s*354d\b"),
    "workplace_harassment": (r"\bworkplace\b.*\bharass", r"\bposh\b", r"\binternal complaints committee\b"),
    "online_harassment": (r"\bonline harass\w*\b", r"\bobscene (?:content|messages?)\b", r"\bit act\s*67a\b"),
    "identity_theft": (r"\bidentity theft\b", r"\bimpersonat\w+\b", r"\baadhaar\b"),
    "online_bullying": (r"\bonline bully\w*\b", r"\bcyberbully\w*\b"),
    "matrimonial_fraud": (
        r"\bmatrimonial (?:fraud|scam)\b",
        r"\bnri groom\b",
        r"\bfraudulent marriage\b",
        r"\bcheated by a groom",
        r"\bmatrimonial site\b.*\b(?:cheat|con|fraud|scam)",
        r"\bnri\b.*\b(?:cheat|con|fraud|scam)",
    ),
    "financial_scams": (r"\bfinancial scam\b", r"\bupi (?:fraud|scam)\b", r"\binvestment fraud\b"),
    "child_exploitation": (r"\bchild (?:sex|sexual|exploit)", r"\bpocso\b", r"\bchildline\b"),
    "trafficking": (r"\btraffick\w*\b", r"\bimmoral traffic\b"),
    "intimate_content_sharing": (
        r"\bintimate (?:photos?|images?|videos?|content)\b",
        r"\brevenge porn\b",
        r"\bnon-consensual intimate\b",
    ),
    "exposing_personal_info": (r"\bdoxx\w*\b", r"\bpersonal (?:information|data) (?:leak|online)\b", r"\bpublished my (?:phone|address)\b"),
    "social_exclusion": (
        r"\bsocial(?:ly)? exclud\w*\b",
        r"\bostraci[sz]\w*\b",
        r"\bcommunity boycott\b",
        r"\bshunned me\b",
        r"\bentire community has been keeping me secluded",
        r"\bcommunity has (?:been )?(?:keeping me )?secluded",
    ),
    "rape": (r"\braped?\b", r"\bipc\s*376\b", r"\bgang\s*rape\b"),
}


def _dialogue_blob(record: dict) -> str:
    parts = [str(record.get("knowledge") or "")]
    for utterance in record.get("utterances") or []:
        parts.append(str(utterance.get("utterance") or ""))
    return " ".join(parts).lower()


def infer_categories(record: dict) -> List[str]:
    blob = _dialogue_blob(record)
    hits = []
    for name, patterns in PAPER_CRIME_CATEGORIES.items():
        if any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in patterns):
            hits.append(name)
    return hits


def first_agent_history(record: dict) -> Optional[str]:
    utterances = sorted(record.get("utterances") or [], key=lambda u: int(u["utterance_no"]))
    history: List[str] = []
    for utterance in utterances:
        role = utterance.get("author_role")
        text = f"{role}: {str(utterance.get('utterance') or '').strip()}"
        if role in {"bot", "agent"} and history:
            return " ".join(history)
        history.append(text)
    return None


def pick_representatives(path: Path) -> Dict[str, dict]:
    chosen: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if first_agent_history(record) is None:
                continue
            for category in infer_categories(record):
                if category not in chosen:
                    chosen[category] = record
            if len(chosen) == len(PAPER_CRIME_CATEGORIES):
                break
    return chosen


def run_category_smoke(
    input_path: Path,
    pipeline: Optional[KnowledgeTripletPipeline] = None,
) -> Dict[str, object]:
    chosen = pick_representatives(input_path)
    if pipeline is None:
        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
        )
    failures: List[str] = []
    results: Dict[str, dict] = {}
    for category in PAPER_CRIME_CATEGORIES:
        record = chosen.get(category)
        if record is None:
            msg = f"FAIL category={category} dialogue_id=<none> reason=no_matching_dialogue"
            print(msg)
            failures.append(msg)
            continue
        dialogue_id = str(record.get("dialogue_id") or "")
        history = first_agent_history(record) or ""
        result = pipeline.run_hybrid(
            record.get("knowledge") or "",
            history,
            enable_live=False,
            dialogue_id=dialogue_id,
            turn=0,
        )
        text_ok = bool((result.final_knowledge_text or "").strip())
        sources_ok = bool(result.final_knowledge_sources)
        payload = {
            "dialogue_id": dialogue_id,
            "final_knowledge_text_nonempty": text_ok,
            "final_knowledge_sources": list(result.final_knowledge_sources or []),
        }
        results[category] = payload
        if not text_ok or not sources_ok:
            msg = (
                f"FAIL category={category} dialogue_id={dialogue_id} "
                f"text_ok={text_ok} sources={payload['final_knowledge_sources']}"
            )
            print(msg)
            failures.append(msg)
        else:
            print(f"PASS category={category} dialogue_id={dialogue_id}")
    return {
        "failures": failures,
        "results": results,
        "missing_categories": [name for name in PAPER_CRIME_CATEGORIES if name not in chosen],
        "ok": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static KTC on one dialogue per crime category.")
    parser.add_argument("--input", type=Path, default=Path("KARE-data/KARE/Data/KARE.jsonl"))
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    summary = run_category_smoke(args.input)
    if not summary["ok"]:
        print("category smoke failures:")
        for line in summary["failures"]:
            print(f"  {line}")
        raise SystemExit(1)
    print("all categories passed")


if __name__ == "__main__":
    main()
