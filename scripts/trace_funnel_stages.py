#!/usr/bin/env python
"""Diagnose why the KTC cosine gate passes so few static triplets.

Read-only instrumentation: monkeypatches ``ktc.pipeline.rank_candidates`` for
this process only. Does not change production gate thresholds or logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ktc.entity_extraction import extract_entities, has_confident_entities  # noqa: E402
from ktc.knowledge_item import KnowledgeCandidate  # noqa: E402
from ktc.live_knowledge import victim_utterances_from_history  # noqa: E402
from ktc.live_summarize import is_scraped_boilerplate  # noqa: E402
from ktc.pipeline import KnowledgeTripletPipeline  # noqa: E402
from ktc.ranking import MIN_COSINE, apply_score_gate  # noqa: E402
from ktc.reply_knowledge import _ANECDOTE, _NOT_VICTIM_FACING  # noqa: E402
from scripts.aggregate_funnel_report import iter_agent_turns  # noqa: E402
from scripts.preprocess_kare import load_dialogues  # noqa: E402
from scripts.smoke_test_by_category import infer_categories  # noqa: E402

logger = logging.getLogger("ktc.trace_funnel")

_CAPTURE: Dict[str, Any] = {}


def _preview_candidate(candidate: KnowledgeCandidate, score: Optional[float] = None) -> Dict[str, Any]:
    payload = {
        "text": (candidate.text or "")[:240],
        "source": candidate.source,
        "domain": candidate.domain,
    }
    if score is not None:
        payload["score"] = round(float(score), 4)
    return payload


def _ktc_usable_reject_reason(candidate: KnowledgeCandidate) -> Optional[str]:
    """Mirror ``is_ktc_usable`` rejection causes without changing production code."""
    if candidate.source == "counseling_bank":
        return "counseling_bank_source"
    text = (candidate.text or "").strip()
    if len(text) < 12:
        return "text_too_short"
    if is_scraped_boilerplate(text):
        return "scraped_boilerplate"
    if _NOT_VICTIM_FACING.search(text):
        return "not_victim_facing"
    if _ANECDOTE.search(text):
        return "anecdote"
    return None


def _tracing_rank_candidates(
    dialog_history: str,
    candidates: Iterable[KnowledgeCandidate],
    top_k: int = 16,
    ranker=None,
    min_cosine: float = MIN_COSINE,
):
    from ktc.ranking import SentenceBertRanker

    if ranker is None:
        ranker = SentenceBertRanker()
    candidate_list = list(candidates)
    result = ranker.rank_candidates_with_scores(
        dialog_history, candidate_list, top_k=max(len(candidate_list), 1)
    )
    per_candidate: List[Dict[str, Any]] = []
    for candidate, score in zip(result.candidates, result.scores):
        cosine_pass = float(score) >= float(min_cosine)
        usable_reason = _ktc_usable_reject_reason(candidate)
        if not cosine_pass:
            gate_reason = f"cosine {float(score):.4f} < min_cosine {float(min_cosine)}"
            logger.debug(
                "gate_reject reason=below_min_cosine score=%.4f threshold=%.4f text=%r",
                float(score),
                float(min_cosine),
                (candidate.text or "")[:160],
            )
        elif usable_reason:
            gate_reason = f"post_gate_filter:{usable_reason}"
            logger.debug(
                "gate_reject reason=%s score=%.4f text=%r",
                usable_reason,
                float(score),
                (candidate.text or "")[:160],
            )
        else:
            gate_reason = None
        per_candidate.append(
            {
                "text": (candidate.text or "")[:240],
                "source": candidate.source,
                "score": round(float(score), 4),
                "cosine_pass": cosine_pass,
                "usable_pass": usable_reason is None,
                "gate_passed": cosine_pass and usable_reason is None,
                "reject_reason": gate_reason,
            }
        )
    kept, _scores, top1 = apply_score_gate(
        result.candidates, result.scores, min_cosine=min_cosine, top_k=top_k
    )
    _CAPTURE["pool"] = candidate_list
    _CAPTURE["ranked_with_scores"] = list(zip(result.candidates, result.scores))
    _CAPTURE["per_candidate"] = per_candidate
    _CAPTURE["cosine_kept"] = kept
    _CAPTURE["top1"] = float(top1 or 0.0)
    _CAPTURE["min_cosine"] = float(min_cosine)
    return kept, top1


def _install_rank_tracer() -> None:
    import ktc.pipeline as pipeline_mod

    pipeline_mod.rank_candidates = _tracing_rank_candidates  # type: ignore[assignment]


def entity_group(entities: Sequence[Dict[str, Any]]) -> str:
    sources = {str(item.get("source") or "") for item in entities}
    if not entities:
        return "no_entities"
    has_lex = "lexicon" in sources
    has_weak = bool(sources & {"spacy_ner", "noun_chunk"})
    if has_lex and not has_weak:
        return "lexicon_only"
    if has_weak and not has_lex:
        return "ner_or_chunk_no_lexicon"
    if has_lex and has_weak:
        return "lexicon_plus_ner_or_chunk"
    return "other"


def _serialize_entities(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in entities:
        out.append(
            {
                "text": item.get("text"),
                "source": item.get("source"),
                "category": item.get("category"),
            }
        )
    return out


def trace_turn(
    pipeline: KnowledgeTripletPipeline,
    *,
    dialogue_id: str,
    turn_id: int,
    category: str,
    knowledge_text: str,
    history: str,
) -> Dict[str, Any]:
    _CAPTURE.clear()
    victim_turns = victim_utterances_from_history(history)
    victim_span = " ".join(victim_turns[-2:])
    span_entities = extract_entities(victim_span, nlp=pipeline._get_nlp())
    used_history_fallback = (not has_confident_entities(span_entities)) and bool(history.strip())

    result = pipeline.run_hybrid(
        knowledge_text,
        history,
        enable_live=False,
        dialogue_id=str(dialogue_id),
        turn=turn_id,
    )

    pool: List[KnowledgeCandidate] = list(_CAPTURE.get("pool") or [])
    ranked_pairs: List[Tuple[KnowledgeCandidate, float]] = list(_CAPTURE.get("ranked_with_scores") or [])
    per_candidate: List[Dict[str, Any]] = list(_CAPTURE.get("per_candidate") or [])
    min_cosine = float(_CAPTURE.get("min_cosine") or MIN_COSINE)

    cosine_pass_count = sum(1 for row in per_candidate if row.get("cosine_pass"))
    gate_pass_count = sum(1 for row in per_candidate if row.get("gate_passed"))
    top5 = [
        {"text": (c.text or "")[:240], "source": c.source, "score": round(float(s), 4)}
        for c, s in ranked_pairs[:5]
    ]
    top_ranked = None
    if ranked_pairs:
        cand, score = ranked_pairs[0]
        top_ranked = {
            **_preview_candidate(cand, score),
            "reject_reason": (per_candidate[0].get("reject_reason") if per_candidate else None),
        }

    funnel = result.knowledge_funnel or {}
    sources = list(result.final_knowledge_sources or [])
    record = {
        "dialogue_id": str(dialogue_id),
        "turn_id": int(turn_id),
        "category": category,
        "entities": _serialize_entities(result.entities or []),
        "entity_group": entity_group(result.entities or []),
        "used_history_fallback": bool(used_history_fallback),
        "span_entity_count": len(span_entities),
        "ranking_query": (result.ranking_query or "")[:400],
        "no_passages_used": bool(result.no_passages_used),
        "passages_used_count": len(result.passages_used or []),
        "static_triplets": int(funnel.get("static_triplets") or 0),
        "candidates_retrieved": len(pool),
        "candidates_retrieved_preview": [_preview_candidate(c) for c in pool[:5]],
        "candidates_after_ranking": len(ranked_pairs),
        "candidates_after_ranking_top5_scores": top5,
        "gate_input_count": len(per_candidate),
        "gate_passed_count": gate_pass_count,
        "cosine_passed_count": cosine_pass_count,
        "funnel_gate_passed": funnel.get("gate_passed"),
        "min_cosine": min_cosine,
        "gate_passed": per_candidate,
        "top1_similarity_score": round(float(result.top1_similarity_score or 0.0), 4),
        "top_ranked": top_ranked,
        "final_knowledge_sources": sources,
        "counseling_bank_used": int(result.counseling_bank_used or 0),
        "final_verbalized_count": int(funnel.get("final_verbalized_count") or 0),
    }
    return record


def collect_turns(dialogues: Sequence[dict]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for dialogue in dialogues:
        cats = infer_categories(dialogue)
        category = cats[0] if cats else "uncategorized"
        dialogue_id = str(dialogue.get("dialogue_id") or "")
        knowledge_text = dialogue.get("knowledge", "") or ""
        for turn_id, history in iter_agent_turns(dialogue):
            turns.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_id": int(turn_id),
                    "category": category,
                    "knowledge_text": knowledge_text,
                    "history": history,
                }
            )
    return turns


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.mean(values)), 3)


def crosstab(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = [
        ("lexicon_only", lambda r: r["entity_group"] == "lexicon_only"),
        (
            "ner_or_chunk_no_lexicon",
            lambda r: r["entity_group"] == "ner_or_chunk_no_lexicon",
        ),
        ("history_fallback", lambda r: r["used_history_fallback"]),
        ("no_history_fallback", lambda r: not r["used_history_fallback"]),
    ]
    rows = []
    for name, pred in groups:
        subset = [r for r in records if pred(r)]
        n = len(subset)
        if n == 0:
            rows.append(
                {
                    "group": name,
                    "n": 0,
                    "gate_pass_rate": None,
                    "mean_candidates_retrieved": 0.0,
                    "mean_candidates_after_ranking": 0.0,
                    "turns_with_gate_pass": 0,
                }
            )
            continue
        passed = sum(1 for r in subset if int(r.get("gate_passed_count") or 0) > 0)
        rows.append(
            {
                "group": name,
                "n": n,
                "gate_pass_rate": round(100.0 * passed / n, 1),
                "mean_candidates_retrieved": _mean([r["candidates_retrieved"] for r in subset]),
                "mean_candidates_after_ranking": _mean(
                    [r["candidates_after_ranking"] for r in subset]
                ),
                "turns_with_gate_pass": passed,
            }
        )
    return rows


def format_crosstab(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f"{'group':<26} {'n':>4} {'gate_pass%':>11} {'mean_retr':>10} {'mean_ranked':>12} {'n_pass':>7}",
        "-" * 74,
    ]
    for row in rows:
        rate = "n/a" if row["gate_pass_rate"] is None else f"{row['gate_pass_rate']:.1f}"
        lines.append(
            f"{row['group']:<26} {row['n']:>4} {rate:>11} "
            f"{row['mean_candidates_retrieved']:>10.3f} "
            f"{row['mean_candidates_after_ranking']:>12.3f} "
            f"{row['turns_with_gate_pass']:>7}"
        )
    return "\n".join(lines)


def empty_bucket_kind(record: Dict[str, Any]) -> Optional[str]:
    sources = list(record.get("final_knowledge_sources") or [])
    if sources == ["supplemental_counseling"]:
        return "supplemental_counseling"
    if not sources:
        return "empty"
    return None


def format_empty_case(record: Dict[str, Any]) -> str:
    ents = record.get("entities") or []
    ent_txt = ", ".join(
        f"{e.get('text')!r}:{e.get('source')}" for e in ents[:8]
    ) or "(none)"
    top = record.get("top_ranked")
    if top:
        why = top.get("reject_reason") or (
            "cleared cosine gate" if float(top.get("score") or 0) >= float(record.get("min_cosine") or MIN_COSINE)
            else "unknown"
        )
        top_line = f"top={top.get('score')} {top.get('text')!r} why={why}"
    else:
        top_line = "top=(no candidates)"
    return (
        f"  dlg={record['dialogue_id']} turn={record['turn_id']} cat={record['category']} "
        f"retr={record['candidates_retrieved']} ranked={record['candidates_after_ranking']} "
        f"passages={record['passages_used_count']} fallback={record['used_history_fallback']}\n"
        f"    entities: {ent_txt}\n"
        f"    {top_line}"
    )


def retrieval_vs_gate(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    zero = sum(1 for r in records if int(r.get("candidates_retrieved") or 0) == 0)
    nonzero_fail = sum(
        1
        for r in records
        if int(r.get("candidates_retrieved") or 0) > 0 and int(r.get("gate_passed_count") or 0) == 0
    )
    nonzero_pass = sum(1 for r in records if int(r.get("gate_passed_count") or 0) > 0)
    zero_no_passage = sum(
        1
        for r in records
        if int(r.get("candidates_retrieved") or 0) == 0 and int(r.get("passages_used_count") or 0) == 0
    )
    zero_openie = sum(
        1
        for r in records
        if int(r.get("candidates_retrieved") or 0) == 0 and int(r.get("passages_used_count") or 0) > 0
    )
    return {
        "n": len(records),
        "zero_candidates_retrieved": zero,
        "zero_no_passage": zero_no_passage,
        "zero_openie_empty": zero_openie,
        "candidates_but_gate_failed": nonzero_fail,
        "gate_passed": nonzero_pass,
    }


def write_diagnosis_md(
    path: Path,
    *,
    n_sample: int,
    seed: int,
    crosstab_rows: Sequence[Dict[str, Any]],
    sample_records: Sequence[Dict[str, Any]],
    counseling_cases: Sequence[Dict[str, Any]],
    empty_cases: Sequence[Dict[str, Any]],
) -> str:
    sample_split = retrieval_vs_gate(sample_records)
    counseling_split = retrieval_vs_gate(counseling_cases)
    empty_split = retrieval_vs_gate(empty_cases)

    lex = next((r for r in crosstab_rows if r["group"] == "lexicon_only"), {})
    weak = next((r for r in crosstab_rows if r["group"] == "ner_or_chunk_no_lexicon"), {})
    lex_rate = lex.get("gate_pass_rate")
    weak_rate = weak.get("gate_pass_rate")
    if lex_rate is None or weak_rate is None or lex.get("n", 0) == 0 or weak.get("n", 0) == 0:
        ner_note = (
            "Not enough turns in one of the entity-source buckets to compare NER/noun-chunk "
            "vs lexicon gate pass rates."
        )
        ner_lower = False
    else:
        ner_lower = weak_rate < lex_rate
        ner_note = (
            f"Lexicon-only gate pass rate is {lex_rate}% (n={lex.get('n')}); "
            f"NER/noun-chunk-with-no-lexicon is {weak_rate}% (n={weak.get('n')}). "
            + (
                "NER/noun-chunk turns pass the gate less often, which would argue for a "
                "confidence-aware threshold rather than more extraction coverage."
                if ner_lower
                else "NER/noun-chunk turns do not pass the gate less often than lexicon-only turns, "
                "so extraction-source confidence is not the main gate problem in this sample."
            )
        )

    def _primary(split: Dict[str, int], label: str) -> str:
        n = split["n"] or 1
        z = split["zero_candidates_retrieved"]
        f = split["candidates_but_gate_failed"]
        return (
            f"{label}: n={split['n']}; zero retrieved={z} ({100.0 * z / n:.0f}% "
            f"[no passage survived 0.38={split.get('zero_no_passage', 0)}, "
            f"passage but OpenIE empty={split.get('zero_openie_empty', 0)}]); "
            f"retrieved but failed cosine/usability gate={f} ({100.0 * f / n:.0f}%); "
            f"gate passed={split['gate_passed']}"
        )

    # Single most important distinction: among counseling-only + empty.
    both = list(counseling_cases) + list(empty_cases)
    both_split = retrieval_vs_gate(both)
    if both_split["zero_candidates_retrieved"] >= both_split["candidates_but_gate_failed"]:
        primary = "retrieval coverage"
        recommend = (
            "Fix retrieval coverage first: most counseling-only/empty turns never put a "
            "static candidate into the ranker (passage cosine < 0.38 and/or OpenIE produced "
            "no triplets). Lowering the candidate gate would not help turns with an empty pool."
        )
    else:
        primary = "threshold/ranking"
        recommend = (
            "Fix the candidate gate/ranking first: most counseling-only/empty turns do retrieve "
            "candidates, but they fail min_cosine=0.38 or the post-gate usability filter."
        )

    body = f"""# KTC gate diagnosis

Sample: {n_sample} turns, seed={seed}, static pipeline (no live retrieval).
Candidate gate: per-candidate cosine >= {MIN_COSINE}, then `is_ktc_usable`, then substring dedup.
Funnel `gate_passed` is the count after those filters.

## Cross-tab (step 2)

```
{format_crosstab(crosstab_rows)}
```

## NER/noun-chunk vs lexicon

{ner_note}

## Empty / counseling-only: retrieval vs gate

{_primary(sample_split, "main sample")}
{_primary(counseling_split, "supplemental_counseling-only (20)")}
{_primary(empty_split, "fully empty (20)")}

**Primary bottleneck: {primary}.**

## Recommendation

{recommend}

## Notes

- `candidates_retrieved` is the static OpenIE pool (`static_candidates_from_triplets`) before ranking.
  An empty pool usually means no knowledge passage cleared the *passage* cosine gate (also 0.38)
  or OpenIE extracted zero triplets from the passages that did.
- Ranker scores every retrieved candidate; `candidates_after_ranking` equals the pool size.
- Counseling-bank facts are injected *after* the gate and are not in `candidates_retrieved`.
"""
    path.write_text(body, encoding="utf-8")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace KTC funnel stages on a corpus sample.")
    parser.add_argument("--input", type=Path, default=Path("KARE-data/KARE/Data/KARE.jsonl"))
    parser.add_argument("--n", type=int, default=50, help="Number of turns to sample for the main trace.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    parser.add_argument("--empty-n", type=int, default=20, dest="empty_n")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("ktc.trace_funnel").setLevel(logging.DEBUG)

    dialogues = load_dialogues(args.input)
    all_turns = collect_turns(dialogues)
    rng = random.Random(args.seed)
    shuffled = all_turns[:]
    rng.shuffle(shuffled)
    if len(shuffled) < args.n:
        raise SystemExit(f"only {len(shuffled)} turns available, need --n {args.n}")

    _install_rank_tracer()
    pipeline = KnowledgeTripletPipeline(
        verbalization_backend="template",
        coref_backend="heuristic",
    )

    sample_specs = shuffled[: args.n]
    rest = shuffled[args.n :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = args.output_dir / f"trace_{stamp}.jsonl"

    sample_records: List[Dict[str, Any]] = []
    with trace_path.open("w", encoding="utf-8") as handle:
        for spec in sample_specs:
            record = trace_turn(pipeline, **spec)
            sample_records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counseling: List[Dict[str, Any]] = [
        r for r in sample_records if empty_bucket_kind(r) == "supplemental_counseling"
    ]
    empty: List[Dict[str, Any]] = [r for r in sample_records if empty_bucket_kind(r) == "empty"]
    extra_records: List[Dict[str, Any]] = []
    for spec in rest:
        if len(counseling) >= args.empty_n and len(empty) >= args.empty_n:
            break
        record = trace_turn(pipeline, **spec)
        extra_records.append(record)
        kind = empty_bucket_kind(record)
        if kind == "supplemental_counseling" and len(counseling) < args.empty_n:
            counseling.append(record)
        elif kind == "empty" and len(empty) < args.empty_n:
            empty.append(record)
    counseling = counseling[: args.empty_n]
    empty = empty[: args.empty_n]

    rows = crosstab(sample_records)
    table = format_crosstab(rows)
    print("=== Cross-tab gate pass rate by entity source / history fallback ===")
    print(table)
    print()
    print(f"=== {len(counseling)} supplemental_counseling-only turns ===")
    for rec in counseling:
        print(format_empty_case(rec))
    print()
    print(f"=== {len(empty)} fully empty turns ===")
    for rec in empty:
        print(format_empty_case(rec))

    md_path = args.output_dir / f"gate_diagnosis_{stamp}.md"
    md = write_diagnosis_md(
        md_path,
        n_sample=args.n,
        seed=args.seed,
        crosstab_rows=rows,
        sample_records=sample_records,
        counseling_cases=counseling,
        empty_cases=empty,
    )
    print()
    print(md)
    print(f"wrote {trace_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
