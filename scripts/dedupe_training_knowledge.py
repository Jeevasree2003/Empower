"""Cap overused counseling-bank knowledge strings in KDPT training targets.

Reads data/preprocessed/{train,valid,test}.json and writes data/preprocessed_v2/
without touching the original 15h splits.

A knowledge field is "{train_text} __knowledge__ {eval_text}". High-frequency
identical train_text blobs collapse KDPT into a handful of posters. Those
train halves are replaced with no_passages_used; the eval half is kept so KF1
still measures overlap with the original knowledge.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

KNOWLEDGE_SEP = "__knowledge__"
NO_PASSAGE_USED = "no_passages_used"


def split_knowledge(field: str) -> tuple[str, str]:
    if isinstance(field, list):
        field = field[0] if field else ""
    raw = (field or "").strip()
    if KNOWLEDGE_SEP in raw:
        train_half, eval_half = raw.split(KNOWLEDGE_SEP, 1)
        return train_half.strip(), eval_half.strip()
    return raw, raw


def join_knowledge(train_half: str, eval_half: str) -> str:
    return f"{train_half} {KNOWLEDGE_SEP} {eval_half}".strip()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def knowledge_stats(rows: list[dict]) -> tuple[Counter, int, int]:
    counts: Counter = Counter()
    empty = 0
    for row in rows:
        train_half, _ = split_knowledge(row.get("knowledge", [""])[0] if isinstance(row.get("knowledge"), list) else row.get("knowledge", ""))
        counts[train_half] += 1
        if not train_half or train_half == NO_PASSAGE_USED:
            empty += 1
    return counts, empty, len(rows)


def capped_templates(counts: Counter, n_rows: int, top_k: int, min_count: int) -> set[str]:
    """Templates that are both frequent and among the top-k most common."""
    cap = max(min_count, int(0.002 * n_rows))
    banned = set()
    for text, freq in counts.most_common(top_k):
        if not text or text == NO_PASSAGE_USED:
            continue
        if freq > cap:
            banned.add(text)
    return banned


def rewrite_rows(rows: list[dict], banned: set[str], keep_per_template: int) -> tuple[list[dict], int]:
    seen: Counter = Counter()
    n_rewritten = 0
    out = []
    for row in rows:
        field = row.get("knowledge", [""])
        if isinstance(field, list):
            raw = field[0] if field else ""
        else:
            raw = field or ""
        train_half, eval_half = split_knowledge(raw)
        new_train = train_half
        if train_half in banned:
            seen[train_half] += 1
            if seen[train_half] > keep_per_template:
                new_train = NO_PASSAGE_USED
                n_rewritten += 1
        new_row = dict(row)
        new_row["knowledge"] = [join_knowledge(new_train, eval_half)]
        out.append(new_row)
    return out, n_rewritten


def format_preview(text: str, n: int = 120) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/preprocessed_v2"))
    parser.add_argument("--report", type=Path, default=Path("reports/knowledge_dedupe_v2.txt"))
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--min_count", type=int, default=80)
    args = parser.parse_args()

    train = load_jsonl(args.input_dir / "train.json")
    valid = load_jsonl(args.input_dir / "valid.json")
    test = load_jsonl(args.input_dir / "test.json")

    before, empty_before, n_train = knowledge_stats(train)
    banned = capped_templates(before, n_train, args.top_k, args.min_count)

    cap = max(args.min_count, int(0.002 * n_train))
    train_out, n_tr = rewrite_rows(train, banned, cap)
    valid_out, n_va = rewrite_rows(valid, banned, cap)
    test_out, n_te = rewrite_rows(test, banned, cap)

    after, empty_after, _ = knowledge_stats(train_out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.json", train_out)
    write_jsonl(args.output_dir / "valid.json", valid_out)
    write_jsonl(args.output_dir / "test.json", test_out)

    lines = []
    lines.append(f"input:  {args.input_dir.resolve()}")
    lines.append(f"output: {args.output_dir.resolve()}")
    lines.append(f"top_k={args.top_k} min_count={args.min_count}")
    lines.append("")
    lines.append(f"train rows: {n_train}")
    lines.append(f"unique train-half BEFORE: {len(before)} / {n_train} ({100 * len(before) / n_train:.2f}%)")
    lines.append(f"no_passages_used BEFORE: {empty_before} ({100 * empty_before / n_train:.2f}%)")
    lines.append(f"templates capped: {len(banned)}")
    lines.append(f"rows rewritten: train={n_tr} valid={n_va} test={n_te}")
    lines.append(f"unique train-half AFTER:  {len(after)} / {n_train} ({100 * len(after) / n_train:.2f}%)")
    lines.append(f"no_passages_used AFTER:  {empty_after} ({100 * empty_after / n_train:.2f}%)")
    lines.append("")
    lines.append("top-10 BEFORE:")
    for text, freq in before.most_common(10):
        lines.append(f"  {freq:6d}  {format_preview(text)}")
    lines.append("")
    lines.append("top-10 AFTER:")
    for text, freq in after.most_common(10):
        lines.append(f"  {freq:6d}  {format_preview(text)}")
    lines.append("")
    lines.append("capped templates:")
    for text in sorted(banned, key=lambda t: -before[t]):
        lines.append(f"  {before[text]:6d}  {format_preview(text, 160)}")

    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
