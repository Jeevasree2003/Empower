"""One-off: diversity and repeat examples from a full-text knowledge audit dump."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples",
        type=Path,
        nargs="?",
        default=Path("reports") / "audit_compare_100_samples_no_fallback_full.jsonl",
    )
    parser.add_argument("--examples_per_string", type=int, default=4)
    args = parser.parse_args()

    rows = []
    with args.samples.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    full_texts = [row.get("final_knowledge_text_full") or "" for row in rows]
    nonempty = [t for t in full_texts if t.strip()]
    unique_nonempty = len(set(nonempty))
    counts = Counter(full_texts)
    by_text = defaultdict(list)
    for row in rows:
        by_text[row.get("final_knowledge_text_full") or ""].append(row)

    print(f"file={args.samples}")
    print(f"total_turns={len(rows)}")
    print(f"nonempty={len(nonempty)} empty={len(rows) - len(nonempty)}")
    print(f"unique_full_texts_among_nonempty={unique_nonempty} vs nonempty={len(nonempty)}")
    print(f"unique_including_empty={len(set(full_texts))} vs total={len(full_texts)}")
    print("-" * 60)
    print("Top 10 exact final_knowledge_text_full strings:")
    for i, (text, n) in enumerate(counts.most_common(10), start=1):
        preview = text if text else "(empty)"
        print(f"\n[{i}] n={n}")
        print(preview)

    print("\n" + "=" * 60)
    print("Example turns for top 3 most-repeated strings")
    for rank, (text, n) in enumerate(counts.most_common(3), start=1):
        label = text if text else "(empty)"
        print(f"\n### Top {rank} (n={n})")
        print(f"TEXT: {label}")
        examples = by_text[text][: args.examples_per_string]
        for ex in examples:
            print("-" * 40)
            print(f"dialogue_id={ex.get('dialogue_id')} turn_id={ex.get('turn_id')}")
            print(f"dialog_history_tail={ex.get('dialog_history_tail')!r}")


if __name__ == "__main__":
    main()