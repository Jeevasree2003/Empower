#!/usr/bin/env python
"""Count unique knowledge blobs across KARE.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    hash_to_ids: dict[str, list[str]] = defaultdict(list)
    total = 0

    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            knowledge = record.get("knowledge", "") or ""
            if isinstance(knowledge, list):
                knowledge = " ".join(str(k) for k in knowledge)
            digest = hashlib.sha256(knowledge.encode("utf-8")).hexdigest()
            hash_to_ids[digest].append(str(record.get("dialogue_id", total - 1)))

    unique = len(hash_to_ids)
    largest_group = max(hash_to_ids.values(), key=len)
    largest_size = len(largest_group)

    print(f"total_dialogues: {total}")
    print(f"unique_knowledge_blobs: {unique}")
    print(f"largest_shared_group_size: {largest_size}")
    print(f"largest_group_dialogue_ids_sample: {largest_group[:10]}")
    print()
    print("group_size_histogram (size -> count of groups):")
    size_counts: dict[int, int] = defaultdict(int)
    for ids in hash_to_ids.values():
        size_counts[len(ids)] += 1
    for size in sorted(size_counts):
        print(f"  {size:5d} dialogues/share blob: {size_counts[size]} groups")


if __name__ == "__main__":
    main()
