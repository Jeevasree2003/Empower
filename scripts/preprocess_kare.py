"""Stage 3 — Convert raw KARE dialogues into EMPOWER training JSON files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ktc.pipeline import KnowledgeTripletPipeline
from ktc.triplet import Triplet

KNOWLEDGE_SEP = "__knowledge__"
NO_PASSAGE_USED = "no_passages_used"

ROLE_MAP = {
    "bot": "agent",
    "agent": "agent",
    "user": "victim",
    "victim": "victim",
}


def load_dialogues(path: Path) -> List[dict]:
    dialogues = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    return dialogues


def split_dialogues(
    dialogues: List[dict],
    train_size: int = 4000,
    valid_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    total = train_size + valid_size + test_size
    if len(dialogues) < total:
        # Smoke-test path: use whatever dialogues we have (all in train).
        return {"train": dialogues, "valid": [], "test": []}

    shuffled = dialogues.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    return {
        "train": shuffled[:train_size],
        "valid": shuffled[train_size : train_size + valid_size],
        "test": shuffled[train_size + valid_size : total],
    }


def format_utterance(utterance: dict) -> str:
    role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
    return f"{role}: {utterance['utterance'].strip()}"


def build_turn_examples(
    dialogue: dict,
    pipeline: KnowledgeTripletPipeline,
    knowledge_mode: str = "ktc",
) -> List[dict]:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    knowledge_text = dialogue.get("knowledge", "") or ""
    examples: List[dict] = []
    history: List[str] = []
    filtered_triplets: Optional[List[Triplet]] = None

    for utterance in utterances:
        formatted = format_utterance(utterance)
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])

        if role == "agent" and history:
            dialog_history = " ".join(history)
            if knowledge_mode == "raw":
                verbalized = pipeline.run_raw_knowledge(knowledge_text)
            else:
                if filtered_triplets is None:
                    filtered_triplets = pipeline.get_filtered_triplets(knowledge_text)
                verbalized = pipeline.run(knowledge_text, dialog_history, filtered=filtered_triplets)

            if verbalized:
                knowledge_for_training = " ".join(verbalized)
                knowledge_for_eval = knowledge_for_training
            else:
                knowledge_for_training = NO_PASSAGE_USED
                knowledge_for_eval = NO_PASSAGE_USED

            knowledge_field = f"{knowledge_for_training} {KNOWLEDGE_SEP} {knowledge_for_eval}".strip()
            examples.append(
                {
                    "history": history.copy(),
                    "knowledge": [knowledge_field],
                    "response": utterance["utterance"].strip(),
                }
            )

        history.append(formatted)

    return examples


def write_split(path: Path, examples: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def preprocess(
    input_path: Path,
    output_dir: Path,
    train_size: int = 4000,
    valid_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
    knowledge_mode: str = "ktc",
    verbalization_backend: str = "template",
    top_k: int = 26,
    max_dialogues: Optional[int] = None,
) -> Dict[str, int]:
    dialogues = load_dialogues(input_path)
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]

    splits = split_dialogues(dialogues, train_size, valid_size, test_size, seed=seed)
    pipeline = KnowledgeTripletPipeline(
        top_k=top_k,
        verbalization_backend=verbalization_backend,
    )

    counts = {}
    for split_name, split_data in splits.items():
        examples: List[dict] = []
        for dialogue in split_data:
            examples.extend(build_turn_examples(dialogue, pipeline, knowledge_mode=knowledge_mode))
        counts[split_name] = write_split(output_dir / f"{split_name}.json", examples)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Preprocess KARE.jsonl into EMPOWER train/valid/test JSONL files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../../KARE-data/KARE/Data/KARE.jsonl"),
        help="Path to KARE.jsonl (one dialogue per line).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../data/preprocessed"),
        help="Directory for train.json, valid.json, and test.json.",
    )
    parser.add_argument("--train_size", type=int, default=4000)
    parser.add_argument("--valid_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--knowledge_mode",
        choices=["ktc", "raw"],
        default="ktc",
        help="Use full KTC pipeline or raw knowledge text (EMPOWER-KTC ablation).",
    )
    parser.add_argument(
        "--verbalization_backend",
        choices=["template", "llm"],
        default="template",
        help="Verbalization backend for Stage 2e.",
    )
    parser.add_argument("--top_k", type=int, default=26)
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=None,
        help="Process only the first N dialogues (useful for smoke tests).",
    )
    args = parser.parse_args()

    counts = preprocess(
        input_path=args.input,
        output_dir=args.output_dir,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        knowledge_mode=args.knowledge_mode,
        verbalization_backend=args.verbalization_backend,
        top_k=args.top_k,
        max_dialogues=args.max_dialogues,
    )

    for split_name, count in counts.items():
        print(f"{split_name}: {count} turns written to {args.output_dir / (split_name + '.json')}")


if __name__ == "__main__":
    main()
