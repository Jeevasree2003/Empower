"""Stage 3 — Convert raw KARE dialogues into EMPOWER training JSON files."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ktc.live_config import LiveRetrievalConfig
from ktc.pipeline import KnowledgeTripletPipeline

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


def process_dialogue_turn(
    pipeline: KnowledgeTripletPipeline,
    knowledge_text: str,
    dialog_history: str,
    *,
    knowledge_mode: str = "ktc",
    enable_live: bool = False,
    dialogue_id: str = "",
    turn: object = "",
):
    """Run one agent-reply turn. Returns (HybridRunResult or None, verbalized sentences)."""
    if knowledge_mode == "raw":
        return None, pipeline.run_raw_knowledge(knowledge_text)
    result = pipeline.run_hybrid(
        knowledge_text,
        dialog_history,
        enable_live=enable_live,
        dialogue_id=str(dialogue_id or ""),
        turn=turn,
    )
    return result, result.verbalized


def build_turn_examples(
    dialogue: dict,
    pipeline: KnowledgeTripletPipeline,
    knowledge_mode: str = "ktc",
    enable_live: bool = False,
) -> List[dict]:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    knowledge_text = dialogue.get("knowledge", "") or ""
    examples: List[dict] = []
    history: List[str] = []
    dialogue_id = str(dialogue.get("dialogue_id") or "")
    bot_turn = 0

    for utterance in utterances:
        formatted = format_utterance(utterance)
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])

        if role == "agent" and history:
            dialog_history = " ".join(history)
            _result, verbalized = process_dialogue_turn(
                pipeline,
                knowledge_text,
                dialog_history,
                knowledge_mode=knowledge_mode,
                enable_live=enable_live,
                dialogue_id=dialogue_id,
                turn=bot_turn,
            )
            bot_turn += 1

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
    verbalization_backend: str = "llm",
    coref_backend: str = "heuristic",
    top_k: int = 8,
    max_dialogues: Optional[int] = None,
    enable_live: bool = False,
    per_dialogue_live_budget: Optional[int] = None,
) -> Dict[str, int]:
    dialogues = load_dialogues(input_path)
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]

    splits = split_dialogues(dialogues, train_size, valid_size, test_size, seed=seed)
    live_config = LiveRetrievalConfig.load()
    if per_dialogue_live_budget is not None:
        live_config.per_dialogue_budget = int(per_dialogue_live_budget)
    pipeline = KnowledgeTripletPipeline(
        top_k=top_k,
        verbalization_backend=verbalization_backend,
        coref_backend=coref_backend,
        live_config=live_config,
    )

    counts = {}
    for split_name, split_data in splits.items():
        examples: List[dict] = []
        for dialogue in split_data:
            examples.extend(
                build_turn_examples(
                    dialogue, pipeline, knowledge_mode=knowledge_mode, enable_live=enable_live
                )
            )
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
        default="llm",
        help="Verbalization backend for Stage 2e (llm=Groq/OpenAI-compatible; template=offline).",
    )
    parser.add_argument(
        "--coref_backend",
        choices=["heuristic", "model"],
        default="heuristic",
        help="Coreference backend for Stage 2c (heuristic=local rules, model=coreferee).",
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=None,
        help="Process only the first N dialogues (useful for smoke tests).",
    )
    parser.add_argument(
        "--enable-live",
        action="store_true",
        default=False,
        help="Opt in to hybrid live retrieval (off by default; uses one shared API budget).",
    )
    parser.add_argument(
        "--per-dialogue-live-budget",
        type=int,
        default=None,
        metavar="N",
        help="Max live API calls per dialogue_id (default: unset, global budget only).",
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
        coref_backend=args.coref_backend,
        top_k=args.top_k,
        max_dialogues=args.max_dialogues,
        enable_live=args.enable_live,
        per_dialogue_live_budget=args.per_dialogue_live_budget,
    )

    for split_name, count in counts.items():
        print(f"{split_name}: {count} turns written to {args.output_dir / (split_name + '.json')}")


if __name__ == "__main__":
    main()
