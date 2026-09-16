"""Build a local passage list from KARE.jsonl or preprocessed_v2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ktc.knowledge_item import KnowledgeCandidate
from ktc.passages import split_knowledge_passages
from ktc.ranking import encode_texts_cached, get_ranker
from ktc.reply_knowledge import is_reply_usable

from demo.paths import KARE_JSONL, PREPROCESSED_V2

logger = logging.getLogger(__name__)

MAX_PASSAGES = 4000
MIN_CHARS = 40
NO_PASSAGE = "no_passages_used"
KNOWLEDGE_SEP = "__knowledge__"


def _knowledge_blobs_from_kare(path: Path) -> List[str]:
    blobs: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            knowledge = record.get("knowledge") or ""
            if isinstance(knowledge, list):
                knowledge = knowledge[0] if knowledge else ""
            if knowledge:
                blobs.append(str(knowledge))
    return blobs


def _knowledge_blobs_from_preprocessed(split_dir: Path) -> List[str]:
    blobs: List[str] = []
    for name in ("train.json", "valid.json", "test.json"):
        path = split_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                field = row.get("knowledge") or [""]
                raw = field[0] if isinstance(field, list) else str(field)
                train_half = raw.split(KNOWLEDGE_SEP, 1)[0].strip()
                eval_half = ""
                if KNOWLEDGE_SEP in raw:
                    eval_half = raw.split(KNOWLEDGE_SEP, 1)[1].strip()
                for part in (eval_half, train_half):
                    if part and part.lower() != NO_PASSAGE:
                        blobs.append(part)
    return blobs


def collect_passages(
    kare_path: Optional[Path] = None,
    preprocessed_dir: Optional[Path] = None,
    max_passages: int = MAX_PASSAGES,
) -> Tuple[List[str], str]:
    kare_path = kare_path or KARE_JSONL
    preprocessed_dir = preprocessed_dir or PREPROCESSED_V2
    blobs: List[str] = []
    source = "empty"
    if kare_path.exists():
        blobs = _knowledge_blobs_from_kare(kare_path)
        source = str(kare_path)
    elif preprocessed_dir.exists():
        blobs = _knowledge_blobs_from_preprocessed(preprocessed_dir)
        source = str(preprocessed_dir)
    else:
        logger.warning("No KARE.jsonl or preprocessed_v2 found; corpus is empty")
        return [], source

    seen = set()
    passages: List[str] = []
    for blob in blobs:
        for passage in split_knowledge_passages(blob):
            text = (passage or "").strip()
            if len(text) < MIN_CHARS:
                continue
            probe = KnowledgeCandidate(text=text, source="static_dataset")
            if not is_reply_usable(probe):
                continue
            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            passages.append(text)
            if len(passages) >= max_passages:
                logger.info("corpus_capped n=%s source=%s", len(passages), source)
                return passages, source
    logger.info("corpus_ready n=%s source=%s", len(passages), source)
    return passages, source


class PassageCorpus:
    """Rank local knowledge windows against a query with the KTC ranker."""

    def __init__(self, passages: Optional[Sequence[str]] = None, ranker=None):
        loaded, source = collect_passages() if passages is None else (list(passages), "injected")
        self.passages = loaded
        self.source = source
        self.ranker = ranker if ranker is not None else get_ranker("auto")
        self._matrix = None
        model = getattr(self.ranker, "model", None)
        if model is not None and self.passages:
            self._matrix = encode_texts_cached(model, self.passages)

    def top_k(self, query: str, k: int = 8) -> List[Tuple[str, float]]:
        if not query.strip() or not self.passages:
            return []
        model = getattr(self.ranker, "model", None)
        if model is not None and self._matrix is not None:
            import numpy as np

            query_vec = encode_texts_cached(model, [query])[0]
            scores = self._matrix @ query_vec
            order = np.argsort(-scores)[:k]
            return [(self.passages[i], float(scores[i])) for i in order]
        if hasattr(self.ranker, "cosine_to_query"):
            scored = self.ranker.cosine_to_query(query, self.passages)
        else:
            q = set(query.lower().split())
            scored = [len(q.intersection(p.lower().split())) / (1.0 + len(q)) for p in self.passages]
        ranked = sorted(zip(self.passages, scored), key=lambda item: item[1], reverse=True)
        return [(text, float(score)) for text, score in ranked[:k] if score > 0]
