"""Split KARE knowledge blobs into passages and keep only those matching the user need."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ktc.cleaning import clean_knowledge_text

_K_SPLIT = re.compile(r"<K\d+>", re.IGNORECASE)

# Keep at most this many knowledge passages for OpenIE.
DEFAULT_PASSAGE_TOP_N = 3


def split_knowledge_passages(raw_text: str) -> List[str]:
    """Split a KARE knowledge field on ``<K#>`` tags, then clean each chunk."""
    text = raw_text or ""
    parts = _K_SPLIT.split(text)
    passages: List[str] = []
    for part in parts:
        cleaned = clean_knowledge_text(part)
        if cleaned:
            passages.append(cleaned)
    if passages:
        return passages
    cleaned = clean_knowledge_text(text)
    return [cleaned] if cleaned else []


def select_relevant_passages(
    passages: Sequence[str],
    query: str,
    ranker,
    top_n: int = DEFAULT_PASSAGE_TOP_N,
    min_cosine: float = 0.38,
) -> List[Tuple[str, float]]:
    """Return ``(passage, score)`` for the top passages at or above *min_cosine*."""
    if not query.strip() or not passages:
        return []

    model = getattr(ranker, "model", None)
    if model is None:
        return [(p, 1.0) for p in list(passages)[:top_n]]

    import numpy as np

    query_emb = model.encode(query.strip(), normalize_embeddings=True)
    embs = model.encode(list(passages), normalize_embeddings=True)
    scores = np.dot(embs, query_emb)
    order = np.argsort(-scores)
    selected: List[Tuple[str, float]] = []
    for idx in order:
        score = float(scores[idx])
        if score < min_cosine:
            continue
        selected.append((passages[int(idx)], score))
        if len(selected) >= top_n:
            break
    return selected
