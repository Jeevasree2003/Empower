"""Split KARE knowledge blobs into passages and keep only those matching the user need."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ktc.cleaning import clean_knowledge_text

_K_SPLIT = re.compile(r"<K\d+>", re.IGNORECASE)

# Keep at most this many knowledge passages for OpenIE.
DEFAULT_PASSAGE_TOP_N = 3


def split_knowledge_passages(raw_text: str) -> List[str]:
    """Split a KARE knowledge field on ``<K#>`` tags or sentence windows."""
    text = raw_text or ""
    tagged = _K_SPLIT.split(text)
    chunks = tagged if len(tagged) > 1 else [text]
    passages: List[str] = []
    for part in chunks:
        cleaned = clean_knowledge_text(part)
        if not cleaned:
            continue
        passages.extend(_sentence_windows(cleaned))
    return passages


def _sentence_windows(cleaned: str, window: int = 2) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 20]
    if not sentences:
        return [cleaned] if cleaned else []
    if len(sentences) <= 3:
        return [cleaned]
    windows: List[str] = []
    for i in range(0, len(sentences), window):
        chunk = " ".join(sentences[i : i + window]).strip()
        if chunk:
            windows.append(chunk)
    return windows


def _is_junk_passage(passage: str) -> bool:
    lowered = passage.lower()
    if "online legal india" in lowered or "received your complaint request" in lowered:
        return True
    if "consumer complaint against mental harassment" in lowered:
        return True
    if lowered.count(" i ") >= 4 and ("they called" in lowered or "terminate me" in lowered):
        return True
    return False


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
    usable = [p for p in passages if not _is_junk_passage(p)]
    if not usable:
        return []

    model = getattr(ranker, "model", None)
    if model is None:
        return [(p, 1.0) for p in usable[:top_n]]

    import numpy as np

    query_emb = model.encode(query.strip(), normalize_embeddings=True)
    embs = model.encode(list(usable), normalize_embeddings=True)
    scores = np.dot(embs, query_emb)
    order = np.argsort(-scores)
    selected: List[Tuple[str, float]] = []
    for idx in order:
        score = float(scores[idx])
        if score < min_cosine:
            continue
        selected.append((usable[int(idx)], score))
        if len(selected) >= top_n:
            break
    return selected


def select_dual_domain_passages(
    passages: Sequence[str],
    base_query: str,
    ranker,
    top_n: int = DEFAULT_PASSAGE_TOP_N,
    min_cosine: float = 0.38,
    include_legal: bool = False,
    include_clinical: bool = True,
) -> List[Tuple[str, float]]:
    """Score passages against the user need; only expand legal when the turn is legal."""
    if not base_query.strip():
        return []
    queries = [base_query]
    if include_clinical:
        queries.append(f"{base_query} mental health counseling crisis helpline safety")
    if include_legal:
        queries.append(f"{base_query} FIR police complaint Indian law helpline protection")
    merged: List[Tuple[str, float]] = []
    seen = set()
    for query in queries:
        for passage, score in select_relevant_passages(
            passages, query, ranker, top_n=top_n, min_cosine=min_cosine
        ):
            key = passage[:200]
            if key in seen:
                continue
            seen.add(key)
            merged.append((passage, score))
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged[: max(top_n, 2)]
