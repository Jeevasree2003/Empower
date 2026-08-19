"""Split noisy KARE knowledge into passages and keep only those relevant to the user."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ktc.cleaning import clean_knowledge_text

MAX_PASSAGES = 3
MIN_PASSAGE_SCORE = 0.28
TARGET_WINDOW_CHARS = 550


def split_knowledge_passages(knowledge_text: str) -> List[str]:
    """Split scraped knowledge into independently rankable passages."""
    raw = (knowledge_text or "").replace("\u2019", "'")
    tagged = re.split(r"<K\d+>", raw)
    tagged = [p.strip() for p in tagged if len(p.strip()) > 40]
    if len(tagged) >= 2:
        return [_clean_keep(p) for p in tagged if _clean_keep(p)]

    chunks = re.split(
        r"(?:\n{2,})|(?<=[.!?])\s+(?=(?:Conclusion:|The best way|Once you|I am not|"
        r"Your body|We have received|Team Online))",
        raw,
    )
    chunks = [c.strip() for c in chunks if len(c.strip()) > 80]
    if len(chunks) >= 3:
        cleaned = [_clean_keep(c) for c in chunks]
        return [c for c in cleaned if c]

    cleaned = clean_knowledge_text(raw)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 20]
    if not sentences:
        return [cleaned] if cleaned else []

    windows: List[str] = []
    buf: List[str] = []
    size = 0
    for sent in sentences:
        buf.append(sent)
        size += len(sent)
        if size >= TARGET_WINDOW_CHARS:
            windows.append(" ".join(buf))
            buf, size = [], 0
    if buf:
        windows.append(" ".join(buf))
    return windows


def _clean_keep(text: str) -> str:
    cleaned = clean_knowledge_text(text)
    return cleaned if len(cleaned) > 40 else ""


def select_relevant_passages(
    knowledge_text: str,
    ranking_query: str,
    ranker,
    max_passages: int = MAX_PASSAGES,
    min_score: float = MIN_PASSAGE_SCORE,
) -> Tuple[List[str], List[float]]:
    """Return the top passages whose cosine score vs *ranking_query* clears *min_score*."""
    passages = split_knowledge_passages(knowledge_text)
    if not passages or not (ranking_query or "").strip():
        return [], []

    from ktc.knowledge_item import KnowledgeCandidate

    candidates = [KnowledgeCandidate(text=p, source="static_dataset") for p in passages]
    result = ranker.rank_candidates_with_scores(
        ranking_query, candidates, top_k=len(candidates)
    )
    kept: List[str] = []
    scores: List[float] = []
    for cand, score in zip(result.candidates, result.scores):
        if score < min_score:
            continue
        kept.append(cand.text)
        scores.append(score)
        if len(kept) >= max_passages:
            break
    return kept, scores
