"""Stage 2d — Relevance ranking with Sentence-BERT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ktc.knowledge_item import KnowledgeCandidate
from ktc.triplet import Triplet


@dataclass
class CandidateRankingResult:
    """Ranked knowledge candidates (static + live) with similarity scores."""

    candidates: List[KnowledgeCandidate]
    scores: List[float]
    top1_score: float


@dataclass
class RankingResult:
    """Ranked triplets plus similarity scores for visibility / quality flagging."""

    triplets: List[Triplet]
    scores: List[float]
    top1_score: float


class SentenceBertRanker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def rank_with_scores(
        self, dialog_history: str, triplets: Iterable[Triplet], top_k: int = 26
    ) -> RankingResult:
        triplet_list = list(triplets)
        if not triplet_list:
            return RankingResult(triplets=[], scores=[], top1_score=0.0)

        history_emb = self.model.encode(dialog_history, normalize_embeddings=True)
        triplet_texts = [t.as_text() for t in triplet_list]
        triplet_embs = self.model.encode(triplet_texts, normalize_embeddings=True)

        scores = np.dot(triplet_embs, history_emb)
        order = np.argsort(scores)[::-1][:top_k]
        ranked = [triplet_list[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]
        top1 = ranked_scores[0] if ranked_scores else 0.0
        return RankingResult(triplets=ranked, scores=ranked_scores, top1_score=top1)

    def rank(self, dialog_history: str, triplets: Iterable[Triplet], top_k: int = 26) -> List[Triplet]:
        return self.rank_with_scores(dialog_history, triplets, top_k=top_k).triplets

    def rank_candidates_with_scores(
        self, dialog_history: str, candidates: Iterable[KnowledgeCandidate], top_k: int = 26
    ) -> CandidateRankingResult:
        candidate_list = list(candidates)
        if not candidate_list:
            return CandidateRankingResult(candidates=[], scores=[], top1_score=0.0)

        history_emb = self.model.encode(dialog_history, normalize_embeddings=True)
        texts = [c.text for c in candidate_list]
        embs = self.model.encode(texts, normalize_embeddings=True)

        scores = np.dot(embs, history_emb)
        order = np.argsort(scores)[::-1][:top_k]
        ranked = [candidate_list[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]
        top1 = ranked_scores[0] if ranked_scores else 0.0
        return CandidateRankingResult(candidates=ranked, scores=ranked_scores, top1_score=top1)


def rank_triplets(
    dialog_history: str,
    triplets: Iterable[Triplet],
    top_k: int = 26,
    ranker: Optional[SentenceBertRanker] = None,
) -> Tuple[List[Triplet], float]:
    """Rank triplets by cosine similarity to dialog history; return (ranked, top1_score)."""
    if ranker is None:
        ranker = SentenceBertRanker()
    result = ranker.rank_with_scores(dialog_history, triplets, top_k=top_k)
    return result.triplets, result.top1_score


def rank_candidates(
    dialog_history: str,
    candidates: Iterable[KnowledgeCandidate],
    top_k: int = 26,
    ranker: Optional[SentenceBertRanker] = None,
) -> Tuple[List[KnowledgeCandidate], float]:
    if ranker is None:
        ranker = SentenceBertRanker()
    result = ranker.rank_candidates_with_scores(dialog_history, candidates, top_k=top_k)
    return result.candidates, result.top1_score
