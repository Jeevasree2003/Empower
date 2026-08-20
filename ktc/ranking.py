"""Stage 2d — Relevance ranking with Sentence-BERT and optional cross-encoder rerank."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ktc.knowledge_item import KnowledgeCandidate
from ktc.triplet import Triplet

logger = logging.getLogger(__name__)


def _configure_hf_token() -> None:
    """Use HF_TOKEN from the environment/.env so SBERT downloads are authenticated."""
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        return
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    else:
        logger.info("HF_TOKEN not set; Sentence-BERT will use anonymous Hugging Face downloads")


def _content_key(text: str) -> tuple:
    return tuple(w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in {"a", "an", "the"})


def dedupe_knowledge_candidates(candidates: Iterable[KnowledgeCandidate]) -> List[KnowledgeCandidate]:
    """Drop exact (head, relation) duplicates and near-identical candidate text."""
    kept: List[KnowledgeCandidate] = []
    seen_keys = set()
    seen_texts: List[str] = []
    for candidate in candidates:
        if candidate.triplet is not None:
            key = (_content_key(candidate.triplet.head), _content_key(candidate.triplet.relation))
        else:
            key = _content_key(candidate.text)[:10]
        if key in seen_keys:
            continue
        norm = re.sub(r"\s+", " ", (candidate.text or "").lower()).strip()
        if any(SequenceMatcher(None, norm, prior).ratio() >= 0.92 for prior in seen_texts):
            continue
        seen_keys.add(key)
        seen_texts.append(norm)
        kept.append(candidate)
    return kept


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


@runtime_checkable
class CandidateRanker(Protocol):
    def rank_candidates_with_scores(
        self, dialog_history: str, candidates: Iterable[KnowledgeCandidate], top_k: int = 26
    ) -> CandidateRankingResult: ...


def _stable_rank_order(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of top-*top_k* scores with deterministic tie-breaking."""
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)
    indices = np.arange(n)
    # lexsort: last key is primary — sort by score desc, then index asc for ties.
    return np.lexsort((indices, -scores))[:top_k]


DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.38


def apply_score_gate(
    candidates: List[KnowledgeCandidate],
    scores: List[float],
    min_score: float = DEFAULT_MIN_SCORE,
    max_k: int = DEFAULT_TOP_K,
) -> CandidateRankingResult:
    """Keep only ranked items at/above *min_score*, capped at *max_k*."""
    kept_c: List[KnowledgeCandidate] = []
    kept_s: List[float] = []
    for candidate, score in zip(candidates, scores):
        if score < min_score:
            continue
        kept_c.append(candidate)
        kept_s.append(score)
        if len(kept_c) >= max_k:
            break
    top1 = kept_s[0] if kept_s else 0.0
    return CandidateRankingResult(candidates=kept_c, scores=kept_s, top1_score=top1)


class SentenceBertRanker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        _configure_hf_token()
        self.model = SentenceTransformer(model_name)

    def rank_with_scores(
        self, dialog_history: str, triplets: Iterable[Triplet], top_k: int = 26
    ) -> RankingResult:
        triplet_list = list(triplets)
        if not triplet_list:
            return RankingResult(triplets=[], scores=[], top1_score=0.0)

        history_text = dialog_history.strip() or " "
        history_emb = self.model.encode(history_text, normalize_embeddings=True)
        triplet_texts = [t.as_text() for t in triplet_list]
        triplet_embs = self.model.encode(triplet_texts, normalize_embeddings=True)

        scores = np.dot(triplet_embs, history_emb)
        order = _stable_rank_order(scores, top_k)
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

        history_text = dialog_history.strip() or " "
        history_emb = self.model.encode(history_text, normalize_embeddings=True)
        texts = [c.text for c in candidate_list]
        embs = self.model.encode(texts, normalize_embeddings=True)

        scores = np.dot(embs, history_emb)
        order = _stable_rank_order(scores, top_k)
        ranked = [candidate_list[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]
        top1 = ranked_scores[0] if ranked_scores else 0.0
        return CandidateRankingResult(candidates=ranked, scores=ranked_scores, top1_score=top1)


class CrossEncoderReranker:
    """Bi-encoder retrieve + cross-encoder rerank for sharper relevance scoring.

    Experimental — evaluated on a random 25-dialogue sample (seed=42, eval_ranking_sample.py)
    and NOT adopted as the default ranker. Bi-encoder hybrid top-1 beat static in 8/25 dialogues
    vs 5/25 with cross-encoder rerank; live_api reached top-1 in 8/25 (bi) vs 4/25 (CE).
    CE helps some cases (840, 245) but often demotes live content the bi-encoder had surfaced.
    Use via ``pipeline.ranker = CrossEncoderReranker()`` for experiments only.
    """

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        retrieve_top_n: int = 32,
    ):
        from sentence_transformers import CrossEncoder

        self.bi_encoder = SentenceBertRanker(bi_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.retrieve_top_n = retrieve_top_n

    def rank_candidates_with_scores(
        self, dialog_history: str, candidates: Iterable[KnowledgeCandidate], top_k: int = 26
    ) -> CandidateRankingResult:
        candidate_list = list(candidates)
        if not candidate_list:
            return CandidateRankingResult(candidates=[], scores=[], top1_score=0.0)

        shortlist_n = min(self.retrieve_top_n, len(candidate_list))
        bi_result = self.bi_encoder.rank_candidates_with_scores(
            dialog_history, candidate_list, top_k=shortlist_n
        )
        shortlist = bi_result.candidates
        history_text = dialog_history.strip() or " "
        pairs = [(history_text, c.text) for c in shortlist]
        ce_scores = np.array(self.cross_encoder.predict(pairs), dtype=float)
        order = _stable_rank_order(ce_scores, top_k)
        ranked = [shortlist[i] for i in order]
        ranked_scores = [float(ce_scores[i]) for i in order]
        top1 = ranked_scores[0] if ranked_scores else 0.0
        return CandidateRankingResult(candidates=ranked, scores=ranked_scores, top1_score=top1)


def rank_triplets(
    dialog_history: str,
    triplets: Iterable[Triplet],
    top_k: int = DEFAULT_TOP_K,
    ranker: Optional[SentenceBertRanker] = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Tuple[List[Triplet], float]:
    """Rank triplets by cosine similarity to the ranking query; gate on *min_score*."""
    if ranker is None:
        ranker = SentenceBertRanker()
    triplet_list = list(triplets)
    result = ranker.rank_with_scores(dialog_history, triplet_list, top_k=max(top_k, len(triplet_list)))
    candidates = [KnowledgeCandidate(text=t.as_text(), source="static_dataset", triplet=t) for t in result.triplets]
    gated = apply_score_gate(candidates, result.scores, min_score=min_score, max_k=top_k)
    ranked = [c.triplet for c in gated.candidates if c.triplet is not None]
    return ranked, gated.top1_score


def rank_candidates(
    dialog_history: str,
    candidates: Iterable[KnowledgeCandidate],
    top_k: int = DEFAULT_TOP_K,
    ranker: Optional[CandidateRanker] = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Tuple[List[KnowledgeCandidate], float]:
    pool = dedupe_knowledge_candidates(candidates)
    if ranker is None:
        ranker = SentenceBertRanker()
    result = ranker.rank_candidates_with_scores(dialog_history, pool, top_k=max(top_k, len(pool)))
    gated = apply_score_gate(result.candidates, result.scores, min_score=min_score, max_k=top_k)
    return gated.candidates, gated.top1_score
