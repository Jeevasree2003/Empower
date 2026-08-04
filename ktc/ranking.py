"""Stage 2d — Relevance ranking with Sentence-BERT."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from ktc.triplet import Triplet


class SentenceBertRanker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def rank(self, dialog_history: str, triplets: Iterable[Triplet], top_k: int = 26) -> List[Triplet]:
        triplet_list = list(triplets)
        if not triplet_list:
            return []

        history_emb = self.model.encode(dialog_history, normalize_embeddings=True)
        triplet_texts = [t.as_text() for t in triplet_list]
        triplet_embs = self.model.encode(triplet_texts, normalize_embeddings=True)

        scores = np.dot(triplet_embs, history_emb)
        order = np.argsort(scores)[::-1][:top_k]
        return [triplet_list[i] for i in order]


def rank_triplets(
    dialog_history: str,
    triplets: Iterable[Triplet],
    top_k: int = 26,
    ranker: Optional[SentenceBertRanker] = None,
) -> List[Triplet]:
    if ranker is None:
        ranker = SentenceBertRanker()
    return ranker.rank(dialog_history, triplets, top_k=top_k)
