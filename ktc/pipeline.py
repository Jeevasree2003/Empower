"""End-to-end KTC orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ktc.cleaning import clean_knowledge_text
from ktc.coreference import resolve_coreferences
from ktc.extraction import extract_triplets
from ktc.filtering import filter_triplets
from ktc.knowledge_item import KnowledgeCandidate
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_knowledge import fetch_live_knowledge, static_candidates_from_triplets, victim_utterance_from_history
from ktc.ranking import SentenceBertRanker, rank_candidates, rank_triplets
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_template, verbalize_triplets


@dataclass
class HybridRunResult:
    verbalized: List[str]
    top1_similarity_score: float
    ranked_candidates: List[KnowledgeCandidate]
    live_enabled: bool


@dataclass
class KnowledgeTripletPipeline:
    """Run stages 2a–2e for one dialog turn, optionally augmented with live retrieval."""

    top_k: int = 26
    openie_backend: str = "spacy"
    verbalization_backend: str = "template"
    ranker: Optional[SentenceBertRanker] = field(default=None, repr=False)
    _nlp: Optional[object] = field(default=None, repr=False)
    _knowledge_cache: Dict[str, List[Triplet]] = field(default_factory=dict, repr=False)
    live_config: LiveRetrievalConfig = field(default_factory=LiveRetrievalConfig.load)
    api_budget: ApiCallBudget = field(default=None, repr=False)

    def __post_init__(self):
        if self.api_budget is None:
            self.api_budget = ApiCallBudget(self.live_config.max_api_calls_per_run)

    def _get_nlp(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _get_ranker(self) -> SentenceBertRanker:
        if self.ranker is None:
            self.ranker = SentenceBertRanker()
        return self.ranker

    def get_filtered_triplets(self, knowledge_text: str) -> List[Triplet]:
        """Run cleaning → extraction → coreference → filtering; cache by knowledge hash."""
        digest = hashlib.sha256((knowledge_text or "").encode("utf-8")).hexdigest()
        if digest not in self._knowledge_cache:
            cleaned = clean_knowledge_text(knowledge_text)
            nlp = self._get_nlp()
            raw = extract_triplets(cleaned, backend=self.openie_backend, nlp=nlp)
            resolved = resolve_coreferences(raw, cleaned, nlp=nlp)
            self._knowledge_cache[digest] = filter_triplets(resolved)
        return self._knowledge_cache[digest]

    def _verbalize_candidates(self, ranked: List[KnowledgeCandidate]) -> List[str]:
        sentences: List[str] = []
        for candidate in ranked:
            if candidate.source == "static_dataset" and candidate.triplet is not None:
                sentences.append(verbalize_template(candidate.triplet))
            else:
                text = candidate.text.strip()
                if text and text[-1] not in ".!?":
                    text += "."
                sentences.append(text)
        return sentences

    def run(
        self,
        knowledge_text: str,
        dialog_history: str,
        filtered: Optional[List[Triplet]] = None,
        enable_live: Optional[bool] = None,
    ) -> List[str]:
        return self.run_hybrid(
            knowledge_text, dialog_history, filtered=filtered, enable_live=enable_live
        ).verbalized

    def run_with_score(
        self,
        knowledge_text: str,
        dialog_history: str,
        filtered: Optional[List[Triplet]] = None,
        enable_live: Optional[bool] = None,
    ) -> Tuple[List[str], float]:
        result = self.run_hybrid(
            knowledge_text, dialog_history, filtered=filtered, enable_live=enable_live
        )
        return result.verbalized, result.top1_similarity_score

    def run_hybrid(
        self,
        knowledge_text: str,
        dialog_history: str,
        filtered: Optional[List[Triplet]] = None,
        enable_live: Optional[bool] = None,
    ) -> HybridRunResult:
        if filtered is None:
            filtered = self.get_filtered_triplets(knowledge_text)

        pool = static_candidates_from_triplets(filtered)
        live_on = self.live_config.enable_live_retrieval if enable_live is None else enable_live

        if live_on:
            victim_text = victim_utterance_from_history(dialog_history)
            live_candidates, _queries, _raw = fetch_live_knowledge(
                victim_text,
                self.live_config,
                self.api_budget,
                nlp=self._get_nlp(),
            )
            pool.extend(live_candidates)

        ranked, top1_score = rank_candidates(
            dialog_history,
            pool,
            top_k=self.top_k,
            ranker=self._get_ranker(),
        )
        verbalized = self._verbalize_candidates(ranked)
        return HybridRunResult(
            verbalized=verbalized,
            top1_similarity_score=top1_score,
            ranked_candidates=ranked,
            live_enabled=live_on,
        )

    def run_raw_knowledge(self, knowledge_text: str) -> List[str]:
        text = knowledge_text.strip()
        return [text] if text else []

    def inspect(self, knowledge_text: str, dialog_history: str, enable_live: Optional[bool] = None) -> dict:
        cleaned = clean_knowledge_text(knowledge_text)
        nlp = self._get_nlp()
        raw = extract_triplets(cleaned, backend=self.openie_backend, nlp=nlp)
        resolved = resolve_coreferences(raw, cleaned, nlp=nlp)
        filtered = filter_triplets(resolved)
        hybrid = self.run_hybrid(
            knowledge_text,
            dialog_history,
            filtered=filtered,
            enable_live=enable_live,
        )
        return {
            "cleaned_knowledge_preview": cleaned[:500],
            "raw_triplets": [t.to_dict() for t in raw],
            "resolved_triplets": [t.to_dict() for t in resolved],
            "filtered_triplets": [t.to_dict() for t in filtered],
            "ranked_candidates": [c.to_dict() for c in hybrid.ranked_candidates],
            "top1_similarity_score": hybrid.top1_similarity_score,
            "live_retrieval_enabled": hybrid.live_enabled,
            "verbalized": hybrid.verbalized,
        }
