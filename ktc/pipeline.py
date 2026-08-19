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
from ktc.passages import select_relevant_passages
from ktc.ranking import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, CandidateRanker, SentenceBertRanker, rank_candidates
from ktc.ranking_query import build_ranking_query
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_triplets


@dataclass
class HybridRunResult:
    verbalized: List[str]
    top1_similarity_score: float
    ranked_candidates: List[KnowledgeCandidate]
    live_enabled: bool


@dataclass
class KnowledgeTripletPipeline:
    """Run stages 2a–2e for one dialog turn, optionally augmented with live retrieval."""

    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    max_passages: int = 3
    openie_backend: str = "spacy"
    coref_backend: str = "none"
    verbalization_backend: str = "llm"
    ranker: Optional[CandidateRanker] = field(default=None, repr=False)
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

    def _get_ranker(self) -> CandidateRanker:
        if self.ranker is None:
            self.ranker = SentenceBertRanker()
        return self.ranker

    def get_filtered_triplets(
        self,
        knowledge_text: str,
        ranking_query: str = "",
    ) -> List[Triplet]:
        """Clean → keep relevant passages → extract → optional coref → filter."""
        cache_key = hashlib.sha256(
            f"{knowledge_text or ''}\n{ranking_query or ''}\n{self.coref_backend}".encode("utf-8")
        ).hexdigest()
        if cache_key in self._knowledge_cache:
            return self._knowledge_cache[cache_key]

        if not (ranking_query or "").strip():
            self._knowledge_cache[cache_key] = []
            return []

        passages, _scores = select_relevant_passages(
            knowledge_text,
            ranking_query,
            self._get_ranker(),
            max_passages=self.max_passages,
        )
        if not passages:
            self._knowledge_cache[cache_key] = []
            return []

        nlp = self._get_nlp()
        combined: List[Triplet] = []
        for passage in passages:
            raw = extract_triplets(passage, backend=self.openie_backend, nlp=nlp)
            resolved = resolve_coreferences(raw, passage, nlp=nlp, backend=self.coref_backend)
            combined.extend(filter_triplets(resolved))

        self._knowledge_cache[cache_key] = combined
        return combined

    def _verbalize_candidates(self, ranked: List[KnowledgeCandidate]) -> List[str]:
        result: List[Optional[str]] = [None] * len(ranked)
        static_triplets: List[Triplet] = []
        static_indices: List[int] = []

        for i, candidate in enumerate(ranked):
            if candidate.source == "static_dataset" and candidate.triplet is not None:
                static_indices.append(i)
                static_triplets.append(candidate.triplet)
            else:
                text = candidate.text.strip()
                if text and text[-1] not in ".!?":
                    text += "."
                result[i] = text

        if static_triplets:
            verbalized = verbalize_triplets(
                static_triplets,
                backend=self.verbalization_backend,
                model=self.live_config.llm_model,
                llm_config=self.live_config,
            )
            for idx, sentence in zip(static_indices, verbalized):
                result[idx] = sentence

        return [sentence for sentence in result if sentence is not None]

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
        ranking_query = build_ranking_query(dialog_history, nlp=self._get_nlp())
        if filtered is None:
            filtered = self.get_filtered_triplets(knowledge_text, ranking_query=ranking_query)

        pool = static_candidates_from_triplets(filtered)
        live_on = self.live_config.enable_live_retrieval if enable_live is None else enable_live

        if live_on:
            from dataclasses import replace

            live_cfg = self.live_config
            if not live_cfg.enable_live_retrieval:
                live_cfg = replace(self.live_config, enable_live_retrieval=True)
            victim_text = victim_utterance_from_history(dialog_history)
            live_candidates, _queries, _raw = fetch_live_knowledge(
                victim_text,
                live_cfg,
                self.api_budget,
                nlp=self._get_nlp(),
            )
            pool.extend(live_candidates)

        if not pool:
            return HybridRunResult(
                verbalized=[],
                top1_similarity_score=0.0,
                ranked_candidates=[],
                live_enabled=live_on,
            )

        query_for_rank = ranking_query or dialog_history
        ranked, top1_score = rank_candidates(
            query_for_rank,
            pool,
            top_k=self.top_k,
            min_score=self.min_score,
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
        ranking_query = build_ranking_query(dialog_history, nlp=self._get_nlp())
        if ranking_query.strip():
            passages, passage_scores = select_relevant_passages(
                knowledge_text,
                ranking_query,
                self._get_ranker(),
                max_passages=self.max_passages,
            )
        else:
            passages, passage_scores = [], []
        nlp = self._get_nlp()
        raw: List[Triplet] = []
        resolved: List[Triplet] = []
        for passage in passages:
            passage_raw = extract_triplets(passage, backend=self.openie_backend, nlp=nlp)
            raw.extend(passage_raw)
            resolved.extend(
                resolve_coreferences(passage_raw, passage, nlp=nlp, backend=self.coref_backend)
            )
        filtered = filter_triplets(resolved)
        hybrid = self.run_hybrid(
            knowledge_text,
            dialog_history,
            filtered=filtered,
            enable_live=enable_live,
        )
        cleaned = clean_knowledge_text(knowledge_text)
        return {
            "cleaned_knowledge_preview": cleaned[:500],
            "ranking_query": ranking_query,
            "selected_passages": passages,
            "passage_scores": passage_scores,
            "raw_triplets": [t.to_dict() for t in raw],
            "resolved_triplets": [t.to_dict() for t in resolved],
            "filtered_triplets": [t.to_dict() for t in filtered],
            "ranked_candidates": [c.to_dict() for c in hybrid.ranked_candidates],
            "top1_similarity_score": hybrid.top1_similarity_score,
            "live_retrieval_enabled": hybrid.live_enabled,
            "verbalized": hybrid.verbalized,
        }
