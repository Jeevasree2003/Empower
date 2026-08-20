"""End-to-end KTC orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ktc.coreference import resolve_coreferences
from ktc.extraction import extract_triplets
from ktc.filtering import filter_triplets
from ktc.knowledge_item import KnowledgeCandidate
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_knowledge import fetch_live_knowledge, static_candidates_from_triplets, victim_utterance_from_history
from ktc.passages import select_relevant_passages, split_knowledge_passages
from ktc.ranking import (
    MAX_RANKED,
    MIN_COSINE,
    CandidateRanker,
    SentenceBertRanker,
    rank_candidates,
    ranking_query_from_history,
)
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_triplets


@dataclass
class HybridRunResult:
    verbalized: List[str]
    top1_similarity_score: float
    ranked_candidates: List[KnowledgeCandidate]
    live_enabled: bool
    ranking_query: str = ""
    passages_used: List[str] = field(default_factory=list)
    no_passages_used: bool = False


@dataclass
class KnowledgeTripletPipeline:
    """Run stages 2a–2e for one dialog turn, optionally augmented with live retrieval."""

    top_k: int = MAX_RANKED
    min_cosine: float = MIN_COSINE
    passage_top_n: int = 3
    openie_backend: str = "spacy"
    coref_backend: str = "heuristic"
    verbalization_backend: str = "llm"
    ranker: Optional[CandidateRanker] = field(default=None, repr=False)
    _nlp: Optional[object] = field(default=None, repr=False)
    _passage_cache: Dict[str, List[Triplet]] = field(default_factory=dict, repr=False)
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

    def _triplets_from_passage(self, passage: str) -> List[Triplet]:
        digest = hashlib.sha256(
            f"{self.coref_backend}\n{self.openie_backend}\n{passage}".encode("utf-8")
        ).hexdigest()
        if digest not in self._passage_cache:
            nlp = self._get_nlp()
            raw = extract_triplets(passage, backend=self.openie_backend, nlp=nlp)
            resolved = resolve_coreferences(raw, passage, nlp=nlp, backend=self.coref_backend)
            self._passage_cache[digest] = filter_triplets(resolved)
        return self._passage_cache[digest]

    def get_filtered_triplets(
        self,
        knowledge_text: str,
        dialog_history: str = "",
    ) -> List[Triplet]:
        """Extract from relevant passages when a user query exists; else the full blob."""
        passages = split_knowledge_passages(knowledge_text)
        query = ranking_query_from_history(dialog_history, nlp=self._get_nlp()) if dialog_history else ""
        if query:
            selected = select_relevant_passages(
                passages,
                query,
                self._get_ranker(),
                top_n=self.passage_top_n,
                min_cosine=self.min_cosine,
            )
            chosen = [text for text, _score in selected]
        else:
            chosen = passages
        triplets: List[Triplet] = []
        seen = set()
        for passage in chosen:
            for triplet in self._triplets_from_passage(passage):
                key = (triplet.head.lower(), triplet.relation.lower(), triplet.tail.lower())
                if key not in seen:
                    seen.add(key)
                    triplets.append(triplet)
        return triplets

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
        nlp = self._get_nlp()
        query = ranking_query_from_history(dialog_history, nlp=nlp)
        passages = split_knowledge_passages(knowledge_text)
        selected_passages = select_relevant_passages(
            passages,
            query,
            self._get_ranker(),
            top_n=self.passage_top_n,
            min_cosine=self.min_cosine,
        )
        passages_used = [text for text, _score in selected_passages]

        if filtered is None:
            if query and passages_used:
                filtered = []
                seen = set()
                for passage in passages_used:
                    for triplet in self._triplets_from_passage(passage):
                        key = (triplet.head.lower(), triplet.relation.lower(), triplet.tail.lower())
                        if key not in seen:
                            seen.add(key)
                            filtered.append(triplet)
            elif query:
                filtered = []
            else:
                filtered = []

        pool = static_candidates_from_triplets(filtered)
        live_on = self.live_config.enable_live_retrieval if enable_live is None else enable_live

        if live_on:
            victim_text = victim_utterance_from_history(dialog_history)
            live_candidates, _queries, _raw = fetch_live_knowledge(
                victim_text,
                self.live_config,
                self.api_budget,
                nlp=nlp,
                enabled=True,
            )
            pool.extend(live_candidates)

        rank_text = query or dialog_history
        ranked, top1_score = rank_candidates(
            rank_text,
            pool,
            top_k=self.top_k,
            ranker=self._get_ranker(),
            min_cosine=self.min_cosine,
        )
        verbalized = self._verbalize_candidates(ranked)
        return HybridRunResult(
            verbalized=verbalized,
            top1_similarity_score=top1_score,
            ranked_candidates=ranked,
            live_enabled=live_on,
            ranking_query=query,
            passages_used=passages_used,
            no_passages_used=not passages_used,
        )

    def run_raw_knowledge(self, knowledge_text: str) -> List[str]:
        text = knowledge_text.strip()
        return [text] if text else []

    def inspect(self, knowledge_text: str, dialog_history: str, enable_live: Optional[bool] = None) -> dict:
        hybrid = self.run_hybrid(
            knowledge_text,
            dialog_history,
            enable_live=enable_live,
        )
        nlp = self._get_nlp()
        used_text = " ".join(hybrid.passages_used)
        raw = extract_triplets(used_text, backend=self.openie_backend, nlp=nlp) if used_text else []
        resolved = (
            resolve_coreferences(raw, used_text, nlp=nlp, backend=self.coref_backend) if used_text else []
        )
        filtered = filter_triplets(resolved)
        return {
            "ranking_query": hybrid.ranking_query,
            "passages_used": hybrid.passages_used,
            "no_passages_used": hybrid.no_passages_used,
            "cleaned_knowledge_preview": used_text[:500],
            "raw_triplets": [t.to_dict() for t in raw],
            "resolved_triplets": [t.to_dict() for t in resolved],
            "filtered_triplets": [t.to_dict() for t in filtered],
            "ranked_candidates": [c.to_dict() for c in hybrid.ranked_candidates],
            "top1_similarity_score": hybrid.top1_similarity_score,
            "live_retrieval_enabled": hybrid.live_enabled,
            "verbalized": hybrid.verbalized,
        }
