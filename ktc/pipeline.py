"""End-to-end KTC orchestration."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from ktc.coreference import resolve_coreferences
from ktc.counseling_bank import (
    DOMAIN_CLINICAL,
    DOMAIN_LEGAL,
    content_need_domains,
    counseling_candidates,
)
from ktc.entity_extraction import extract_entities
from ktc.extraction import extract_triplets
from ktc.filtering import filter_triplets
from ktc.knowledge_item import KnowledgeCandidate
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_knowledge import (
    fetch_live_knowledge,
    static_candidates_from_triplets,
    victim_utterances_from_history,
)
from ktc.passages import select_dual_domain_passages, split_knowledge_passages
from ktc.query_builder import SearchQuery, build_queries, dialogue_situations
from ktc.ranking import (
    MAX_RANKED,
    MIN_COSINE,
    CandidateRanker,
    SentenceBertRanker,
    rank_candidates,
    ranking_query_from_history,
)
from ktc.reply_knowledge import is_ktc_usable, is_reply_usable
from ktc.synthesis import DEFAULT_SYNTHESIS_MODEL, synthesize_evidence
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_triplets

_SCAM_NOISE = re.compile(
    r"romance scam|kinjal|high return scheme|fake kyc|investment scheme|"
    r"unsolicited communication via sms|awarding you a life-changing amount",
    re.I,
)
_VIOLENCE_HINT = re.compile(r"\b(murder|kill|rape|assault|drowning|intimidation)\b", re.I)


@dataclass
class HybridRunResult:
    verbalized: List[str]
    top1_similarity_score: float
    ranked_candidates: List[KnowledgeCandidate]
    live_enabled: bool
    ranking_query: str = ""
    passages_used: List[str] = field(default_factory=list)
    no_passages_used: bool = False
    counseling_bank_used: int = 0
    filtered_triplets: List[Triplet] = field(default_factory=list)
    constructed_queries: List[SearchQuery] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    situations: List[str] = field(default_factory=list)
    static_verbalized: List[str] = field(default_factory=list)
    live_verbalized: List[str] = field(default_factory=list)
    victim_span: str = ""
    supplemental_counseling: List[KnowledgeCandidate] = field(default_factory=list)
    synthesized_knowledge: Optional[str] = None


@dataclass
class KnowledgeTripletPipeline:
    """Run stages 2a–2e for one dialog turn, optionally augmented with live retrieval."""

    top_k: int = MAX_RANKED
    min_cosine: float = MIN_COSINE
    passage_top_n: int = 3
    openie_backend: str = "spacy"
    coref_backend: str = "heuristic"
    verbalization_backend: str = "llm"
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
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
            search_domains = content_need_domains(
                extract_entities(query, nlp=self._get_nlp()), query
            )
            selected = select_dual_domain_passages(
                passages,
                query,
                self._get_ranker(),
                top_n=self.passage_top_n,
                min_cosine=self.min_cosine,
                include_legal=DOMAIN_LEGAL in search_domains,
                include_clinical=DOMAIN_CLINICAL in search_domains or True,
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
            if candidate.triplet is not None:
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
        synthesize: bool = False,
    ) -> List[str]:
        return self.run_hybrid(
            knowledge_text,
            dialog_history,
            filtered=filtered,
            enable_live=enable_live,
            synthesize=synthesize,
        ).verbalized

    def run_with_score(
        self,
        knowledge_text: str,
        dialog_history: str,
        filtered: Optional[List[Triplet]] = None,
        enable_live: Optional[bool] = None,
        synthesize: bool = False,
    ) -> Tuple[List[str], float]:
        result = self.run_hybrid(
            knowledge_text,
            dialog_history,
            filtered=filtered,
            enable_live=enable_live,
            synthesize=synthesize,
        )
        return result.verbalized, result.top1_similarity_score

    def run_hybrid(
        self,
        knowledge_text: str,
        dialog_history: str,
        filtered: Optional[List[Triplet]] = None,
        enable_live: Optional[bool] = None,
        synthesize: bool = False,
    ) -> HybridRunResult:
        nlp = self._get_nlp()
        query = ranking_query_from_history(dialog_history, nlp=nlp)
        victim_span = " ".join(victim_utterances_from_history(dialog_history)[-2:])
        entities = extract_entities(victim_span, nlp=nlp)
        situations = dialogue_situations(victim_span)
        passages = split_knowledge_passages(knowledge_text)
        search_domains = content_need_domains(entities, victim_span)
        selected_passages = select_dual_domain_passages(
            passages,
            query,
            self._get_ranker(),
            top_n=self.passage_top_n,
            min_cosine=self.min_cosine,
            include_legal=DOMAIN_LEGAL in search_domains,
            include_clinical=True,
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
            else:
                filtered = []

        if query and _VIOLENCE_HINT.search(query):
            filtered = [t for t in filtered if not _SCAM_NOISE.search(t.as_text())]

        constructed = build_queries(
            entities,
            max_queries=self.live_config.max_live_queries_per_dialogue,
            victim_text=victim_span,
        )
        pool = static_candidates_from_triplets(filtered)
        live_on = self.live_config.enable_live_retrieval if enable_live is None else enable_live

        if live_on:
            live_candidates, constructed, _raw = fetch_live_knowledge(
                victim_span,
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
        ranked = [c for c in ranked if is_ktc_usable(c)]
        supplemental = counseling_candidates(entities, victim_span)
        verbalized = self._verbalize_candidates(ranked)
        static_verbalized = []
        if filtered:
            for sentence in verbalize_triplets(
                filtered[:12],
                backend="template",
                model=self.live_config.llm_model,
                llm_config=self.live_config,
            ):
                probe = KnowledgeCandidate(text=sentence, source="static_dataset")
                if is_reply_usable(probe):
                    static_verbalized.append(sentence)
        live_verbalized = [
            sentence
            for candidate, sentence in zip(ranked, verbalized)
            if candidate.source == "live_api"
        ]
        synthesized_knowledge = None
        if synthesize:
            evidence: List[KnowledgeCandidate] = []
            if len(verbalized) == len(ranked):
                evidence = [replace(candidate, text=sentence) for candidate, sentence in zip(ranked, verbalized)]
            else:
                evidence = list(ranked)
            evidence.extend(supplemental)
            synthesized_knowledge = synthesize_evidence(
                evidence,
                dialog_history,
                backend="llm",
                model=self.synthesis_model,
                llm_config=self.live_config,
            )
        return HybridRunResult(
            verbalized=verbalized,
            top1_similarity_score=top1_score,
            ranked_candidates=ranked,
            live_enabled=live_on,
            ranking_query=query,
            passages_used=passages_used,
            no_passages_used=not passages_used,
            counseling_bank_used=len(supplemental),
            filtered_triplets=filtered,
            constructed_queries=constructed,
            entities=entities,
            situations=situations,
            static_verbalized=static_verbalized,
            live_verbalized=live_verbalized,
            victim_span=victim_span,
            supplemental_counseling=supplemental,
            synthesized_knowledge=synthesized_knowledge,
        )

    def run_raw_knowledge(self, knowledge_text: str) -> List[str]:
        text = knowledge_text.strip()
        return [text] if text else []

    def inspect(
        self,
        knowledge_text: str,
        dialog_history: str,
        enable_live: Optional[bool] = None,
        synthesize: bool = False,
    ) -> dict:
        hybrid = self.run_hybrid(
            knowledge_text,
            dialog_history,
            enable_live=enable_live,
            synthesize=synthesize,
        )
        used_text = " ".join(hybrid.passages_used)
        module_knowledge = list(hybrid.verbalized)
        return {
            "query_field_note": (
                "verbalized is Stage 2e over OpenIE triplets from gated KARE passages and live pages. "
                "supplemental_counseling is trigger-matched local facts and is not mixed into verbalized. "
                "synthesized_knowledge is LLM-3 evidence synthesis over ranked + supplemental candidates; "
                "it is null unless inspect/run_hybrid is called with synthesize=True. "
                "Live ranked_candidates[].query is the Tavily search string; it is null/absent on static triplets."
            ),
            "victim_span": hybrid.victim_span,
            "entities": hybrid.entities,
            "situations": hybrid.situations,
            "ranking_query": hybrid.ranking_query,
            "constructed_queries": [q.to_dict() for q in hybrid.constructed_queries],
            "static_knowledge": {
                "passages_used": hybrid.passages_used,
                "no_passages_used": hybrid.no_passages_used,
                "filtered_triplets": [t.to_dict() for t in hybrid.filtered_triplets[:20]],
                "verbalized": hybrid.static_verbalized,
            },
            "live_knowledge": {
                "enabled": hybrid.live_enabled,
                "verbalized": hybrid.live_verbalized,
            },
            "counseling_bank_used": hybrid.counseling_bank_used,
            "supplemental_counseling": [c.to_dict() for c in hybrid.supplemental_counseling],
            "reply_knowledge": hybrid.verbalized,
            "verbalized": hybrid.verbalized,
            "synthesized_knowledge": hybrid.synthesized_knowledge,
            "module_knowledge": module_knowledge,
            "ranked_candidates": [c.to_dict() for c in hybrid.ranked_candidates],
            "top1_similarity_score": hybrid.top1_similarity_score,
            "passages_used_count": len(hybrid.passages_used),
            "passages_used_preview": [text[:180] for text in hybrid.passages_used],
            "filtered_triplet_count": len(hybrid.filtered_triplets),
            "cleaned_knowledge_preview": used_text[:240],
            "no_passages_used": hybrid.no_passages_used,
            "live_retrieval_enabled": hybrid.live_enabled,
        }
