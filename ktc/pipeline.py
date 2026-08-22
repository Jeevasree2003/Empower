"""End-to-end KTC orchestration."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

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

logger = logging.getLogger(__name__)

_SCAM_NOISE = re.compile(
    r"romance scam|kinjal|high return scheme|fake kyc|investment scheme|"
    r"unsolicited communication via sms|awarding you a life-changing amount",
    re.I,
)
_VIOLENCE_HINT = re.compile(r"\b(murder|kill|rape|assault|drowning|intimidation)\b", re.I)
_CLINICAL_IN_TEXT = re.compile(
    r"\b(helpline|kiran|icall|distress|counsel|112\b|mental health|suicide)\b",
    re.I,
)
_LEGAL_IN_TEXT = re.compile(
    r"\b(fir|police|section|ipc|crpc|nalsa|181\b|pwdva|posh|legal aid|cognizable)\b",
    re.I,
)


def _normalize_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".!?")


def _dedup_texts(texts: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for text in texts:
        key = _normalize_sentence(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _dedup_ranked_candidates(candidates: Sequence[KnowledgeCandidate]) -> List[KnowledgeCandidate]:
    out: List[KnowledgeCandidate] = []
    seen = set()
    for item in candidates:
        key = _normalize_sentence(item.text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _domain_represented(verbalized: Sequence[str], domain: Optional[str]) -> bool:
    blob = " ".join(verbalized)
    if domain == DOMAIN_CLINICAL:
        return bool(_CLINICAL_IN_TEXT.search(blob))
    if domain == DOMAIN_LEGAL:
        return bool(_LEGAL_IN_TEXT.search(blob))
    return False


def assemble_final_knowledge_text(
    verbalized: Sequence[str],
    supplemental: Sequence[KnowledgeCandidate],
) -> Tuple[str, List[str]]:
    """KT for training/response generation: gated OpenIE plus missing-domain bank facts."""
    sentences: List[str] = []
    sources: List[str] = []
    seen = set()

    def add(text: str, source: str) -> None:
        raw = (text or "").strip()
        if not raw:
            return
        key = _normalize_sentence(raw)
        if not key or key in seen:
            return
        seen.add(key)
        if raw[-1] not in ".!?":
            raw += "."
        sentences.append(raw)
        if source not in sources:
            sources.append(source)

    for sentence in verbalized:
        add(sentence, "verbalized")

    if verbalized:
        for fact in supplemental:
            if fact.domain and _domain_represented(verbalized, fact.domain):
                continue
            add(fact.text, "supplemental_counseling")
    else:
        for fact in supplemental:
            add(fact.text, "supplemental_counseling")

    return " ".join(sentences), sources


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
    final_knowledge_text: str = ""
    final_knowledge_sources: List[str] = field(default_factory=list)
    live_elapsed_seconds: Optional[float] = None
    knowledge_funnel: Optional[Dict[str, object]] = None
    live_page_stats: List[Dict] = field(default_factory=list)


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
        live_elapsed_seconds = None
        live_funnel = None

        if live_on:
            started = time.monotonic()
            live_candidates, constructed, _raw, live_funnel = fetch_live_knowledge(
                victim_span,
                self.live_config,
                self.api_budget,
                nlp=nlp,
                enabled=True,
                ranker=self._get_ranker(),
                min_cosine=self.min_cosine,
            )
            live_elapsed_seconds = time.monotonic() - started
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
        ranked = _dedup_ranked_candidates(ranked)
        supplemental = counseling_candidates(entities, victim_span)
        verbalized = _dedup_texts(self._verbalize_candidates(ranked))
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
        final_knowledge_text, final_sources = assemble_final_knowledge_text(
            verbalized, supplemental
        )
        live_pages = list((live_funnel.pages if live_funnel else []) or [])
        verbalized_keys = {_normalize_sentence(text) for text in verbalized}
        for page in live_pages:
            made = []
            for text in (page.get("openie_texts") or []) + (page.get("sentence_relevance_texts") or []):
                if _normalize_sentence(text) in verbalized_keys:
                    made.append(text)
            page["made_verbalized"] = made
        funnel = {
            "live_sentences": (live_funnel.live_sentences if live_funnel else 0),
            "live_triplets": (live_funnel.live_triplets if live_funnel else 0),
            "live_sentence_relevance": (live_funnel.live_sentence_relevance if live_funnel else 0),
            "static_triplets": len(filtered or []),
            "gate_passed": len(ranked),
            "final_verbalized_count": len(verbalized),
        }
        logger.info(
            "knowledge_funnel live_sentences=%s live_triplets=%s live_sentence_relevance=%s "
            "static_triplets=%s gate_passed=%s final_verbalized_count=%s",
            funnel["live_sentences"],
            funnel["live_triplets"],
            funnel["live_sentence_relevance"],
            funnel["static_triplets"],
            funnel["gate_passed"],
            funnel["final_verbalized_count"],
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
            final_knowledge_text=final_knowledge_text,
            final_knowledge_sources=final_sources,
            live_elapsed_seconds=live_elapsed_seconds,
            knowledge_funnel=funnel,
            live_page_stats=live_pages,
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
                "final_knowledge_text is the KT for training/response generation: verbalized, plus "
                "supplemental facts whose domain is missing from verbalized (or supplemental alone "
                "when verbalized is empty). "
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
                "elapsed_seconds": hybrid.live_elapsed_seconds,
                "pages": hybrid.live_page_stats,
                "funnel": hybrid.knowledge_funnel,
            },
            "counseling_bank_used": hybrid.counseling_bank_used,
            "supplemental_counseling": [c.to_dict() for c in hybrid.supplemental_counseling],
            "reply_knowledge": hybrid.verbalized,
            "verbalized": hybrid.verbalized,
            "synthesized_knowledge": hybrid.synthesized_knowledge,
            "final_knowledge_text": hybrid.final_knowledge_text,
            "final_knowledge_sources": hybrid.final_knowledge_sources,
            "module_knowledge": module_knowledge,
            "ranked_candidates": [c.to_dict() for c in hybrid.ranked_candidates],
            "top1_similarity_score": hybrid.top1_similarity_score,
            "passages_used_count": len(hybrid.passages_used),
            "passages_used_preview": [text[:180] for text in hybrid.passages_used],
            "filtered_triplet_count": len(hybrid.filtered_triplets),
            "cleaned_knowledge_preview": used_text[:240],
            "no_passages_used": hybrid.no_passages_used,
            "live_retrieval_enabled": hybrid.live_enabled,
            "knowledge_funnel": hybrid.knowledge_funnel,
        }
