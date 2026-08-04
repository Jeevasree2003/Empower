"""End-to-end KTC orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ktc.coreference import resolve_coreferences
from ktc.extraction import extract_triplets
from ktc.filtering import filter_triplets
from ktc.ranking import SentenceBertRanker, rank_triplets
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_triplets


@dataclass
class KnowledgeTripletPipeline:
    """Run stages 2a–2e for one dialog turn."""

    top_k: int = 26
    openie_backend: str = "spacy"
    verbalization_backend: str = "template"
    ranker: Optional[SentenceBertRanker] = field(default=None, repr=False)
    _nlp: Optional[object] = field(default=None, repr=False)

    def _get_nlp(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def run(self, knowledge_text: str, dialog_history: str) -> List[str]:
        nlp = self._get_nlp()

        raw = extract_triplets(knowledge_text, backend=self.openie_backend, nlp=nlp)
        # Coreference must run before filtering: rule (b) drops any head without
        # a noun, and a bare pronoun head ("it", "they") is tagged as a pronoun,
        # not a noun, so pronoun-headed triplets would be deleted before the
        # coreference module ever got a chance to resolve them.
        resolved = resolve_coreferences(raw, knowledge_text, nlp=nlp)
        filtered = filter_triplets(resolved)
        ranked = rank_triplets(
            dialog_history,
            filtered,
            top_k=self.top_k,
            ranker=self.ranker,
        )
        return verbalize_triplets(ranked, backend=self.verbalization_backend)

    def run_raw_knowledge(self, knowledge_text: str) -> List[str]:
        """Ablation helper: return unstructured knowledge as a single passage."""
        text = knowledge_text.strip()
        return [text] if text else []

    def inspect(self, knowledge_text: str, dialog_history: str) -> dict:
        nlp = self._get_nlp()
        raw = extract_triplets(knowledge_text, backend=self.openie_backend, nlp=nlp)
        resolved = resolve_coreferences(raw, knowledge_text, nlp=nlp)
        filtered = filter_triplets(resolved)
        ranked = rank_triplets(
            dialog_history,
            filtered,
            top_k=self.top_k,
            ranker=self.ranker,
        )
        verbalized = verbalize_triplets(ranked, backend=self.verbalization_backend)
        return {
            "raw_triplets": [t.to_dict() for t in raw],
            "resolved_triplets": [t.to_dict() for t in resolved],
            "filtered_triplets": [t.to_dict() for t in filtered],
            "ranked_triplets": [t.to_dict() for t in ranked],
            "verbalized": verbalized,
        }
