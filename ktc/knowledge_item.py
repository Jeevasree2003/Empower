"""Unified knowledge candidate for static + live ranking pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ktc.triplet import Triplet


@dataclass
class KnowledgeCandidate:
    """One rankable knowledge item from static KTC or live retrieval."""

    text: str
    source: str  # "static_dataset" | "live_api" | "counseling_bank"
    url: Optional[str] = None
    query: Optional[str] = None
    triplet: Optional[Triplet] = None
    domain: Optional[str] = None  # "legal" | "clinical" | None
    extraction_method: Optional[str] = None  # "openie" | "sentence_relevance"

    def to_dict(self) -> dict:
        payload = {
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "query": self.query,
            "domain": self.domain,
            "extraction_method": self.extraction_method,
        }
        if self.triplet is not None:
            payload["triplet"] = self.triplet.to_dict()
        return payload
