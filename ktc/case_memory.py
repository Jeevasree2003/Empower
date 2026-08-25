"""Process-local per-dialogue memory for situations, entities, queries, and facts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

from ktc.query_builder import SearchQuery

logger = logging.getLogger(__name__)


def _norm_query(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _entity_key(entity: Dict[str, str]) -> tuple:
    return (
        " ".join((entity.get("text") or "").strip().lower().split()),
        (entity.get("category") or "").strip().lower(),
    )


@dataclass
class CaseMemory:
    dialogue_id: str
    situations_seen: List[str] = field(default_factory=list)
    entities_seen: List[Dict[str, str]] = field(default_factory=list)
    queries_issued: Set[str] = field(default_factory=set)
    facts_retrieved: List[str] = field(default_factory=list)

    def update_situations(self, new: List[str]) -> List[str]:
        """Append unseen situation names; return only names added this call."""
        added: List[str] = []
        seen = set(self.situations_seen)
        for name in new or []:
            token = (name or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            self.situations_seen.append(token)
            added.append(token)
        if added:
            logger.info(
                "case_memory_update dialogue_id=%s new_situations=%s situations_seen=%s",
                self.dialogue_id,
                added,
                self.situations_seen,
            )
        return added

    def update_entities(self, new: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Append unseen (text, category) entities; return only those added this call."""
        added: List[Dict[str, str]] = []
        seen = {_entity_key(item) for item in self.entities_seen}
        for item in new or []:
            key = _entity_key(item)
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            stored = {"text": item.get("text", "").strip(), "category": item.get("category", "").strip()}
            self.entities_seen.append(stored)
            added.append(stored)
        if added:
            logger.info(
                "case_memory_update dialogue_id=%s new_entities=%s",
                self.dialogue_id,
                added,
            )
        return added

    def record_queries(self, queries: List[SearchQuery]) -> List[SearchQuery]:
        """Drop query texts already issued; record the rest."""
        fresh: List[SearchQuery] = []
        for query in queries or []:
            key = _norm_query(query.text)
            if not key or key in self.queries_issued:
                continue
            self.queries_issued.add(key)
            fresh.append(query)
        logger.info(
            "case_memory_queries dialogue_id=%s kept=%s issued=%s",
            self.dialogue_id,
            len(fresh),
            len(self.queries_issued),
        )
        return fresh

    def record_facts(self, facts: List[str]) -> None:
        seen = {" ".join((item or "").strip().lower().split()) for item in self.facts_retrieved}
        for fact in facts or []:
            raw = (fact or "").strip()
            key = " ".join(raw.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            self.facts_retrieved.append(raw)
