"""Stage 0.5 — Search query construction from extracted entities (Table I templates)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ktc.entity_extraction import (
    CATEGORY_CRIME,
    CATEGORY_LEGAL,
    CATEGORY_MEDIUM,
    CATEGORY_MENTAL_HEALTH,
)

logger = logging.getLogger(__name__)

QUERY_SYSTEM_INSTRUCTION = """You construct web search queries for an Indian victim-support knowledge system.

Rules (mandatory):
1. Every query must be specific and unambiguous — at least 4 words, never a single keyword.
2. Queries must be answerable by government, legal, health, or verified NGO sources.
3. For legal and helpline topics, include freshness qualifiers: "2026", "current", or "latest".
4. Prefer India-specific phrasing ("in India", Indian law names) where relevant.
5. Do not produce vague queries like "help" or "abuse" alone.
"""

CURRENT_YEAR = "2026"

# Map colloquial victim language to specific search phrases (Table I specificity).
_CANONICAL_QUERY_PHRASES: Dict[str, str] = {
    "insane": "mental health distress",
    "scared": "anxiety and fear",
    "dying": "emotional distress and suicidal ideation",
    "kill": "murder or homicide threat",
    "threaten": "criminal intimidation",
    "threat": "criminal intimidation",
    "threat to life": "threat to life and domestic violence",
    "life in risk": "threat to life and domestic violence",
    "abuse": "domestic abuse or violence",
}


def _canonical_phrase(entity: str, category: str) -> str:
    key = entity.strip().lower()
    if key in _CANONICAL_QUERY_PHRASES:
        return _CANONICAL_QUERY_PHRASES[key]
    return entity.strip()


@dataclass
class SearchQuery:
    text: str
    entity_text: str
    entity_category: str
    template: str

    def to_dict(self) -> dict:
        return {
            "query": self.text,
            "entity_text": self.entity_text,
            "entity_category": self.entity_category,
            "template": self.template,
        }


def _crime_queries(entity: str, medium: Optional[str] = None) -> List[SearchQuery]:
    queries = [
        SearchQuery(
            f"What is {entity} under Indian law?",
            entity,
            CATEGORY_CRIME,
            "crime_definition",
        ),
        SearchQuery(
            f"How to report {entity} in India official procedure {CURRENT_YEAR}",
            entity,
            CATEGORY_CRIME,
            "crime_report_india",
        ),
    ]
    if medium:
        queries.append(
            SearchQuery(
                f"how to report {entity} on {medium} in India {CURRENT_YEAR}",
                entity,
                CATEGORY_CRIME,
                "crime_medium_report",
            )
        )
    return queries


_CRISIS_RAW_TERMS = frozenset({"dying", "suicidal", "suicide", "insane", "hopeless", "kill myself", "end my life"})


def _is_crisis_entity(raw_text: str, canonical: str) -> bool:
    combined = f"{raw_text} {canonical}".lower()
    if any(term in combined for term in _CRISIS_RAW_TERMS):
        return True
    return "suicidal" in combined or "ideation" in combined


def _mental_health_queries(entity: str, raw_entity: str = "") -> List[SearchQuery]:
    queries: List[SearchQuery] = []
    if _is_crisis_entity(raw_entity, entity):
        queries.extend(
            [
                SearchQuery(
                    f"24/7 suicide prevention helpline number India {CURRENT_YEAR}",
                    entity,
                    CATEGORY_MENTAL_HEALTH,
                    "mh_crisis_helpline",
                ),
                SearchQuery(
                    f"mental health crisis support helpline India {CURRENT_YEAR}",
                    entity,
                    CATEGORY_MENTAL_HEALTH,
                    "mh_crisis_support",
                ),
            ]
        )
    else:
        queries.append(
            SearchQuery(
                f"What are the symptoms of {entity}?",
                entity,
                CATEGORY_MENTAL_HEALTH,
                "mh_symptoms",
            )
        )
    queries.append(
        SearchQuery(
            f"current treatment for {entity} in India {CURRENT_YEAR}",
            entity,
            CATEGORY_MENTAL_HEALTH,
            "mh_treatment",
        )
    )
    return queries


def _legal_queries(entity: str) -> List[SearchQuery]:
    if entity.lower().startswith("section") or "66" in entity:
        return [
            SearchQuery(
                f"What is {entity} Indian Penal Code or IT Act?",
                entity,
                CATEGORY_LEGAL,
                "legal_section_definition",
            ),
            SearchQuery(
                f"punishment under {entity} India {CURRENT_YEAR} latest",
                entity,
                CATEGORY_LEGAL,
                "legal_section_punishment",
            ),
        ]
    return [
        SearchQuery(
            f"What is {entity} in Indian law {CURRENT_YEAR}?",
            entity,
            CATEGORY_LEGAL,
            "legal_general",
        ),
        SearchQuery(
            f"latest official helpline for {entity} in India {CURRENT_YEAR}",
            entity,
            CATEGORY_LEGAL,
            "legal_helpline",
        ),
    ]


def _medium_queries(entity: str, crime_hint: Optional[str] = None) -> List[SearchQuery]:
    crime = crime_hint or "online abuse"
    return [
        SearchQuery(
            f"how to report {crime} on {entity} in India {CURRENT_YEAR}",
            entity,
            CATEGORY_MEDIUM,
            "medium_report",
        ),
    ]


def build_queries(
    entities: List[Dict[str, str]],
    max_queries: int = 3,
    crime_context: Optional[str] = None,
) -> List[SearchQuery]:
    """Build prioritized search queries from extracted entities."""
    if not entities:
        return []

    medium_entities = [e["text"] for e in entities if e["category"] == CATEGORY_MEDIUM]
    primary_medium = medium_entities[0] if medium_entities else None

    if crime_context is None:
        for e in entities:
            if e["category"] == CATEGORY_CRIME:
                crime_context = e["text"]
                break

    built: List[SearchQuery] = []
    for entity in entities:
        raw_text = entity["text"]
        category = entity["category"]
        text = _canonical_phrase(raw_text, category)
        if category == CATEGORY_CRIME:
            built.extend(_crime_queries(text, medium=primary_medium))
        elif category == CATEGORY_MENTAL_HEALTH:
            built.extend(_mental_health_queries(text, raw_entity=raw_text))
        elif category == CATEGORY_LEGAL:
            built.extend(_legal_queries(text))
        elif category == CATEGORY_MEDIUM:
            built.extend(_medium_queries(text, crime_hint=crime_context))

    # Deduplicate while preserving order
    seen = set()
    unique: List[SearchQuery] = []
    for q in built:
        key = q.text.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    selected = unique[:max_queries]
    for q in selected:
        logger.info(
            "constructed_query entity=%r category=%s template=%s query=%r",
            q.entity_text,
            q.entity_category,
            q.template,
            q.text,
        )
    return selected
