"""Build the text used for passage and triplet ranking."""

from __future__ import annotations

from typing import List

from ktc.entity_extraction import extract_entities
from ktc.live_knowledge import victim_utterances_from_history


def build_ranking_query(dialog_history: str, nlp=None) -> str:
    """Query from recent victim utterances plus compact entity/intent terms.

    Full mixed bot/user history is not used. Greeting-only history yields "".
    """
    utterances = victim_utterances_from_history(dialog_history)
    if not utterances:
        return ""

    recent = utterances[-2:]
    latest = recent[-1]
    parts: List[str] = []
    if len(recent) == 2:
        parts.append(recent[0])
    parts.append(latest)
    parts.append(latest)

    entities = extract_entities(latest, nlp=nlp)
    if entities:
        parts.append(" ".join(e["text"] for e in entities))
    return " ".join(p.strip() for p in parts if p.strip())
