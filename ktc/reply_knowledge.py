"""Keep only victim-facing knowledge sentences for counselor replies."""

from __future__ import annotations

import re
from typing import Iterable, List

from ktc.knowledge_item import KnowledgeCandidate

_NOT_VICTIM_FACING = re.compile(
    r"policy makers|member states|daly|economic loss|primary health care level|"
    r"online legal india|complaint request|hello team|hi tejal|"
    r"who estimates|statistical",
    re.I,
)

_GROUNDED_FACT = re.compile(
    r"helpline|112\b|181\b|kiran|icall|nalsa|fir\b|police|section\s+\d+|"
    r"crpc|ipc|bns|cybercrime|protection of women|counsel|"
    r"distress|emergency|legal aid|domestic violence|cognizable|"
    r"posh|maintenance|residence order|internal complaints|"
    r"recovery agent|intimidation",
    re.I,
)

_PHONE = re.compile(r"\d{3,}")

_ANECDOTE = re.compile(
    r"\b(my husband|my dad|my in-laws|they called|terminate me|"
    r"kicked me|tortured me|i am working|i took the)\b",
    re.I,
)


def is_victim_facing(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    return not bool(_NOT_VICTIM_FACING.search(text))


def is_reply_usable(candidate: KnowledgeCandidate) -> bool:
    """True when the sentence can be handed to a counselor as knowledge."""
    text = (candidate.text or "").strip()
    if not is_victim_facing(text):
        return False
    if candidate.source == "counseling_bank":
        return True
    if candidate.source == "static_dataset":
        return bool(_GROUNDED_FACT.search(text) or _PHONE.search(text)) and not _ANECDOTE.search(text)
    if _ANECDOTE.search(text):
        return False
    if candidate.source == "live_api":
        if re.search(r"\[\.+\.\.\]|is not rape|^--", text, re.I):
            return False
        return bool(_GROUNDED_FACT.search(text) or _PHONE.search(text))
    return bool(_GROUNDED_FACT.search(text) or _PHONE.search(text))


def assemble_reply_knowledge(
    candidates: Iterable[KnowledgeCandidate],
    top_k: int = 8,
) -> List[KnowledgeCandidate]:
    """Order clinical then legal then other, dropping anecdotes and policy text."""
    clinical: List[KnowledgeCandidate] = []
    legal: List[KnowledgeCandidate] = []
    other: List[KnowledgeCandidate] = []
    seen = set()
    for item in candidates:
        text = item.text.strip()
        key = text.lower()
        if key in seen or not is_reply_usable(item):
            continue
        seen.add(key)
        if item.domain == "clinical":
            clinical.append(item)
        elif item.domain == "legal":
            legal.append(item)
        else:
            other.append(item)
    return (clinical + legal + other)[:top_k]
