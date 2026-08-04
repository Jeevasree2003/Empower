"""Stage 2c — Coreference resolution for pronoun heads."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from ktc.triplet import Triplet

PRONOUNS = {
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "their",
    "theirs",
    "this",
    "that",
    "these",
    "those",
}


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def _collect_noun_phrases(text: str, nlp) -> List[str]:
    phrases: List[str] = []
    for sentence in _sentences(text):
        doc = nlp(sentence)
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if phrase and phrase.lower() not in PRONOUNS:
                phrases.append(phrase)
    return phrases


def _head_starts_with_pronoun(head: str) -> bool:
    first = head.strip().split()[0].lower()
    return first in PRONOUNS


def _resolve_pronoun(pronoun: str, noun_phrases: List[str]) -> Optional[str]:
    if not noun_phrases:
        return None
    pronoun = pronoun.lower()
    if pronoun in {"he", "him", "his"}:
        for phrase in reversed(noun_phrases):
            if phrase.lower().split()[0] not in PRONOUNS:
                return phrase
    if pronoun in {"she", "her", "hers"}:
        for phrase in reversed(noun_phrases):
            if phrase.lower().split()[0] not in PRONOUNS:
                return phrase
    return noun_phrases[-1]


def resolve_coreferences(triplets: Iterable[Triplet], knowledge_text: str, nlp=None) -> List[Triplet]:
    """Replace pronoun heads using noun-phrase chains over the knowledge text."""
    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm")

    noun_phrases = _collect_noun_phrases(knowledge_text, nlp)
    resolved: List[Triplet] = []

    for triplet in triplets:
        if not _head_starts_with_pronoun(triplet.head):
            resolved.append(triplet)
            continue

        pronoun = triplet.head.strip().split()[0]
        replacement = _resolve_pronoun(pronoun, noun_phrases)
        if replacement is None:
            resolved.append(triplet)
            continue

        remainder = " ".join(triplet.head.strip().split()[1:])
        new_head = f"{replacement} {remainder}".strip()
        resolved.append(Triplet(head=new_head, relation=triplet.relation, tail=triplet.tail))

    return resolved
