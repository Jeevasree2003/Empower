"""Stage 2b — Triplet filtering rules from the paper."""

from __future__ import annotations

import re
from typing import Iterable, List

import nltk

from ktc.triplet import Triplet

CONJUNCTIONS = {
    "and",
    "or",
    "but",
    "nor",
    "yet",
    "so",
    "for",
    "although",
    "because",
    "since",
    "while",
    "whereas",
    "if",
    "when",
    "unless",
    "until",
    "though",
}

# Rule 6 — reject bare pronoun / deictic heads (coref failed or never ran).
_GENERIC_PRONOUN_HEADS = frozenset(
    {"it", "this", "that", "these", "those", "he", "she", "they", "we", "i", "you"}
)

# Rule 7 — reject tails that are only an unresolved pronoun.
_UNRESOLVED_PRONOUN_TAILS = frozenset(
    {"it", "this", "that", "these", "those", "he", "she", "him", "her", "them", "they", "we", "us"}
)

# Relation tokens that are not verbs on their own (stopword-only relations).
_RELATION_STOPWORDS = frozenset(
    {"a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as"}
)


def _ensure_nltk():
    """Require NLTK corpora already installed. Do not download at runtime."""
    from ktc.nltk_setup import missing_filter_resources, setup_command

    missing = missing_filter_resources()
    if missing:
        raise RuntimeError(
            "NLTK data missing: "
            + ", ".join(missing)
            + f". Run `{setup_command()}` once after install (not at filter time)."
        )


def _contains_noun(text: str) -> bool:
    _ensure_nltk()
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    return any(tag.startswith("NN") for _, tag in tags)


def _contains_verb(text: str) -> bool:
    _ensure_nltk()
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    return any(tag.startswith("VB") for _, tag in tags)


def _starts_with_conjunction(text: str) -> bool:
    first = re.split(r"\s+", text.strip())[0].lower().strip(".,;:!?'\"()[]{}")
    return first in CONJUNCTIONS


def _is_bare_pronoun_head(head: str) -> bool:
    words = re.findall(r"[a-z']+", head.strip().lower())
    return len(words) == 1 and words[0] in _GENERIC_PRONOUN_HEADS


def _is_unresolved_pronoun_tail(tail: str) -> bool:
    words = re.findall(r"[a-z']+", tail.strip().lower())
    return len(words) == 1 and words[0] in _UNRESOLVED_PRONOUN_TAILS


def _relation_is_stopword_only(relation: str) -> bool:
    words = re.findall(r"[a-z']+", relation.strip().lower())
    return bool(words) and all(w in _RELATION_STOPWORDS for w in words)


def passes_filters(triplet: Triplet) -> bool:
    if triplet.tail.strip().lower() == triplet.head.strip().lower():
        return False
    if not _contains_noun(triplet.head):
        return False
    relation_words = triplet.relation.split()
    tail_words = triplet.tail.split()
    if relation_words and tail_words and relation_words[-1].lower() == tail_words[0].lower():
        return False
    if not _contains_verb(triplet.as_text()):
        return False
    if _starts_with_conjunction(triplet.head) or _starts_with_conjunction(triplet.tail):
        return False
    if _is_bare_pronoun_head(triplet.head):
        return False
    if _is_unresolved_pronoun_tail(triplet.tail):
        return False
    if _relation_is_stopword_only(triplet.relation):
        return False
    return True


def filter_triplets(triplets: Iterable[Triplet]) -> List[Triplet]:
    return [t for t in triplets if passes_filters(t)]
