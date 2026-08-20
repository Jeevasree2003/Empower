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
    {
        "it",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "they",
        "we",
        "i",
        "you",
        "me",
        "us",
        "him",
        "her",
        "them",
    }
)

# Rule 7 — reject tails that are only an unresolved pronoun.
_UNRESOLVED_PRONOUN_TAILS = frozenset(
    {"it", "this", "that", "these", "those", "he", "she", "him", "her", "them", "they", "we", "us"}
)

# Weak copula+prep relations that are almost never informative.
_WEAK_RELATIONS = re.compile(r"^(?:is|was|are|were)\s+(?:to|of)$", re.IGNORECASE)

_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

_BOILERPLATE_FRAGMENTS = (
    "logged in to post a comment",
    "will be in touch with you",
    "online legal india",
    "received your complaint request",
    "consumer complaint against mental harassment",
    "appreciate your efforts in reaching out",
)

_MAX_TAIL_WORDS = 12
_MAX_HEAD_WORDS = 10

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


def _is_garbage_span(text: str) -> bool:
    """OCR / broken tokens that should never reach verbalization."""
    lowered = text.lower()
    if re.search(r"\b(?:ca|did|wo|is|are|was)\s+n't\b", lowered):
        return True
    if re.search(r"\b(?:withoiut|wlhe|ammount|loam form|dipressed|harrasment)\b", lowered):
        return True
    words = re.findall(r"[a-z']+", lowered)
    if len(words) == 1 and len(words[0]) >= 4 and not re.search(r"[aeiou]", words[0]):
        return True
    return False


def _is_bare_pronoun_head(head: str) -> bool:
    words = re.findall(r"[a-z']+", head.strip().lower())
    return len(words) == 1 and words[0] in _GENERIC_PRONOUN_HEADS


def _is_deictic_subject_head(head: str) -> bool:
    """Comment-thread subjects like they/he/I/you — not 'my husband' or 'Our Legal Team'."""
    first = head.strip().split()[0].lower().strip(".,;:!?'\"()[]{}")
    return first in {"i", "me", "you", "we", "they", "he", "she", "it"}


def _is_unresolved_pronoun_tail(tail: str) -> bool:
    words = re.findall(r"[a-z']+", tail.strip().lower())
    return len(words) == 1 and words[0] in _UNRESOLVED_PRONOUN_TAILS


def _relation_is_stopword_only(relation: str) -> bool:
    words = re.findall(r"[a-z']+", relation.strip().lower())
    return bool(words) and all(w in _RELATION_STOPWORDS for w in words)


def _has_repeated_token(text: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) > 2]
    return any(a.lower() == b.lower() for a, b in zip(words, words[1:]))


def _token_jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z']+", a.lower()))
    sb = set(re.findall(r"[a-z']+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _contains_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(frag in lowered for frag in _BOILERPLATE_FRAGMENTS)


def _near_duplicate_head_tail(head: str, tail: str) -> bool:
    if _token_jaccard(head, tail) >= 0.8:
        return True
    h = set(re.findall(r"[a-z']+", head.lower()))
    t = set(re.findall(r"[a-z']+", tail.lower()))
    if h and t and (h <= t or t <= h) and min(len(h), len(t)) >= 2:
        return True
    return False


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
    if _is_bare_pronoun_head(triplet.head) or _is_deictic_subject_head(triplet.head):
        return False
    if _is_unresolved_pronoun_tail(triplet.tail):
        return False
    if _is_garbage_span(triplet.head) or _is_garbage_span(triplet.relation) or _is_garbage_span(triplet.tail):
        return False
    if triplet.relation.strip().lower() in {"m", "am", "'m"}:
        return False
    if _relation_is_stopword_only(triplet.relation):
        return False
    if _WEAK_RELATIONS.match(triplet.relation.strip()):
        return False
    if _has_repeated_token(triplet.as_text()):
        return False
    if _contains_boilerplate(triplet.as_text()):
        return False
    if len(triplet.tail.split()) > _MAX_TAIL_WORDS or len(triplet.head.split()) > _MAX_HEAD_WORDS:
        return False
    if _DATE_RE.search(triplet.tail) and len(triplet.tail.split()) <= 4:
        return False
    if _near_duplicate_head_tail(triplet.head, triplet.tail):
        return False
    return True


def filter_triplets(triplets: Iterable[Triplet]) -> List[Triplet]:
    return [t for t in triplets if passes_filters(t)]
