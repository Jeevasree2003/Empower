"""Stage 2b — Triplet filtering rules from the paper, plus malformed-output guards."""

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

_MAX_HEAD_TOKENS = 12
_MAX_TAIL_TOKENS = 18
_MAX_RELATION_TOKENS = 8

_BARE_PREP_COPULAS = frozenset(
    {
        "is to",
        "is of",
        "was of",
        "are of",
        "were of",
        "has been approached",
        "have been approached",
    }
)

_BOILERPLATE_RE = re.compile(
    r"team online legal|will be in touch|online legal india|hi tejal|"
    r"kindly stay calm|how may i help you|good morning from rakshak|"
    r"hope you and your family",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://|\bwww\.", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s\-()]{8,}\d)\b")
_DATE_RE = re.compile(
    r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.IGNORECASE,
)
_MALFORMED_PUNCT_RE = re.compile(r"\(\.|,\s*,|\.{2,}|\s-\s-\s")


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


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _has_repeated_token(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9']+", text or "")
    for left, right in zip(tokens, tokens[1:]):
        if left.lower() == right.lower() and len(left) > 2:
            return True
    return False


def _content_tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in {"a", "an", "the"}]


def _near_duplicate_head_tail(head: str, tail: str) -> bool:
    h = _content_tokens(head)
    t = _content_tokens(tail)
    if not h or not t:
        return False
    if h == t:
        return True
    if set(t) <= set(h) and len(set(t)) <= 2:
        return True
    overlap = len(set(h) & set(t))
    return overlap / max(len(set(h)), len(set(t))) >= 0.85


def passes_filters(triplet: Triplet) -> bool:
    if triplet.tail.strip().lower() == triplet.head.strip().lower():
        return False
    if _near_duplicate_head_tail(triplet.head, triplet.tail):
        return False
    if _token_count(triplet.head) > _MAX_HEAD_TOKENS or _token_count(triplet.tail) > _MAX_TAIL_TOKENS:
        return False
    if _token_count(triplet.relation) > _MAX_RELATION_TOKENS:
        return False
    if _has_repeated_token(triplet.head) or _has_repeated_token(triplet.tail):
        return False
    relation_norm = re.sub(r"\s+", " ", triplet.relation.strip().lower())
    if relation_norm in _BARE_PREP_COPULAS:
        return False
    blob = triplet.as_text()
    if _BOILERPLATE_RE.search(blob):
        return False
    if _URL_RE.search(blob) or _PHONE_RE.search(blob) or _DATE_RE.search(blob):
        return False
    if _MALFORMED_PUNCT_RE.search(blob):
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
