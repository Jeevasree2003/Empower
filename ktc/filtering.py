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


def _ensure_nltk():
    for resource in ("punkt", "averaged_perceptron_tagger", "wordnet"):
        try:
            nltk.data.find(
                "tokenizers/punkt"
                if resource == "punkt"
                else f"taggers/{resource}"
                if resource == "averaged_perceptron_tagger"
                else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


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
    return True


def filter_triplets(triplets: Iterable[Triplet]) -> List[Triplet]:
    return [t for t in triplets if passes_filters(t)]
