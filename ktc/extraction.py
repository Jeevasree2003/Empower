"""Stage 2a — Open Information Extraction."""

from __future__ import annotations

import re
from typing import Iterable, List

from ktc.triplet import Triplet


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _relation_span(token) -> str:
    """Build the relation from the verb plus its aux/neg/particle children only.

    NOTE: ``token.subtree`` is rooted at the ROOT verb, so it spans the *whole*
    sentence (subject, verb, and object are all descendants of ROOT). Using it
    directly re-includes the subject text inside the relation. We instead take
    only auxiliary/negation/particle children (e.g. "does not", "give up") plus
    the verb itself, ordered by position.
    """
    aux = [t for t in token.children if t.dep_ in {"aux", "auxpass", "neg"}]
    prt = [t for t in token.children if t.dep_ == "prt"]
    span = sorted(aux + [token] + prt, key=lambda t: t.i)
    return " ".join(t.text for t in span).strip()


def _extract_with_spacy(sentence: str, nlp) -> List[Triplet]:
    doc = nlp(sentence)
    triplets: List[Triplet] = []

    for token in doc:
        if token.dep_ != "ROOT" or token.pos_ not in {"VERB", "AUX"}:
            continue

        subjects = [t for t in token.lefts if t.dep_ in {"nsubj", "nsubjpass", "csubj"}]
        if not subjects:
            subjects = [t for t in token.children if t.dep_ in {"nsubj", "nsubjpass", "csubj"}]
        objects = [
            t
            for t in token.rights
            if t.dep_ in {"dobj", "pobj", "attr", "dative", "oprd", "acomp", "obj"}
        ]
        if not objects:
            objects = [t for t in token.children if t.dep_ in {"dobj", "pobj", "attr", "obj", "acomp"}]

        if not subjects or not objects:
            continue

        head = " ".join(t.text for t in subjects[0].subtree).strip()
        relation = _relation_span(token)
        tail = " ".join(t.text for t in objects[0].subtree).strip()
        if head and relation and tail:
            triplets.append(Triplet(head=head, relation=relation, tail=tail))

    return triplets


def extract_triplets(knowledge_text: str, backend: str = "spacy", nlp=None) -> List[Triplet]:
    """Extract raw (head, relation, tail) candidates from knowledge text."""
    triplets: List[Triplet] = []
    for sentence in _sentences(knowledge_text):
        if backend == "spacy":
            if nlp is None:
                import spacy

                nlp = spacy.load("en_core_web_sm")
            triplets.extend(_extract_with_spacy(sentence, nlp))
        else:
            raise ValueError(f"Unsupported OpenIE backend: {backend}")

    deduped = []
    seen = set()
    for triplet in triplets:
        key = (triplet.head.lower(), triplet.relation.lower(), triplet.tail.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(triplet)
    return deduped
