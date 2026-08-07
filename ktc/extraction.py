"""Stage 2a — Open Information Extraction."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

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


def _span_text(token) -> str:
    return " ".join(t.text for t in token.subtree).strip()


def _expand_conjuncts(token) -> List:
    """Return *token* and any coordinated conjunct siblings."""
    group = [token]
    for child in token.children:
        if child.dep_ == "conj":
            group.extend(_expand_conjuncts(child))
    return group


def _agent_from_passive(verb_token) -> Optional[object]:
    """Find the semantic agent in a passive construction (``by`` phrase)."""
    for child in verb_token.children:
        if child.dep_ == "agent":
            for sub in child.children:
                if sub.dep_ == "pobj":
                    return sub
        if child.dep_ == "prep" and child.text.lower() == "by":
            for sub in child.children:
                if sub.dep_ == "pobj":
                    return sub
    return None


def _subjects_for_verb(verb_token) -> List:
    subjects = [t for t in verb_token.lefts if t.dep_ in {"nsubj", "nsubjpass", "csubj"}]
    if not subjects:
        subjects = [t for t in verb_token.children if t.dep_ in {"nsubj", "nsubjpass", "csubj"}]
    return subjects


def _objects_for_verb(verb_token) -> List:
    objects = [
        t
        for t in verb_token.rights
        if t.dep_ in {"dobj", "pobj", "attr", "dative", "oprd", "acomp", "obj"}
    ]
    if not objects:
        objects = [
            t for t in verb_token.children if t.dep_ in {"dobj", "pobj", "attr", "obj", "acomp"}
        ]
    return objects


def _triplets_from_verb(verb_token) -> List[Triplet]:
    """Extract one or more triplets from a single ROOT verb."""
    subjects = _subjects_for_verb(verb_token)
    objects = _objects_for_verb(verb_token)
    relation = _relation_span(verb_token)
    triplets: List[Triplet] = []

    passive_subj = next((s for s in subjects if s.dep_ == "nsubjpass"), None)
    agent = _agent_from_passive(verb_token) if passive_subj else None

    # Passive with explicit agent: semantic (agent, verb, patient) even without direct object.
    if passive_subj and agent is not None:
        head_spans = _expand_conjuncts(agent)
        tail_spans = _expand_conjuncts(passive_subj)
        for head_tok in head_spans:
            for tail_tok in tail_spans:
                head = _span_text(head_tok)
                tail = _span_text(tail_tok)
                if head and relation and tail:
                    triplets.append(Triplet(head=head, relation=relation, tail=tail))
        if triplets:
            return triplets

    if not subjects or not objects:
        return []

    head_tokens: List = []
    for subj in subjects:
        head_tokens.extend(_expand_conjuncts(subj))

    tail_tokens: List = []
    for obj in objects:
        tail_tokens.extend(_expand_conjuncts(obj))

    for head_tok in head_tokens:
        for tail_tok in tail_tokens:
            head = _span_text(head_tok)
            tail = _span_text(tail_tok)
            if head and relation and tail:
                triplets.append(Triplet(head=head, relation=relation, tail=tail))

    return triplets


def _extract_with_spacy(sentence: str, nlp) -> List[Triplet]:
    doc = nlp(sentence)
    triplets: List[Triplet] = []

    for token in doc:
        if token.dep_ != "ROOT" or token.pos_ not in {"VERB", "AUX"}:
            continue
        triplets.extend(_triplets_from_verb(token))

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
