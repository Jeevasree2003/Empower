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


def _np_core_span(token) -> str:
    """Noun-phrase text without trailing prepositional modifiers.

    ``complaint.subtree`` otherwise swallows ``at the police station``, and a
    separate verb-level ``pobj`` then looks like a second object of ``lodge``.
    """
    drop = set()
    for child in token.children:
        if child.dep_ == "prep":
            drop.update(t.i for t in child.subtree)
    return " ".join(t.text for t in token.subtree if t.i not in drop).strip()


def _noun_prep_pobjs(token) -> List[tuple]:
    pairs: List[tuple] = []
    for child in token.children:
        if child.dep_ != "prep" or not _is_locative_prep(child):
            continue
        for pobj in child.children:
            if pobj.dep_ == "pobj":
                pairs.append((child, pobj))
    return pairs


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


_RELATIVE_PRONOUNS = frozenset({"who", "which", "that", "whom", "whose"})


def _coordination_root(verb_token):
    """Return the head verb in a coordinated verb phrase chain.

    Walk past conjunct heads even when spaCy tags the first predicate as NOUN
    (e.g. ``investigates`` in ``Cyber Cell investigates cases and prosecutes ...``).
    """
    head = verb_token
    seen = set()
    while head.dep_ == "conj" and id(head) not in seen:
        seen.add(id(head))
        parent = head.head
        if parent is head:
            break
        if _is_clause_verb(parent) or parent.dep_ in {"ROOT", "conj"}:
            head = parent
            continue
        break
    return head


def _resolve_subject_token(subject_token, verb_token):
    """Map relative-pronoun subjects to their antecedent noun phrase."""
    if verb_token.dep_ == "relcl" and subject_token.text.lower() in _RELATIVE_PRONOUNS:
        return verb_token.head
    return subject_token


def _subjects_for_verb(verb_token, inherit_conj: bool = True) -> List:
    subjects = [t for t in verb_token.lefts if t.dep_ in {"nsubj", "nsubjpass", "csubj"}]
    if not subjects:
        subjects = [t for t in verb_token.children if t.dep_ in {"nsubj", "nsubjpass", "csubj"}]
    if not subjects and inherit_conj and verb_token.dep_ == "conj":
        root = _coordination_root(verb_token)
        if root is not verb_token:
            subjects = _subjects_for_verb(root, inherit_conj=False)
    return [_resolve_subject_token(subj, verb_token) for subj in subjects]


_CORE_OBJ_DEPS = frozenset({"dobj", "obj", "attr", "dative", "oprd", "acomp", "xcomp"})
_VERB_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"})
_CLAUSE_DEPS = frozenset({"ROOT", "conj", "ccomp", "xcomp", "advcl", "relcl", "acl"})
_SUBJ_OR_OBJ_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "dobj", "obj", "attr"})
# Extra (verb, prep, pobj) triples only for locative/oblique location — not "to face" junk.
_LOCATIVE_PREPS = frozenset(
    {"at", "in", "on", "into", "onto", "from", "near", "inside", "outside", "within", "upon"}
)


def _is_clause_verb(token) -> bool:
    """True for verbs, including finite verbs spaCy mis-tags as nouns."""
    if token.pos_ in {"VERB", "AUX"}:
        return True
    if token.tag_ in _VERB_TAGS:
        return True
    if token.dep_ not in _CLAUSE_DEPS:
        return False
    if token.pos_ not in {"NOUN", "PROPN", "ADJ"}:
        return False
    return any(child.dep_ in _SUBJ_OR_OBJ_DEPS for child in token.children)


def _core_objects_for_verb(verb_token) -> List:
    """Direct/core objects only — never prepositional objects.

    Mixing ``pobj`` into this list with the same relation produces nonsense such as
    ``(victim, lodge, a police station)`` from ``lodge a complaint at the police station``.
    """
    objects = [t for t in verb_token.rights if t.dep_ in _CORE_OBJ_DEPS]
    if not objects:
        objects = [t for t in verb_token.children if t.dep_ in _CORE_OBJ_DEPS]
    return objects


def _is_locative_prep(prep_token) -> bool:
    return prep_token.text.lower() in _LOCATIVE_PREPS


def _prep_pobjs_for_verb(verb_token) -> List[tuple]:
    """Return locative ``(prep_token, pobj_token)`` pairs attached to *verb_token*."""
    pairs: List[tuple] = []
    for child in verb_token.children:
        if child.dep_ != "prep" or not _is_locative_prep(child):
            continue
        for pobj in child.children:
            if pobj.dep_ == "pobj":
                pairs.append((child, pobj))
    return pairs


def _cartesian_triplets(head_tokens, relation: str, tail_tokens, *, tail_span=_span_text) -> List[Triplet]:
    triplets: List[Triplet] = []
    for head_tok in head_tokens:
        for tail_tok in tail_tokens:
            head = _span_text(head_tok)
            tail = tail_span(tail_tok)
            if head and relation and tail:
                triplets.append(Triplet(head=head, relation=relation, tail=tail))
    return triplets


def _triplets_from_verb(verb_token) -> List[Triplet]:
    """Extract one or more triplets from a single verb token (any clause)."""
    subjects = _subjects_for_verb(verb_token)
    core_objects = _core_objects_for_verb(verb_token)
    prep_pairs = _prep_pobjs_for_verb(verb_token)
    relation = _relation_span(verb_token)
    triplets: List[Triplet] = []

    passive_subj = next((s for s in subjects if s.dep_ == "nsubjpass"), None)
    agent = _agent_from_passive(verb_token) if passive_subj else None

    # Passive with explicit agent: semantic (agent, verb, patient) even without direct object.
    if passive_subj and agent is not None:
        head_spans = _expand_conjuncts(agent)
        tail_spans = _expand_conjuncts(passive_subj)
        triplets.extend(_cartesian_triplets(head_spans, relation, tail_spans))
        if triplets:
            return triplets

    if not subjects or (not core_objects and not prep_pairs):
        return []

    head_tokens: List = []
    for subj in subjects:
        head_tokens.extend(_expand_conjuncts(subj))

    if core_objects:
        tail_tokens: List = []
        for obj in core_objects:
            tail_tokens.extend(_expand_conjuncts(obj))
        triplets.extend(
            _cartesian_triplets(head_tokens, relation, tail_tokens, tail_span=_np_core_span)
        )
        for obj in tail_tokens:
            for prep_tok, pobj in _noun_prep_pobjs(obj):
                prep_rel = f"{relation} {prep_tok.text}".strip()
                triplets.extend(
                    _cartesian_triplets(head_tokens, prep_rel, _expand_conjuncts(pobj))
                )

    # Locative/oblique arguments keep the preposition on the relation so they
    # cannot clobber a direct object under the same verb.
    for prep_tok, pobj in prep_pairs:
        prep_rel = f"{relation} {prep_tok.text}".strip()
        triplets.extend(_cartesian_triplets(head_tokens, prep_rel, _expand_conjuncts(pobj)))

    return triplets


def _extract_with_spacy(sentence: str, nlp) -> List[Triplet]:
    doc = nlp(sentence)
    triplets: List[Triplet] = []

    for token in doc:
        if not _is_clause_verb(token):
            continue
        # Skip auxiliary tokens attached to another verb (e.g. "can" in "can approach").
        if token.dep_ in {"aux", "auxpass"}:
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
