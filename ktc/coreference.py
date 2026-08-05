"""Stage 2c — Sentence-local coreference resolution for pronoun heads."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

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
    "we",
    "us",
    "our",
    "ours",
}

MASCULINE = {"he", "him", "his"}
FEMININE = {"she", "her", "hers"}
PLURAL = {"they", "them", "their", "theirs", "we", "us", "our", "ours"}
NEUTER = {"it", "its", "this", "that", "these", "those"}

_FEMININE_HINTS = frozenset(
    {"she", "her", "woman", "women", "mother", "wife", "girl", "daughter", "sister", "ms", "mrs"}
)
_MASCULINE_HINTS = frozenset(
    {"he", "him", "man", "men", "father", "husband", "boy", "son", "brother", "mr", "dad"}
)


def _sentences(text: str, nlp) -> List:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return list(nlp(text).sents)


def _head_starts_with_pronoun(head: str) -> bool:
    first = head.strip().split()[0].lower().strip(".,;:!?'\"()[]{}")
    return first in PRONOUNS


def _pronoun_class(pronoun: str) -> str:
    p = pronoun.lower()
    if p in MASCULINE:
        return "masc"
    if p in FEMININE:
        return "fem"
    if p in PLURAL:
        return "plural"
    return "neuter"


def _pronoun_token_index(sent_doc, head: str) -> Optional[int]:
    """Character offset of the pronoun token in the sentence, if found."""
    first = head.strip().split()[0].lower().strip(".,;:!?'\"()[]{}")
    for token in sent_doc:
        if token.text.lower() == first and token.pos_ == "PRON":
            return token.idx
    return None


def _find_source_sentence(triplet: Triplet, sent_docs: Sequence) -> Optional[object]:
    head_lower = triplet.head.lower()
    for sent_doc in sent_docs:
        if head_lower in sent_doc.text.lower():
            return sent_doc

    head_tokens = set(head_lower.split())
    best_doc = None
    best_overlap = 0
    for sent_doc in sent_docs:
        sent_tokens = set(sent_doc.text.lower().split())
        overlap = len(head_tokens & sent_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_doc = sent_doc
    return best_doc if best_overlap > 0 else None


def _chunk_compatible(chunk, pronoun_class: str) -> bool:
    text_lower = chunk.text.lower()
    tokens = set(re.findall(r"[a-z']+", text_lower))

    if pronoun_class == "masc":
        if tokens & _FEMININE_HINTS:
            return False
        if chunk.root.ent_type_ == "PERSON":
            return True
        return chunk.root.pos_ in {"NOUN", "PROPN"}

    if pronoun_class == "fem":
        if tokens & _MASCULINE_HINTS:
            return False
        if chunk.root.ent_type_ == "PERSON":
            return True
        return chunk.root.pos_ in {"NOUN", "PROPN"}

    if pronoun_class == "plural":
        if chunk.root.morph.get("Number") == ["Plur"]:
            return True
        if any(t.text.endswith("s") and t.pos_ in {"NOUN", "PROPN"} for t in chunk):
            return True
        return chunk.root.pos_ in {"NOUN", "PROPN"}

    # neuter: prefer non-person referents
    if chunk.root.ent_type_ == "PERSON":
        return False
    return chunk.root.pos_ in {"NOUN", "PROPN"}


def _collect_candidates(sent_doc, before_char: Optional[int] = None) -> List:
    candidates = []
    for chunk in sent_doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            continue
        if before_char is not None and chunk.start_char >= before_char:
            continue
        candidates.append(chunk)
    return candidates


def _pick_from_compatible(compatible: List, pronoun_class: str) -> Optional[str]:
  if not compatible:
    return None
  if pronoun_class in {"masc", "fem"}:
    for chunk in reversed(compatible):
      if chunk.root.ent_type_ == "PERSON" or chunk.root.pos_ == "PROPN":
        return chunk.text.strip()
    return compatible[-1].text.strip()
  return compatible[-1].text.strip()


def _pick_antecedent(sent_docs: Sequence, sent_idx: int, pronoun_char: Optional[int], pronoun_class: str):
    """Search same sentence then one preceding sentence for a local antecedent."""
    current = sent_docs[sent_idx]

    same_sent = _collect_candidates(current, before_char=pronoun_char)
    if same_sent:
        compatible = [c for c in same_sent if _chunk_compatible(c, pronoun_class)]
        chosen = _pick_from_compatible(compatible, pronoun_class)
        if chosen:
            return chosen

    if sent_idx > 0:
        prev_candidates = _collect_candidates(sent_docs[sent_idx - 1])
        compatible = [c for c in prev_candidates if _chunk_compatible(c, pronoun_class)]
        if compatible:
            if pronoun_class in {"masc", "fem"}:
                for chunk in compatible:
                    if chunk.root.ent_type_ == "PERSON" or chunk.root.pos_ == "PROPN":
                        return chunk.text.strip()
                return compatible[0].text.strip()
            return compatible[-1].text.strip()
    return None


def _resolve_pronoun_head(pronoun: str, triplet: Triplet, sent_docs: Sequence) -> Optional[str]:
    sent_doc = _find_source_sentence(triplet, sent_docs)
    if sent_doc is None:
        return None

    sent_idx = sent_docs.index(sent_doc)
    pronoun_char = _pronoun_token_index(sent_doc, triplet.head)
    pronoun_class = _pronoun_class(pronoun)
    return _pick_antecedent(sent_docs, sent_idx, pronoun_char, pronoun_class)


def resolve_coreferences(triplets: Iterable[Triplet], knowledge_text: str, nlp=None) -> List[Triplet]:
    """Replace pronoun heads using sentence-local noun-phrase antecedents."""
    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm")

    sent_docs = _sentences(knowledge_text, nlp)
    resolved: List[Triplet] = []

    for triplet in triplets:
        if not _head_starts_with_pronoun(triplet.head):
            resolved.append(triplet)
            continue

        pronoun = triplet.head.strip().split()[0].lower().strip(".,;:!?'\"()[]{}")
        replacement = _resolve_pronoun_head(pronoun, triplet, sent_docs)
        if replacement is None:
            resolved.append(triplet)
            continue

        remainder = " ".join(triplet.head.strip().split()[1:])
        new_head = f"{replacement} {remainder}".strip()
        resolved.append(Triplet(head=new_head, relation=triplet.relation, tail=triplet.tail))

    return resolved
