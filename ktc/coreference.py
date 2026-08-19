"""Stage 2c — Coreference resolution for pronoun heads."""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Sequence

from ktc.triplet import Triplet

logger = logging.getLogger(__name__)

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
    "its",
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

# Dummy "it" constructions — do not resolve (impersonal / extraposition).
_IMPERSONAL_IT_RE = re.compile(
    r"^\s*it\s+(?:is|was|are|were|seems|appears|becomes|became)\s+",
    re.IGNORECASE,
)

_FEMININE_HINTS = frozenset(
    {"she", "her", "woman", "women", "mother", "wife", "girl", "daughter", "sister", "ms", "mrs"}
)
_MASCULINE_HINTS = frozenset(
    {"he", "him", "man", "men", "father", "husband", "boy", "son", "brother", "mr", "dad"}
)

# How many preceding sentences the heuristic searches (same sentence always first).
MAX_ANTECEDENT_LOOKBACK = 3


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


def _is_impersonal_it(triplet: Triplet, sent_doc) -> bool:
    """Skip resolving 'It is/was ...' dummy-subject patterns."""
    first = triplet.head.strip().split()[0].lower()
    if first != "it":
        return False
    return bool(_IMPERSONAL_IT_RE.match(sent_doc.text))


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


def _is_pronoun_chunk(chunk) -> bool:
    """True when a noun-chunk is only a pronoun (including POS mis-tags)."""
    if getattr(chunk.root, "pos_", None) == "PRON":
        return True
    words = re.findall(r"[a-z']+", chunk.text.strip().lower())
    return bool(words) and all(w in PRONOUNS for w in words)


def _collect_candidates(sent_doc, before_char: Optional[int] = None) -> List:
    candidates = []
    for chunk in sent_doc.noun_chunks:
        if _is_pronoun_chunk(chunk):
            continue
        if before_char is not None and chunk.start_char >= before_char:
            continue
        candidates.append(chunk)
    return candidates


def _pick_from_compatible(compatible: List, pronoun_class: str) -> Optional[str]:
    if not compatible:
        return None
    if pronoun_class in {"masc", "fem"}:
        # Prefer nearest preceding PERSON/PROPN; require one for gendered pronouns
        # when multiple candidates exist (reduces wrong substitutions).
        person_chunks = [
            c
            for c in reversed(compatible)
            if not _is_pronoun_chunk(c)
            and (c.root.ent_type_ == "PERSON" or c.root.pos_ == "PROPN")
        ]
        if person_chunks:
            return person_chunks[0].text.strip()
        if len(compatible) == 1 and not _is_pronoun_chunk(compatible[0]):
            return compatible[0].text.strip()
        return None
    return compatible[-1].text.strip()


def _pick_antecedent(
    sent_docs: Sequence,
    sent_idx: int,
    pronoun_char: Optional[int],
    pronoun_class: str,
    max_lookback: int = MAX_ANTECEDENT_LOOKBACK,
) -> Optional[str]:
    """Search same sentence then up to *max_lookback* preceding sentences."""
    current = sent_docs[sent_idx]

    same_sent = _collect_candidates(current, before_char=pronoun_char)
    if same_sent:
        compatible = [c for c in same_sent if _chunk_compatible(c, pronoun_class)]
        chosen = _pick_from_compatible(compatible, pronoun_class)
        if chosen:
            return chosen

    for offset in range(1, max_lookback + 1):
        prev_idx = sent_idx - offset
        if prev_idx < 0:
            break
        prev_candidates = _collect_candidates(sent_docs[prev_idx])
        compatible = [c for c in prev_candidates if _chunk_compatible(c, pronoun_class)]
        if not compatible:
            continue
        if pronoun_class in {"masc", "fem"}:
            for chunk in reversed(compatible):
                if _is_pronoun_chunk(chunk):
                    continue
                if chunk.root.ent_type_ == "PERSON" or chunk.root.pos_ == "PROPN":
                    return chunk.text.strip()
            if len(compatible) == 1 and not _is_pronoun_chunk(compatible[0]):
                return compatible[0].text.strip()
            continue
        return compatible[-1].text.strip()
    return None


def _resolve_pronoun_head(
    pronoun: str,
    triplet: Triplet,
    sent_docs: Sequence,
    max_lookback: int = MAX_ANTECEDENT_LOOKBACK,
) -> Optional[str]:
    sent_doc = _find_source_sentence(triplet, sent_docs)
    if sent_doc is None:
        return None

    if _is_impersonal_it(triplet, sent_doc):
        return None

    sent_idx = sent_docs.index(sent_doc)
    pronoun_char = _pronoun_token_index(sent_doc, triplet.head)
    pronoun_class = _pronoun_class(pronoun)
    return _pick_antecedent(sent_docs, sent_idx, pronoun_char, pronoun_class, max_lookback)


def _resolve_heuristic(
    triplets: Iterable[Triplet],
    knowledge_text: str,
    nlp,
    max_lookback: int = MAX_ANTECEDENT_LOOKBACK,
) -> List[Triplet]:
    sent_docs = _sentences(knowledge_text, nlp)
    resolved: List[Triplet] = []

    for triplet in triplets:
        if not _head_starts_with_pronoun(triplet.head):
            resolved.append(triplet)
            continue

        pronoun = triplet.head.strip().split()[0].lower().strip(".,;:!?'\"()[]{}")
        replacement = _resolve_pronoun_head(pronoun, triplet, sent_docs, max_lookback)
        if replacement is None:
            resolved.append(triplet)
            continue

        remainder = " ".join(triplet.head.strip().split()[1:])
        new_head = f"{replacement} {remainder}".strip()
        resolved.append(Triplet(head=new_head, relation=triplet.relation, tail=triplet.tail))

    return resolved


def _ensure_coreferee(nlp):
    """Attach coreferee to *nlp* if not already present."""
    if nlp.has_pipe("coreferee"):
        return nlp
    try:
        import coreferee  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "coreferee is required for coref_backend='model'. "
            "Install with: pip install coreferee"
        ) from exc
    nlp.add_pipe("coreferee")
    return nlp


def _antecedent_from_coreferee(pronoun_token, doc) -> Optional[str]:
    """Resolve a pronoun token via coreferee chains (first non-pronoun mention)."""
    chains = doc._.coref_chains
    if chains is None:
        return None
    mentions = chains.resolve(pronoun_token)
    if not mentions:
        return None
    for mention in mentions:
        if mention.i == pronoun_token.i:
            continue
        span = mention.doc[mention.i : mention.i + 1]
        # Expand to noun chunk when possible
        for chunk in mention.doc.noun_chunks:
            if mention.i >= chunk.start and mention.i < chunk.end:
                if chunk.root.pos_ != "PRON":
                    return chunk.text.strip()
        if mention.pos_ != "PRON":
            return mention.text.strip()
    return None


def _resolve_model(
    triplets: Iterable[Triplet],
    knowledge_text: str,
    nlp,
) -> List[Triplet]:
    """Resolve pronouns using the optional coreferee spaCy component."""
    nlp = _ensure_coreferee(nlp)
    doc = nlp(knowledge_text)
    resolved: List[Triplet] = []

    for triplet in triplets:
        if not _head_starts_with_pronoun(triplet.head):
            resolved.append(triplet)
            continue

        pronoun_text = triplet.head.strip().split()[0].lower().strip(".,;:!?'\"()[]{}")
        replacement: Optional[str] = None

        for sent in doc.sents:
            if pronoun_text not in sent.text.lower():
                continue
            for token in sent:
                if token.text.lower() == pronoun_text and token.pos_ == "PRON":
                    replacement = _antecedent_from_coreferee(token, doc)
                    if replacement:
                        break
            if replacement:
                break

        if replacement is None:
            resolved.append(triplet)
            continue

        remainder = " ".join(triplet.head.strip().split()[1:])
        new_head = f"{replacement} {remainder}".strip()
        resolved.append(Triplet(head=new_head, relation=triplet.relation, tail=triplet.tail))

    return resolved


def resolve_coreferences(
    triplets: Iterable[Triplet],
    knowledge_text: str,
    nlp=None,
    backend: str = "heuristic",
) -> List[Triplet]:
    """Replace pronoun heads with antecedent noun phrases.

    Parameters
    ----------
    backend:
        ``heuristic`` — sentence-local search (default, no extra deps).
        ``model`` — coreferee spaCy component (``pip install coreferee``).
    """
    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm")

    if backend == "heuristic":
        return _resolve_heuristic(triplets, knowledge_text, nlp)
    if backend == "model":
        try:
            return _resolve_model(triplets, knowledge_text, nlp)
        except ImportError:
            logger.warning("coreferee unavailable; falling back to heuristic coreference")
            return _resolve_heuristic(triplets, knowledge_text, nlp)
    raise ValueError(f"Unsupported coreference backend: {backend}")
