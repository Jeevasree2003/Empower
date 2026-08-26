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
# Pronoun-only intermediate sentences are skipped and do not consume this budget
# as a successful match.
MAX_ANTECEDENT_LOOKBACK = 3
# Skip heuristic substitution on long scraped blobs; filters drop unresolved pronouns.
MAX_HEURISTIC_SENTENCES = 8
MAX_HEURISTIC_CHARS = 1600
_VAGUE_ANTECEDENT_HEADS = frozenset(
    {
        "matter",
        "deal",
        "thing",
        "things",
        "stuff",
        "lot",
        "bit",
        "one",
        "fact",
        "situation",
        "way",
        "point",
        "issue",
    }
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


def _is_vague_antecedent(chunk) -> bool:
    words = re.findall(r"[a-z']+", chunk.text.strip().lower())
    if not words:
        return True
    head = words[-1]
    return head in _VAGUE_ANTECEDENT_HEADS


def _usable_antecedent(chunk, pronoun_class: str) -> bool:
    """Reject pronouns (including NOUN mis-tags), vague NPs, and number mismatches."""
    if _is_pronoun_chunk(chunk):
        return False
    if getattr(chunk.root, "pos_", None) == "PRON":
        return False
    if _is_vague_antecedent(chunk):
        return False
    return _chunk_compatible(chunk, pronoun_class)


def _chunk_compatible(chunk, pronoun_class: str) -> bool:
    if _is_pronoun_chunk(chunk) or getattr(chunk.root, "pos_", None) == "PRON":
        return False
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
        number = chunk.root.morph.get("Number")
        if number == ["Sing"]:
            return False
        if number == ["Plur"]:
            return True
        tags = [getattr(t, "tag_", "") for t in chunk]
        if any(tag in {"NNS", "NNPS"} for tag in tags):
            return True
        # Do not accept a bare singular noun (they → law, we → the matter).
        return False

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
    usable = [c for c in compatible if _usable_antecedent(c, pronoun_class)]
    if not usable:
        return None
    if pronoun_class in {"masc", "fem"}:
        # Prefer nearest preceding PERSON/PROPN; require one for gendered pronouns
        # when multiple candidates exist (reduces wrong substitutions).
        person_chunks = [
            c
            for c in reversed(usable)
            if c.root.ent_type_ == "PERSON" or c.root.pos_ == "PROPN"
        ]
        if person_chunks:
            return person_chunks[0].text.strip()
        if len(usable) == 1:
            return usable[0].text.strip()
        return None
    return usable[-1].text.strip()


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
        compatible = [c for c in same_sent if _usable_antecedent(c, pronoun_class)]
        chosen = _pick_from_compatible(compatible, pronoun_class)
        if chosen:
            return chosen

    for offset in range(1, max_lookback + 1):
        prev_idx = sent_idx - offset
        if prev_idx < 0:
            break
        prev_candidates = _collect_candidates(sent_docs[prev_idx])
        compatible = [c for c in prev_candidates if _usable_antecedent(c, pronoun_class)]
        if not compatible:
            # Pronoun-only / incompatible sentence: keep looking further back.
            continue
        chosen = _pick_from_compatible(compatible, pronoun_class)
        if chosen:
            return chosen
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
    if len(knowledge_text) > MAX_HEURISTIC_CHARS:
        return list(triplets)

    sent_docs = _sentences(knowledge_text, nlp)
    if len(sent_docs) > MAX_HEURISTIC_SENTENCES:
        return list(triplets)

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
