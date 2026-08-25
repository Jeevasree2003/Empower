"""Stage 2a-pre — Clean scraped knowledge text before OpenIE extraction."""

from __future__ import annotations

import re
from typing import List, Sequence

MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)

_BOILERPLATE = (
    "you must be logged in to post a comment",
    "The Legal Team of Online Legal India",
    "Team Online Legal India",
    "will be in touch with you shortly",
    "will be in touch with you soon",
    "We have received your complaint request",
    "we appreciate your efforts in reaching out to us",
    "filing a Consumer Complaint against Mental Harassment",
    "filing a consumer complaint against Mental Harassment",
    "filing consumer complaint against Mental Harassment",
)


def clean_knowledge_text(text: str) -> str:
    """Segment and de-noise raw scraped knowledge before triplet extraction."""
    text = text.replace("\u2019", "'").replace("\ufffd", " ")
    text = re.sub(r"\s+", " ", text.strip())

    # Passage tags used in KARE-Sample.json
    text = re.sub(r"<K\d+>", "\n", text)

    # Month D, YYYY  (e.g. September 5, 2022)
    text = re.sub(
        rf"({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})",
        r"\n\1 \2, \3\n",
        text,
        flags=re.IGNORECASE,
    )
    # D Month, YYYY  (e.g. 17 May, 2021)
    text = re.sub(
        rf"(\d{{1,2}})\s+({MONTHS}),?\s+(\d{{4}})",
        r"\n\1 \2, \3\n",
        text,
        flags=re.IGNORECASE,
    )
    # Year glued to next word: 2021Hello -> 2021 Hello
    text = re.sub(r"(\d{4})([A-Za-z])", r"\1 \2", text)
    # Lowercase letter glued to month: NyaayaSeptember -> Nyaaya September
    text = re.sub(rf"([a-z])({MONTHS})", r"\1 \2", text, flags=re.IGNORECASE)
    # Missing space after punctuation before any letter (e.g. app.please, regard.You)
    text = re.sub(r"([.:;,!?])([A-Za-z])", r"\1 \2", text)
    # Lowercase glued to uppercase within a token (e.g. meNyaaya, religionNyaaya)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    lowered = text.lower()
    for phrase in _BOILERPLATE:
        idx = lowered.find(phrase.lower())
        while idx != -1:
            text = text[:idx] + text[idx + len(phrase) :]
            lowered = text.lower()
            idx = lowered.find(phrase.lower())

    sentences = []
    for chunk in re.split(r"[\n]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", chunk):
            sent = sent.strip()
            if len(sent) <= 20:
                continue
            if _is_comment_noise(sent):
                continue
            sentences.append(sent)

    return " ".join(sentences)


# Indian reporter / docket-style tokens, e.g. 2026:JHHC:16350-DB or "23 2026:JHHC:16350-DB"
_LEGAL_CITATION_RE = re.compile(
    r"(?<!\w)(?:\d{1,4}\s+)?\d{4}:[A-Z]{2,6}:\d+(?:-[A-Z]+)?(?!\w)",
    re.IGNORECASE,
)


def normalize_clause_key(text: str) -> str:
    """Case-fold, squeeze whitespace, and drop trailing punctuation for dedup."""
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".;:!?")


def strip_legal_citations(text: str) -> str:
    """Remove case-number / docket fragments from a sentence."""
    cleaned = _LEGAL_CITATION_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,;:.])", r"\1", cleaned)
    return cleaned.strip(" ,;:-")


def _clause_tokens(key: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", key)


def _tokens_are_subsequence(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    if not shorter or len(shorter) >= len(longer):
        return False
    i = 0
    for token in longer:
        if token == shorter[i]:
            i += 1
            if i == len(shorter):
                return True
    return False


def _is_redundant_clause(key: str, other: str) -> bool:
    """True when ``key`` is a strict string substring of ``other``, or a shorter token subsequence."""
    if not key or key == other:
        return False
    if key in other:
        return True
    short_tokens = _clause_tokens(key)
    long_tokens = _clause_tokens(other)
    if len(short_tokens) < 4:
        return False
    return _tokens_are_subsequence(short_tokens, long_tokens)


def drop_strict_substring_texts(texts: Sequence[str]) -> List[str]:
    """Exact-norm dedup, then drop clauses contained in a longer kept clause."""
    unique: List[str] = []
    seen = set()
    for text in texts:
        key = normalize_clause_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    keys = [normalize_clause_key(text) for text in unique]
    kept: List[str] = []
    for i, key in enumerate(keys):
        if any(_is_redundant_clause(key, other) for j, other in enumerate(keys) if i != j):
            continue
        kept.append(unique[i])
    return kept


def _is_comment_noise(sent: str) -> bool:
    """Drop scraped form replies and first-person comment-thread sentences."""
    lowered = sent.lower()
    if "online legal india" in lowered:
        return True
    if "hello team" in lowered or "hi tejal" in lowered:
        return True
    if lowered.startswith("hi i ") or lowered.startswith("hello i "):
        return True
    return False
