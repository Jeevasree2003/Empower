"""Stage 2a-pre — Clean scraped knowledge text before OpenIE extraction."""

from __future__ import annotations

import re

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
