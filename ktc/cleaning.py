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
    "will be in touch with you shortly",
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
            if len(sent) > 20:
                sentences.append(sent)

    return " ".join(sentences)
