"""Stage 0 — Entity extraction from victim utterances (Table I categories)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Paper Table I style categories: crime, mental_health, legal, medium
CATEGORY_CRIME = "crime"
CATEGORY_MENTAL_HEALTH = "mental_health"
CATEGORY_LEGAL = "legal"
CATEGORY_MEDIUM = "medium"

_CRIME_TERMS = (
    "stalking",
    "cyberstalking",
    "harassment",
    "sexual harassment",
    "rape",
    "gang rape",
    "gang-rape",
    "molestation",
    "assault",
    "domestic violence",
    "abuse",
    "blackmail",
    "extortion",
    "fraud",
    "scam",
    "phishing",
    "identity theft",
    "revenge porn",
    "doxxing",
    "bullying",
    "cyberbullying",
    "threat",
    "threaten",
    "murder",
    "kill",
    "kidnapping",
    "trafficking",
    "dowry",
    "eve teasing",
    "eve-teasing",
    "life in risk",
    "desertion",
    "bigamy",
    "missing person",
)

_MENTAL_HEALTH_TERMS = (
    "depression",
    "anxiety",
    "panic attack",
    "ptsd",
    "trauma",
    "suicide",
    "suicidal",
    "self harm",
    "self-harm",
    "mental health",
    "stress",
    "insomnia",
    "bipolar",
    "schizophrenia",
    "insane",
    "scared",
    "dying",
)

_LEGAL_TERMS = (
    "fir",
    "section",
    "ipc",
    "it act",
    "posh",
    "posh act",
    "complaint",
    "helpline",
    "legal aid",
    "bail",
    "punishment",
    "penalty",
    "ncw",
    "cyber cell",
    "protection order",
)

_MEDIUM_TERMS = (
    "instagram",
    "facebook",
    "whatsapp",
    "twitter",
    "x.com",
    "snapchat",
    "telegram",
    "tiktok",
    "youtube",
    "linkedin",
    "dating app",
    "tinder",
    "bumble",
    "email",
    "sms",
    "phone",
    "online",
    "social media",
)

_SECTION_RE = re.compile(r"\bsection\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
_IT_ACT_SECTION_RE = re.compile(r"\b(?:sec(?:tion)?\.?\s*)?66[A-Za-z]?\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_lexicon_matches(text: str, terms: tuple[str, ...]) -> List[str]:
    lower = _normalize(text)
    found: List[str] = []
    for term in sorted(terms, key=len, reverse=True):
        if term in lower and term not in found:
            # avoid duplicate substrings
            if not any(term in existing for existing in found):
                found.append(term)
    return found


def _entities_from_spacy(text: str, nlp) -> List[Dict[str, str]]:
    doc = nlp(text)
    entities: List[Dict[str, str]] = []
    for ent in doc.ents:
        if ent.label_ in {"ORG", "LAW", "GPE", "PRODUCT"}:
            category = CATEGORY_LEGAL if ent.label_ == "LAW" else CATEGORY_MEDIUM
            if ent.label_ == "ORG" and "police" in ent.text.lower():
                category = CATEGORY_LEGAL
            entities.append({"text": ent.text.strip(), "category": category})
    return entities


def extract_entities(victim_utterance: str, nlp=None) -> List[Dict[str, str]]:
    """Extract tagged entities from a single victim utterance."""
    if not victim_utterance or not victim_utterance.strip():
        return []

    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm")

    seen = set()
    entities: List[Dict[str, str]] = []

    def add(text: str, category: str) -> None:
        key = (_normalize(text), category)
        if key in seen or len(text.strip()) < 2:
            return
        seen.add(key)
        entities.append({"text": text.strip(), "category": category})

    for term in _find_lexicon_matches(victim_utterance, _CRIME_TERMS):
        add(term, CATEGORY_CRIME)

    for term in _find_lexicon_matches(victim_utterance, _MENTAL_HEALTH_TERMS):
        add(term, CATEGORY_MENTAL_HEALTH)

    for term in _find_lexicon_matches(victim_utterance, _LEGAL_TERMS):
        add(term, CATEGORY_LEGAL)

    for term in _find_lexicon_matches(victim_utterance, _MEDIUM_TERMS):
        add(term, CATEGORY_MEDIUM)

    # Composite patterns for domestic-threat utterances (e.g. "husband ... kill me")
    lower = _normalize(victim_utterance)
    if "husband" in lower or "wife" in lower:
        if any(v in lower for v in ("kill", "murder", "beat", "abuse", "threat")):
            add("domestic violence", CATEGORY_CRIME)

    if re.search(r"\blife\s+is\s+in\s+risk\b", lower) or re.search(
        r"\blife\s+(?:at|in)\s+risk\b", lower
    ):
        add("threat to life", CATEGORY_CRIME)

    if re.search(
        r"another marriage|going to get another marriage|not divorced|left me",
        lower,
    ):
        add("desertion", CATEGORY_CRIME)
        add("bigamy", CATEGORY_CRIME)

    if re.search(r"has not returned|did not return|not returned|missing", lower):
        add("missing person", CATEGORY_CRIME)

    if re.search(r"kicked me|out of the house|thrown out", lower):
        add("desertion", CATEGORY_CRIME)

    if re.search(r"\b(posh|workplace|employer)\b", lower) or (
        "office" in lower and re.search(r"called|terminate|harass", lower)
    ):
        add("sexual harassment", CATEGORY_CRIME)
        add("posh", CATEGORY_LEGAL)

    if re.search(r"\b(loan|recovery agent)\b", lower):
        add("harassment", CATEGORY_CRIME)

    if re.search(r"where to go|whom to ask|who to ask|going insane", lower):
        add("mental health", CATEGORY_MENTAL_HEALTH)

    for match in _SECTION_RE.finditer(victim_utterance):
        add(f"section {match.group(1)}", CATEGORY_LEGAL)

    for match in _IT_ACT_SECTION_RE.finditer(victim_utterance):
        add(match.group(0), CATEGORY_LEGAL)

    for ent in _entities_from_spacy(victim_utterance, nlp):
        add(ent["text"], ent["category"])

    return entities


def extract_entities_from_history(dialog_history: str, nlp=None) -> List[Dict[str, str]]:
    """Extract entities from the most recent victim utterance in formatted history."""
    victim_lines = [line for line in dialog_history.split(" victim: ") if line.strip()]
    if not victim_lines:
        return []
    last = victim_lines[-1]
    # strip trailing agent turns if concatenated
    if " agent: " in last:
        last = last.split(" agent: ")[0]
    if last.startswith("victim: "):
        last = last[len("victim: ") :]
    return extract_entities(last, nlp=nlp)
