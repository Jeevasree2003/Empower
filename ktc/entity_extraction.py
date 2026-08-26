"""Stage 0 — Entity extraction from victim utterances (Table I categories)."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

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
    "torture",
    "tortured",
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

_SPACY_NER_LABELS = frozenset({"PERSON", "ORG", "GPE", "LAW", "NORP"})
SOURCE_LEXICON = "lexicon"
SOURCE_SPACY_NER = "spacy_ner"
SOURCE_NOUN_CHUNK = "noun_chunk"

_SECTION_RE = re.compile(r"\bsection\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
_IT_ACT_SECTION_RE = re.compile(r"\b(?:sec(?:tion)?\.?\s*)?66[A-Za-z]?\b", re.IGNORECASE)

logger = logging.getLogger(__name__)

SEMANTIC_ENTITY_THRESHOLD = 0.45

# Victim-phrased exemplars (not lexicon keywords). Keys under crime are comments only.
CATEGORY_EXEMPLARS: Dict[str, Sequence[str]] = {
    CATEGORY_CRIME: (
        # trafficking_or_exploitation — dialogue 111 shelter-home videos / "do business"
        "A gang makes dirty videos of these girls in the shelter and sells them for business.",
        # dialogue 1000 unsolicited obscene / sexual coercion
        "Someone sent me unsolicited obscene sexual content and pressed me to have a relationship.",
        "Men on the internet bluffed me and took all my money after a fake promise.",
        "My husband hits me at home and I am afraid he will hurt the children.",
        "A fake groom's family cheated me over a marriage meeting and robbed me.",
    ),
    CATEGORY_MENTAL_HEALTH: (
        "I cannot sleep and I feel hopeless and panicked every night.",
        "I do not know where to go or whom to talk to and my mind will not slow down.",
        "I feel empty and on edge and I am falling apart inside.",
        "I keep shaking and I cannot stop crying when I think about what happened.",
    ),
    CATEGORY_LEGAL: (
        "I want to file a case but I do not know which office will take it.",
        "I need a lawyer and a protection paper from the court.",
        "Which official number do I call so someone will record my complaint?",
        "I was told to write an application and attach evidence for the commission.",
    ),
    CATEGORY_MEDIUM: (
        "He keeps sending me messages on that photo-sharing app on my phone.",
        "The video arrived in my inbox and then another copy came by chat.",
        "Unknown numbers keep calling me and the same account appears on my feed.",
        "It started on a messenger popup and then showed up in my mail.",
    ),
}

_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\. |\? |! |; |, | and | but |\n)+",
    re.IGNORECASE,
)


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


def _category_for_ner_label(span: str, label: str) -> Optional[str]:
    if label == "LAW":
        return CATEGORY_LEGAL
    if label == "ORG" and "police" in span.lower():
        return CATEGORY_LEGAL
    return None


def _entities_from_spacy_ner(doc) -> List[Dict[str, Optional[str]]]:
    """Keep PERSON/ORG/GPE/LAW/NORP spans as a secondary pass."""
    entities: List[Dict[str, Optional[str]]] = []
    skip_bits = ("mental", "insane", "helpline", "kiran", "icall", "health")
    for ent in doc.ents:
        span = ent.text.strip()
        if ent.label_ not in _SPACY_NER_LABELS:
            continue
        if any(bit in span.lower() for bit in skip_bits):
            continue
        if len(span) < 2:
            continue
        entities.append(
            {
                "text": span,
                "category": _category_for_ner_label(span, ent.label_),
                "source": SOURCE_SPACY_NER,
            }
        )
    return entities


def _entities_from_noun_chunks(doc) -> List[Dict[str, Optional[str]]]:
    """Noun-chunk fallback when NER found nothing; skip unigrams and stopword-only chunks."""
    entities: List[Dict[str, Optional[str]]] = []
    for chunk in doc.noun_chunks:
        tokens = [tok for tok in chunk if not tok.is_space]
        if len(tokens) <= 1:
            continue
        if all(tok.is_stop or tok.is_punct for tok in tokens):
            continue
        span = chunk.text.strip()
        if len(span) < 2:
            continue
        entities.append({"text": span, "category": None, "source": SOURCE_NOUN_CHUNK})
    return entities


def _entities_from_spacy(text: str, nlp) -> List[Dict[str, str]]:
    """Legacy helper: LAW / police-ORG spans used by older call sites."""
    doc = nlp(text)
    entities: List[Dict[str, str]] = []
    for item in _entities_from_spacy_ner(doc):
        if item.get("category"):
            entities.append({"text": item["text"], "category": item["category"]})
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
    lexicon_texts = set()

    def add(text: str, category: Optional[str], source: str = SOURCE_LEXICON) -> None:
        key = (_normalize(text), category or "", source)
        if key in seen or len(text.strip()) < 2:
            return
        seen.add(key)
        item: Dict[str, str] = {"text": text.strip(), "source": source}
        if category:
            item["category"] = category
        else:
            item["category"] = None  # type: ignore[assignment]
        entities.append(item)
        if source == SOURCE_LEXICON:
            lexicon_texts.add(_normalize(text))

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

    if re.search(r"\btortur", lower):
        add("torture", CATEGORY_CRIME)
        add("abuse", CATEGORY_CRIME)

    if re.search(r"raped by\s+\d|gang\s+rape|6\s+\w+\s+men", lower):
        add("gang rape", CATEGORY_CRIME)

    for match in _SECTION_RE.finditer(victim_utterance):
        add(f"section {match.group(1)}", CATEGORY_LEGAL)

    for match in _IT_ACT_SECTION_RE.finditer(victim_utterance):
        add(match.group(0), CATEGORY_LEGAL)

    doc = nlp(victim_utterance)
    ner_hits = _entities_from_spacy_ner(doc)
    for ent in ner_hits:
        if _normalize(ent["text"]) in lexicon_texts:
            continue
        add(ent["text"], ent.get("category"), SOURCE_SPACY_NER)
    if not ner_hits:
        for ent in _entities_from_noun_chunks(doc):
            if _normalize(ent["text"]) in lexicon_texts:
                continue
            add(ent["text"], ent.get("category"), SOURCE_NOUN_CHUNK)

    return entities


def extract_entities_from_history(dialog_history: str, nlp=None) -> List[Dict[str, str]]:
    """Union entities from every victim/user utterance in formatted dialog history."""
    from ktc.live_knowledge import victim_utterances_from_history

    utterances = victim_utterances_from_history(dialog_history)
    if not utterances:
        # Legacy splitter for histories that are not role-prefixed.
        victim_lines = [line for line in dialog_history.split(" victim: ") if line.strip()]
        if not victim_lines:
            return []
        utterances = []
        for line in victim_lines:
            last = line
            if " agent: " in last:
                last = last.split(" agent: ")[0]
            if last.startswith("victim: "):
                last = last[len("victim: ") :]
            if last.strip():
                utterances.append(last.strip())
    seen = set()
    merged: List[Dict[str, str]] = []
    for utterance in utterances:
        for item in extract_entities(utterance, nlp=nlp):
            key = (
                _normalize(item.get("text") or ""),
                item.get("category") or "",
                item.get("source") or "",
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def has_confident_entities(entities: Sequence[Dict[str, str]]) -> bool:
    """True when any entity came from the lexicon or spaCy NER (not noun-chunk-only)."""
    return any(
        item.get("source") in {SOURCE_LEXICON, SOURCE_SPACY_NER} or item.get("category")
        for item in entities or []
    )


def split_utterance_clauses(victim_utterance: str) -> List[str]:
    """Split on sentence boundaries and coordinating conjunctions; drop tiny fragments."""
    text = re.sub(r"\s+", " ", (victim_utterance or "").strip())
    if not text:
        return []
    parts = _CLAUSE_SPLIT_RE.split(text)
    clauses: List[str] = []
    for part in parts:
        clause = part.strip(" .?!,;")
        if not clause:
            continue
        tokens = re.findall(r"[A-Za-z0-9']+", clause)
        if len(tokens) < 3:
            continue
        clauses.append(clause)
    return clauses


def extract_entities_semantic(
    victim_utterance: str,
    ranker,
    threshold: float = SEMANTIC_ENTITY_THRESHOLD,
) -> List[Dict[str, str]]:
    """Tag clauses by cosine similarity to CATEGORY_EXEMPLARS (no lexicon required)."""
    entities, _scores = _semantic_entities_with_scores(victim_utterance, ranker, threshold)
    return entities


def _semantic_entities_with_scores(
    victim_utterance: str,
    ranker,
    threshold: float,
) -> Tuple[List[Dict[str, str]], List[Tuple[str, str, float]]]:
    if ranker is None or not (victim_utterance or "").strip():
        return [], []
    if not hasattr(ranker, "cosine_to_query"):
        return [], []
    clauses = split_utterance_clauses(victim_utterance)
    if not clauses:
        return [], []
    seen = set()
    entities: List[Dict[str, str]] = []
    scores_out: List[Tuple[str, str, float]] = []
    for clause in clauses:
        best_by_category: Dict[str, float] = {}
        for category, exemplars in CATEGORY_EXEMPLARS.items():
            example_list = list(exemplars)
            if not example_list:
                continue
            values = ranker.cosine_to_query(clause, example_list)
            if not values:
                continue
            best_by_category[category] = max(float(score) for score in values)
        for category, score in best_by_category.items():
            if score < threshold:
                continue
            key = (_normalize(clause), category)
            if key in seen:
                continue
            seen.add(key)
            entities.append({"text": clause, "category": category})
            scores_out.append((clause, category, score))
    return entities, scores_out


def resolve_entities(
    victim_utterance: str,
    nlp=None,
    ranker=None,
    threshold: float = SEMANTIC_ENTITY_THRESHOLD,
    log: bool = True,
) -> List[Dict[str, str]]:
    """Lexicon extract_entities first; semantic clause tagging only if lexicon is empty."""
    extracted = extract_entities(victim_utterance, nlp=nlp)
    lexicon = [item for item in extracted if item.get("source") == SOURCE_LEXICON]
    if lexicon:
        if log:
            logger.info("entity_source=lexicon count=%s", len(lexicon))
        return extracted
    if ranker is None or not (victim_utterance or "").strip():
        if log:
            if extracted:
                logger.info("entity_source=spacy count=%s", len(extracted))
            else:
                logger.info("entity_source=none")
        return extracted
    semantic, scored = _semantic_entities_with_scores(victim_utterance, ranker, threshold)
    if semantic:
        if log:
            for clause, category, score in scored:
                logger.info(
                    "entity_source=semantic clause=%r category=%s score=%.3f",
                    clause,
                    category,
                    score,
                )
        return semantic
    if log:
        logger.info("entity_source=none")
    return extracted

