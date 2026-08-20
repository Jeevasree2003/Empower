"""Stage 0.5 — Search query construction from extracted entities (Table I templates)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ktc.entity_extraction import (
    CATEGORY_CRIME,
    CATEGORY_LEGAL,
    CATEGORY_MEDIUM,
    CATEGORY_MENTAL_HEALTH,
    _MEDIUM_TERMS,
)

logger = logging.getLogger(__name__)

QUERY_SYSTEM_INSTRUCTION = """You construct web search queries for an Indian victim-support knowledge system.

Rules (mandatory):
1. Every query must be specific and unambiguous — at least 4 words, never a single keyword.
2. Queries must be answerable by government, legal, health, or verified NGO sources.
3. For legal and helpline topics, include freshness qualifiers: "2026", "current", or "latest".
4. Prefer India-specific phrasing ("in India", Indian law names) where relevant.
5. Do not produce vague queries like "help" or "abuse" alone.
"""

CURRENT_YEAR = "2026"

# Map colloquial victim language to specific search phrases (Table I specificity).
_CANONICAL_QUERY_PHRASES: Dict[str, str] = {
    "insane": "mental health distress",
    "scared": "anxiety and fear",
    "dying": "emotional distress and suicidal ideation",
    "kill": "murder or homicide threat",
    "threaten": "criminal intimidation",
    "threat": "criminal intimidation",
    "threat to life": "threat to life and domestic violence",
    "life in risk": "threat to life and domestic violence",
    "abuse": "domestic abuse or violence",
}

# IPC section numbers for indiacode retrieval (see PIPELINE.md — operative law is BNS
# since July 2024; IPC retained here for search hooks validated against indiacode.nic.in).
_CRIME_IPC_SECTION: Dict[str, str] = {
    "rape": "376",
    "murder": "302",
    "murder or homicide threat": "302",
    "homicide": "302",
    "criminal intimidation": "506",
    "bigamy": "494",
    "desertion": "498A",
}

# Legal entities that mean "lodge/FIR" procedure, not abstract definitions.
_FIR_PROCEDURE_ENTITIES = frozenset({"complaint", "complaints", "fir", "police complaint"})

# Template rank decides budget truncation. Crisis helpline is life-safety and
# must never be crowded out; crime/legal still outrank generic MH/medium queries.
_CATEGORY_PRIORITY = {
    CATEGORY_CRIME: 0,
    CATEGORY_LEGAL: 1,
    CATEGORY_MENTAL_HEALTH: 2,
    CATEGORY_MEDIUM: 3,
}

_TEMPLATE_PRIORITY = {
    "sit_help_seeking": -3,  # match the victim's "where do I go" ask first
    "sit_imminent_safety": -2,
    "mh_crisis_helpline": -1,  # life-safety: must never be crowded out by budget truncation
    "sit_homicide_report": 0,
    "sit_rape_report": 0,
    "sit_pwdva": 0,
    "sit_posh": 0,
    "sit_loan_recovery": 0,
    "sit_fraud": 0,
    "sit_missing": 0,
    "crime_statute_indiacode": 1,
    "crime_report_india": 2,
    "legal_fir_procedure": 2,
    "crime_definition": 3,
    "legal_section_definition": 4,
    "legal_section_punishment": 5,
    "crime_medium_report": 6,
    "legal_general": 7,
    "legal_helpline": 8,
    "mh_crisis_support": 9,
    "mh_symptoms": 10,
    "mh_treatment": 11,
    "medium_report": 12,
}

_SITUATION_RANK_HINTS = {
    "help_seeking": "mental health helpline whom to contact KIRAN iCall",
    "suicide_crisis": "suicide prevention helpline KIRAN 1800-599-0019",
    "homicide_threat": "murder threat FIR police criminal intimidation IPC 506",
    "rape": "rape survivor FIR IPC 376 medical legal aid",
    "kicked_out": "PWDVA residence order maintenance shared household",
    "workplace_harassment": "POSH Internal Complaints Committee workplace harassment",
    "loan_recovery": "RBI recovery agent harassment complaint",
    "investment_fraud": "cybercrime.gov.in online investment fraud",
    "missing_person": "missing person police complaint",
    "desertion_bigamy": "IPC 494 bigamy 498A cruelty",
    "domestic_violence": "domestic violence 498A PWDVA helpline 181",
}


def _canonical_phrase(entity: str, category: str) -> str:
    key = entity.strip().lower()
    if key in _CANONICAL_QUERY_PHRASES:
        return _CANONICAL_QUERY_PHRASES[key]
    return entity.strip()


def _ipc_section_for_crime(entity: str) -> Optional[str]:
    key = entity.strip().lower()
    if key in _CRIME_IPC_SECTION:
        return _CRIME_IPC_SECTION[key]
    for crime, section in _CRIME_IPC_SECTION.items():
        if crime in key or key in crime:
            return section
    return None


def _crime_report_query_text(entity: str) -> str:
    """Build crime_report_india query; avoid leading 'How to' for murder (Tavily misfire)."""
    lower = entity.lower()
    if "murder" in lower or "homicide" in lower:
        return (
            f"murder homicide offence India police FIR filing procedure official {CURRENT_YEAR}"
        )
    return f"How to report {entity} in India official procedure {CURRENT_YEAR}"


def _crime_statute_indiacode_query(entity: str) -> Optional[SearchQuery]:
    section = _ipc_section_for_crime(entity)
    if not section:
        return None
    return SearchQuery(
        f"IPC Section {section} indiacode.nic.in {entity} India",
        entity,
        CATEGORY_CRIME,
        "crime_statute_indiacode",
    )


@dataclass
class SearchQuery:
    text: str
    entity_text: str
    entity_category: str
    template: str

    def to_dict(self) -> dict:
        return {
            "query": self.text,
            "entity_text": self.entity_text,
            "entity_category": self.entity_category,
            "template": self.template,
        }


def _sq(text: str, entity: str, category: str, template: str) -> SearchQuery:
    return SearchQuery(text, entity, category, template)


def _lower(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def dialogue_situations(victim_text: str) -> List[str]:
    """Ordered situation labels inferred from the victim's own words."""
    text = _lower(victim_text)
    if not text:
        return []
    found: List[str] = []

    def add(name: str) -> None:
        if name not in found:
            found.append(name)

    if re.search(r"\b(suicid|kill myself|end my life|dying everyday|dying every day)\b", text):
        add("suicide_crisis")
    if re.search(
        r"where to go|whom to ask|who to ask|whom to contact|don't understand where|"
        r"do not understand where|going insane|i am insane",
        text,
    ):
        add("help_seeking")
    if re.search(r"\b(kill me|killing me|murder me|planning to kill|threat(?:en(?:ed)?)? to kill)\b", text):
        add("homicide_threat")
    elif re.search(r"\b(murder|homicide)\b", text) and "romance" not in text:
        add("homicide_threat")
    if re.search(r"\brape\b|gang[\s-]?rape|raped by", text):
        add("rape")
    if re.search(r"raped by\s+\d|gang[\s-]?rape|6\s+\w+\s+men", text):
        add("gang_rape")
    if re.search(r"\btortur", text):
        add("torture")
    if re.search(r"kicked me|out of the house|thrown out|shared household", text):
        add("kicked_out")
    if re.search(r"\b(posh|workplace|employer|terminate(?:d)? me)\b", text) or (
        "office" in text and re.search(r"called|harass|9\s*pm", text)
    ):
        add("workplace_harassment")
    if re.search(r"\b(loan|recovery agent|repay)\b", text):
        add("loan_recovery")
    if re.search(r"\b(invested|refund|scam|fraud)\b", text) and "kill" not in text:
        add("investment_fraud")
    if re.search(r"has not returned|did not return|not returned|\bmissing\b", text):
        add("missing_person")
    if re.search(r"another marriage|not divorced|left me", text):
        add("desertion_bigamy")
    if re.search(r"\b(husband|wife|in-laws)\b", text) and re.search(
        r"\b(beat|abuse|threat|harass|violence|kick)\b", text
    ):
        add("domestic_violence")
    if "insane" in text or "mental" in text:
        add("help_seeking")
    return found


def ranking_hints_for_dialogue(victim_text: str) -> str:
    """Extra tokens so passage ranking matches the constructed live queries."""
    parts = [_SITUATION_RANK_HINTS[name] for name in dialogue_situations(victim_text) if name in _SITUATION_RANK_HINTS]
    return " ".join(parts)


def _situation_queries(victim_text: str) -> List[SearchQuery]:
    """Queries built from the dialogue situation, not from a single keyword."""
    text = _lower(victim_text)
    queries: List[SearchQuery] = []
    year = CURRENT_YEAR
    for name in dialogue_situations(victim_text):
        if name == "help_seeking":
            queries.extend(
                [
                    _sq(
                        f"KIRAN mental health helpline 1800-599-0019 whom to contact India mohfw {year}",
                        "mental health distress",
                        CATEGORY_MENTAL_HEALTH,
                        "sit_help_seeking",
                    ),
                    _sq(
                        f"where to get mental health help in India official helpline iCall TISS {year}",
                        "mental health distress",
                        CATEGORY_MENTAL_HEALTH,
                        "sit_help_seeking",
                    ),
                ]
            )
        elif name == "suicide_crisis":
            queries.extend(
                [
                    _sq(
                        f"KIRAN 24x7 suicide prevention mental health helpline 1800-599-0019 India {year}",
                        "suicidal ideation",
                        CATEGORY_MENTAL_HEALTH,
                        "mh_crisis_helpline",
                    ),
                    _sq(
                        f"24/7 suicide prevention helpline number India {year}",
                        "suicidal ideation",
                        CATEGORY_MENTAL_HEALTH,
                        "mh_crisis_helpline",
                    ),
                ]
            )
        elif name == "homicide_threat":
            queries.extend(
                [
                    _sq(
                        f"husband threatened to kill wife India police FIR criminal intimidation IPC 506 {year}",
                        "murder or homicide threat",
                        CATEGORY_CRIME,
                        "sit_homicide_report",
                    ),
                    _sq(
                        f"IPC Section 302 506 murder threat police protection India indiacode {year}",
                        "murder or homicide threat",
                        CATEGORY_CRIME,
                        "crime_statute_indiacode",
                    ),
                ]
            )
        elif name == "rape":
            queries.extend(
                [
                    _sq(
                        f"rape survivor file FIR India CrPC 154 IPC 376 medical legal aid official {year}",
                        "rape",
                        CATEGORY_CRIME,
                        "sit_rape_report",
                    ),
                    _sq(
                        f"IPC Section 376 rape India indiacode.nic.in",
                        "rape",
                        CATEGORY_CRIME,
                        "crime_statute_indiacode",
                    ),
                ]
            )
        elif name == "gang_rape":
            queries.append(
                _sq(
                    f"IPC Section 376D gang rape delayed FIR medical examination India indiacode {year}",
                    "gang rape",
                    CATEGORY_CRIME,
                    "sit_rape_report",
                )
            )
        elif name == "torture":
            queries.append(
                _sq(
                    f"cruelty torture IPC 498A Protection of Women from Domestic Violence Act helpline 181 India {year}",
                    "torture",
                    CATEGORY_CRIME,
                    "sit_pwdva",
                )
            )
        elif name == "kicked_out":
            queries.append(
                _sq(
                    f"Protection of Women from Domestic Violence Act residence order maintenance thrown out of house India {year}",
                    "desertion",
                    CATEGORY_LEGAL,
                    "sit_pwdva",
                )
            )
        elif name == "workplace_harassment":
            queries.append(
                _sq(
                    f"POSH Act Internal Complaints Committee workplace sexual harassment India official {year}",
                    "sexual harassment",
                    CATEGORY_LEGAL,
                    "sit_posh",
                )
            )
        elif name == "loan_recovery":
            queries.append(
                _sq(
                    f"RBI guidelines loan recovery agents harassment complaint India {year}",
                    "harassment",
                    CATEGORY_LEGAL,
                    "sit_loan_recovery",
                )
            )
        elif name == "investment_fraud":
            queries.append(
                _sq(
                    f"online investment fraud report National Cyber Crime Reporting Portal cybercrime.gov.in India {year}",
                    "fraud",
                    CATEGORY_CRIME,
                    "sit_fraud",
                )
            )
        elif name == "missing_person":
            queries.append(
                _sq(
                    f"missing person complaint local police station India official procedure {year}",
                    "missing person",
                    CATEGORY_CRIME,
                    "sit_missing",
                )
            )
        elif name == "desertion_bigamy":
            queries.append(
                _sq(
                    f"IPC Section 494 bigamy 498A cruelty desertion India indiacode {year}",
                    "bigamy",
                    CATEGORY_CRIME,
                    "crime_statute_indiacode",
                )
            )
        elif name == "domestic_violence":
            queries.append(
                _sq(
                    f"domestic violence FIR IPC 498A PWDVA women helpline 181 India {year}",
                    "domestic violence",
                    CATEGORY_CRIME,
                    "sit_pwdva",
                )
            )
    # Prefer queries whose tokens overlap the utterance when two situations compete.
    if text:
        tokens = set(re.findall(r"[a-z0-9]+", text))
        queries.sort(
            key=lambda q: (
                _TEMPLATE_PRIORITY.get(q.template, 50),
                -len(tokens & set(re.findall(r"[a-z0-9]+", q.text.lower()))),
            )
        )
    return queries


def _suicidal_language(text: str) -> bool:
    return bool(re.search(r"suicid|kill myself|end my life|dying", _lower(text)))


def _crime_queries(entity: str, medium: Optional[str] = None) -> List[SearchQuery]:
    queries = [
        SearchQuery(
            f"What is {entity} under Indian law?",
            entity,
            CATEGORY_CRIME,
            "crime_definition",
        ),
        SearchQuery(
            _crime_report_query_text(entity),
            entity,
            CATEGORY_CRIME,
            "crime_report_india",
        ),
    ]
    statute = _crime_statute_indiacode_query(entity)
    if statute is not None:
        queries.append(statute)
    if medium and not _medium_is_redundant(medium, entity):
        queries.append(
            SearchQuery(
                _medium_report_query_text(entity, medium),
                entity,
                CATEGORY_CRIME,
                "crime_medium_report",
            )
        )
    return queries


_CRISIS_RAW_TERMS = frozenset({"dying", "suicidal", "suicide", "insane", "hopeless", "kill myself", "end my life"})


def _is_crisis_entity(raw_text: str, canonical: str) -> bool:
    combined = f"{raw_text} {canonical}".lower()
    if any(term in combined for term in _CRISIS_RAW_TERMS):
        return True
    return "suicidal" in combined or "ideation" in combined


def _mental_health_queries(
    entity: str, raw_entity: str = "", victim_text: str = ""
) -> List[SearchQuery]:
    combined = f"{raw_entity} {entity} {victim_text}"
    if _is_crisis_entity(raw_entity, entity) or _is_crisis_entity(victim_text, entity):
        kiran = SearchQuery(
            f"KIRAN mental health helpline 1800-599-0019 mohfw India {CURRENT_YEAR}",
            entity,
            CATEGORY_MENTAL_HEALTH,
            "mh_crisis_helpline",
        )
        if _suicidal_language(combined):
            return [
                kiran,
                SearchQuery(
                    f"24/7 suicide prevention helpline number India {CURRENT_YEAR}",
                    entity,
                    CATEGORY_MENTAL_HEALTH,
                    "mh_crisis_helpline",
                ),
                SearchQuery(
                    f"mental health crisis support helpline India {CURRENT_YEAR}",
                    entity,
                    CATEGORY_MENTAL_HEALTH,
                    "mh_crisis_support",
                ),
            ]
        return [
            kiran,
            SearchQuery(
                f"where to get mental health help whom to contact India official helpline iCall {CURRENT_YEAR}",
                entity,
                CATEGORY_MENTAL_HEALTH,
                "mh_crisis_support",
            ),
        ]
    return [
        SearchQuery(
            f"What are the symptoms of {entity}?",
            entity,
            CATEGORY_MENTAL_HEALTH,
            "mh_symptoms",
        ),
        SearchQuery(
            f"current treatment for {entity} in India {CURRENT_YEAR}",
            entity,
            CATEGORY_MENTAL_HEALTH,
            "mh_treatment",
        ),
    ]


def _legal_queries(entity: str) -> List[SearchQuery]:
    if entity.lower().startswith("section") or "66" in entity:
        return [
            SearchQuery(
                f"What is {entity} Indian Penal Code or IT Act?",
                entity,
                CATEGORY_LEGAL,
                "legal_section_definition",
            ),
            SearchQuery(
                f"punishment under {entity} India {CURRENT_YEAR} latest",
                entity,
                CATEGORY_LEGAL,
                "legal_section_punishment",
            ),
        ]
    lower = entity.strip().lower()
    if lower in _FIR_PROCEDURE_ENTITIES:
        return [
            SearchQuery(
                f"CrPC Section 154 information police station FIR procedure India indiacode {CURRENT_YEAR}",
                entity,
                CATEGORY_LEGAL,
                "legal_fir_procedure",
            ),
            SearchQuery(
                f"latest official helpline for filing police complaint in India {CURRENT_YEAR}",
                entity,
                CATEGORY_LEGAL,
                "legal_helpline",
            ),
        ]
    return [
        SearchQuery(
            f"What is {entity} in Indian law {CURRENT_YEAR}?",
            entity,
            CATEGORY_LEGAL,
            "legal_general",
        ),
        SearchQuery(
            f"latest official helpline for {entity} in India {CURRENT_YEAR}",
            entity,
            CATEGORY_LEGAL,
            "legal_helpline",
        ),
    ]


_MEDIUM_LEXICON = {t.lower() for t in _MEDIUM_TERMS}
_GENERIC_MEDIUMS = frozenset({"online", "phone", "sms", "email", "social media"})


def _normalize_medium(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _medium_is_redundant(medium: str, crime: str) -> bool:
    """True when ``on {medium}`` would repeat words already in the crime phrase."""
    medium_l = medium.strip().lower()
    crime_l = crime.strip().lower()
    if not medium_l or not crime_l:
        return True
    medium_tokens = set(medium_l.split())
    crime_tokens = set(crime_l.split())
    if medium_l in crime_l or crime_l in medium_l:
        return True
    return bool(medium_tokens & crime_tokens)


def _medium_report_query_text(crime: str, medium: str) -> str:
    if _medium_is_redundant(medium, crime):
        return f"how to report {crime} in India {CURRENT_YEAR}"
    if medium.strip().lower() in _GENERIC_MEDIUMS:
        return f"how to report {crime} {medium} in India {CURRENT_YEAR}"
    return f"how to report {crime} on {medium} in India {CURRENT_YEAR}"


def _medium_queries(entity: str, crime_hint: Optional[str] = None) -> List[SearchQuery]:
    crime = crime_hint or "online abuse"
    return [
        SearchQuery(
            _medium_report_query_text(crime, entity),
            entity,
            CATEGORY_MEDIUM,
            "medium_report",
        ),
    ]


def build_queries(
    entities: Optional[List[Dict[str, str]]] = None,
    max_queries: int = 3,
    crime_context: Optional[str] = None,
    victim_text: str = "",
) -> List[SearchQuery]:
    """Build search queries from the dialogue situation, then entity templates."""
    entities = list(entities or [])
    situations = dialogue_situations(victim_text)
    skip_templates = set()
    if situations:
        skip_templates.update(
            {
                "crime_definition",
                "crime_report_india",
                "mh_symptoms",
                "mh_treatment",
                "mh_crisis_helpline",
                "mh_crisis_support",
                "legal_general",
                "legal_helpline",
            }
        )

    medium_entities = [e["text"] for e in entities if e["category"] == CATEGORY_MEDIUM]
    primary_medium = medium_entities[0] if medium_entities else None

    if crime_context is None:
        for e in entities:
            if e["category"] == CATEGORY_CRIME:
                crime_context = e["text"]
                break

    ordered_entities = sorted(
        entities,
        key=lambda entity: _CATEGORY_PRIORITY.get(entity["category"], 99),
    )

    built: List[SearchQuery] = []
    built.extend(_situation_queries(victim_text))
    for entity in ordered_entities:
        raw_text = entity["text"]
        category = entity["category"]
        text = _canonical_phrase(raw_text, category)
        if category == CATEGORY_CRIME:
            if "homicide_threat" in situations:
                continue
            built.extend(_crime_queries(text, medium=primary_medium))
        elif category == CATEGORY_MENTAL_HEALTH:
            built.extend(_mental_health_queries(text, raw_entity=raw_text, victim_text=victim_text))
        elif category == CATEGORY_LEGAL:
            built.extend(_legal_queries(text))
        elif category == CATEGORY_MEDIUM:
            if _normalize_medium(raw_text) not in _MEDIUM_LEXICON:
                continue
            built.extend(_medium_queries(text, crime_hint=crime_context))

    seen = set()
    unique: List[SearchQuery] = []
    for q in built:
        key = q.text.lower()
        if key in seen or q.template in skip_templates:
            continue
        seen.add(key)
        unique.append(q)

    unique.sort(key=lambda q: (_TEMPLATE_PRIORITY.get(q.template, 50), q.text.lower()))
    selected = unique[:max_queries]
    for q in selected:
        logger.info(
            "constructed_query situation=%s entity=%r category=%s template=%s query=%r",
            situations,
            q.entity_text,
            q.entity_category,
            q.template,
            q.text,
        )
    return selected
