"""Stage 0.5 — Search query construction from extracted entities (Table I templates)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    "sit_child_exploitation": 0,
    "sit_online_harassment": 0,
    "sit_identity_theft": 0,
    "sit_online_bullying": 0,
    "sit_matrimonial_fraud": 0,
    "sit_intimate_content": 0,
    "sit_financial_scam": 0,
    "sit_social_exclusion": 0,
    "sit_acid_attack": 0,
    "sit_trafficking": 0,
    "sit_cyberstalking": 0,
    "sit_exposing_personal_info": 0,
    "sit_sexual_assault": 0,
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
    "child_exploitation": "POCSO Act Childline 1098 Child Welfare Committee child sexual abuse trafficking",
    "online_harassment": "IT Act 67 67A obscene content cybercrime.gov.in online sexual coercion",
    "identity_theft": "identity theft impersonation Aadhaar bank KYC cybercrime.gov.in",
    "online_bullying": "cyberbullying online bullying school social media complaint India",
    "matrimonial_fraud": "fraudulent marriage NRI groom matrimonial scam police complaint",
    "intimate_content_sharing": "non-consensual intimate images IT Act 67A cybercrime.gov.in",
    "financial_scam": "financial scam UPI investment fraud cybercrime.gov.in RBI",
    "social_exclusion": "social exclusion ostracism community boycott support helpline India",
    "acid_attack": "acid attack FIR medical treatment compensation India",
    "trafficking": "human trafficking women children Immoral Traffic Prevention Act",
    "cyberstalking": "cyberstalking IPC 354D online stalking cybercrime.gov.in",
    "exposing_personal_information": "doxxing personal data leak IT Act privacy complaint India",
    "sexual_assault": "sexual assault molestation FIR IPC 354 medical legal aid",
}

# One descriptive exemplar per crime situation for SBERT cosine fallback.
# Regex-ladder labels are included so unmatched turns share the same embedding space.
SITUATION_EXEMPLARS: Dict[str, str] = {
    "suicide_crisis": (
        "I want to kill myself and end my life because I am dying every day from despair."
    ),
    "help_seeking": (
        "I am going insane and I do not understand where to go or whom to ask for help."
    ),
    "homicide_threat": (
        "My husband is planning to kill me and has threatened to murder me."
    ),
    "rape": (
        "I was raped last night and the men who raped me threatened me not to complain."
    ),
    "gang_rape": (
        "Several men gang raped me and I need to file a delayed FIR and get a medical exam."
    ),
    "torture": (
        "My father tortures and beats my grandmother for property and I need medical help."
    ),
    "kicked_out": (
        "My husband kicked me out of the house with the children and I have nowhere to stay."
    ),
    "workplace_harassment": (
        "My employer sexually harassed me at the office and then terminated me after I protested."
    ),
    "loan_recovery": (
        "Loan recovery agents are threatening and shaming me at odd hours to repay money."
    ),
    "investment_fraud": (
        "I invested money in an online scheme and they refuse to refund me after the scam."
    ),
    "missing_person": (
        "A family member has not returned home and I need to file a missing person complaint."
    ),
    "desertion_bigamy": (
        "My husband left me and entered another marriage even though we are not divorced."
    ),
    "domestic_violence": (
        "My husband and in-laws beat and abuse me at home and I fear more violence."
    ),
    "child_exploitation": (
        "A shelter home owner sexually assaults teen girls in his care and exploits children and minors."
    ),
    "online_harassment": (
        "Someone sent unsolicited obscene sexual content on social media messenger and coerced me into a sexual relationship."
    ),
    "identity_theft": (
        "Someone stole my identity and is using my name, Aadhaar, and bank details to impersonate me."
    ),
    "online_bullying": (
        "Classmates and peers mock, name-call, and bully me every day on social media and messaging apps."
    ),
    "matrimonial_fraud": (
        "A fake NRI groom and his family cheated me in a fraudulent marriage proposal and robbed me."
    ),
    "intimate_content_sharing": (
        "Someone is sharing my private intimate photos and videos without my consent to defame me."
    ),
    "financial_scam": (
        "Fraudsters tricked me into a UPI or bank transfer and a fake investment financial scam."
    ),
    "social_exclusion": (
        "My community has ostracized and socially excluded me, and I am isolated from family and neighbours."
    ),
    "acid_attack": (
        "Someone threw acid on my face and I need emergency medical care and to file a police complaint."
    ),
    "trafficking": (
        "Women and children are being sold, transported, and held against their will in a trafficking racket."
    ),
    "cyberstalking": (
        "A stalker follows me online, tracks my accounts, and sends repeated threatening messages on Instagram."
    ),
    "exposing_personal_information": (
        "Someone published my phone number, address, and private personal information online without consent."
    ),
    "sexual_assault": (
        "A man touched my private parts without consent and sexually assaulted me when I protested."
    ),
}

SEMANTIC_SITUATION_THRESHOLD = 0.45
SEMANTIC_SITUATION_MAX = 3


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
        f"IPC Section {section} indiacode.nic.in indiankanoon.org {entity} India",
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


def _situation_cosine_scores(victim_text: str, ranker) -> Dict[str, float]:
    """Cosine of victim_text against every SITUATION_EXEMPLARS sentence."""
    names = list(SITUATION_EXEMPLARS.keys())
    exemplars = [SITUATION_EXEMPLARS[name] for name in names]
    scores: Dict[str, float] = {name: 0.0 for name in names}
    if not (victim_text or "").strip() or ranker is None or not exemplars:
        return scores
    if hasattr(ranker, "cosine_to_query"):
        values = ranker.cosine_to_query(victim_text, exemplars)
        for name, score in zip(names, values):
            scores[name] = float(score)
        return scores
    model = getattr(ranker, "model", None)
    if model is None:
        return scores
    from ktc.ranking import encode_texts_cached
    import numpy as np

    batch = encode_texts_cached(model, [victim_text.strip(), *exemplars])
    query_emb = batch[0]
    text_embs = batch[1:]
    dots = np.dot(text_embs, query_emb)
    for name, score in zip(names, dots):
        scores[name] = float(score)
    return scores


def dialogue_situations_semantic(
    victim_text: str,
    ranker,
    threshold: float = SEMANTIC_SITUATION_THRESHOLD,
) -> List[str]:
    """Regex ladder first; if empty, SBERT cosine against situation exemplars."""
    names, _meta = resolve_dialogue_situations(victim_text, ranker=ranker, threshold=threshold)
    return names


def resolve_dialogue_situations(
    victim_text: str,
    ranker=None,
    threshold: float = SEMANTIC_SITUATION_THRESHOLD,
    log: bool = True,
) -> Tuple[List[str], Dict[str, object]]:
    """Situations plus source metadata for Stage 0.5 logging."""
    regex = dialogue_situations(victim_text)
    if regex:
        if log:
            logger.info("situation_source=regex categories=%s", ",".join(regex))
        return regex, {"source": "regex", "scores": {}}
    if not (victim_text or "").strip() or ranker is None:
        if log:
            logger.info("situation_source=semantic score= category=")
        return [], {"source": "semantic", "scores": {}}
    scores = _situation_cosine_scores(victim_text, ranker)
    ranked = sorted(
        ((name, score) for name, score in scores.items() if score >= threshold),
        key=lambda item: (-item[1], item[0]),
    )
    names = [name for name, _score in ranked[:SEMANTIC_SITUATION_MAX]]
    if log:
        if names:
            top_name, top_score = ranked[0]
            logger.info(
                "situation_source=semantic score=%.3f category=%s",
                top_score,
                top_name,
            )
        else:
            logger.info("situation_source=semantic score= category=")
    meta_scores = {name: scores[name] for name in names}
    return names, {"source": "semantic", "scores": meta_scores}


def ranking_hints_for_dialogue(victim_text: str, ranker=None) -> str:
    """Extra tokens so passage ranking matches the constructed live queries."""
    situations, _meta = resolve_dialogue_situations(victim_text, ranker=ranker, log=False)
    parts = [_SITUATION_RANK_HINTS[name] for name in situations if name in _SITUATION_RANK_HINTS]
    return " ".join(parts)


def _situation_queries(
    victim_text: str,
    situations: Optional[Sequence[str]] = None,
) -> List[SearchQuery]:
    """Queries built from the dialogue situation, not from a single keyword."""
    text = _lower(victim_text)
    queries: List[SearchQuery] = []
    year = CURRENT_YEAR
    names = list(situations) if situations is not None else dialogue_situations(victim_text)
    for name in names:
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
                        f"IPC Section 302 506 murder threat police protection India indiacode.nic.in indiankanoon.org {year}",
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
                        f"IPC Section 376 rape India indiacode.nic.in indiankanoon.org",
                        "rape",
                        CATEGORY_CRIME,
                        "crime_statute_indiacode",
                    ),
                ]
            )
        elif name == "gang_rape":
            queries.append(
                _sq(
                    f"IPC Section 376D gang rape delayed FIR medical examination India indiacode.nic.in indiankanoon.org {year}",
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
                    f"IPC Section 494 bigamy 498A cruelty desertion India indiacode.nic.in indiankanoon.org {year}",
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
        elif name == "child_exploitation":
            queries.extend(
                [
                    _sq(
                        f"POCSO Act child sexual abuse trafficking report Child Welfare Committee India {year}",
                        "child sexual exploitation",
                        CATEGORY_CRIME,
                        "sit_child_exploitation",
                    ),
                    _sq(
                        f"Childline 1098 24x7 helpline child in distress India official {year}",
                        "child sexual exploitation",
                        CATEGORY_CRIME,
                        "sit_child_exploitation",
                    ),
                ]
            )
        elif name == "online_harassment":
            queries.extend(
                [
                    _sq(
                        f"IT Act Section 67 67A obscene sexual content online India indiacode.nic.in {year}",
                        "online sexual harassment",
                        CATEGORY_CRIME,
                        "sit_online_harassment",
                    ),
                    _sq(
                        f"report unsolicited obscene messages social media National Cyber Crime Reporting Portal cybercrime.gov.in {year}",
                        "online sexual harassment",
                        CATEGORY_CRIME,
                        "sit_online_harassment",
                    ),
                ]
            )
        elif name == "identity_theft":
            queries.append(
                _sq(
                    f"identity theft impersonation Aadhaar bank KYC report cybercrime.gov.in India {year}",
                    "identity theft",
                    CATEGORY_CRIME,
                    "sit_identity_theft",
                )
            )
        elif name == "online_bullying":
            queries.append(
                _sq(
                    f"cyberbullying online bullying complaint cybercrime.gov.in India {year}",
                    "online bullying",
                    CATEGORY_CRIME,
                    "sit_online_bullying",
                )
            )
        elif name == "matrimonial_fraud":
            queries.append(
                _sq(
                    f"fraudulent marriage NRI groom matrimonial scam police complaint India {year}",
                    "matrimonial fraud",
                    CATEGORY_CRIME,
                    "sit_matrimonial_fraud",
                )
            )
        elif name == "intimate_content_sharing":
            queries.append(
                _sq(
                    f"non-consensual intimate images videos IT Act 67A cybercrime.gov.in India {year}",
                    "intimate image abuse",
                    CATEGORY_CRIME,
                    "sit_intimate_content",
                )
            )
        elif name == "financial_scam":
            queries.append(
                _sq(
                    f"financial scam UPI bank fraud report cybercrime.gov.in India {year}",
                    "financial scam",
                    CATEGORY_CRIME,
                    "sit_financial_scam",
                )
            )
        elif name == "social_exclusion":
            queries.append(
                _sq(
                    f"social exclusion ostracism community boycott support helpline India {year}",
                    "social exclusion",
                    CATEGORY_MENTAL_HEALTH,
                    "sit_social_exclusion",
                )
            )
        elif name == "acid_attack":
            queries.append(
                _sq(
                    f"acid attack FIR medical treatment compensation India official {year}",
                    "acid attack",
                    CATEGORY_CRIME,
                    "sit_acid_attack",
                )
            )
        elif name == "trafficking":
            queries.append(
                _sq(
                    f"human trafficking women children Immoral Traffic Prevention Act India {year}",
                    "trafficking",
                    CATEGORY_CRIME,
                    "sit_trafficking",
                )
            )
        elif name == "cyberstalking":
            queries.append(
                _sq(
                    f"cyberstalking IPC 354D online stalking report cybercrime.gov.in India {year}",
                    "cyberstalking",
                    CATEGORY_CRIME,
                    "sit_cyberstalking",
                )
            )
        elif name == "exposing_personal_information":
            queries.append(
                _sq(
                    f"personal data leaked online doxxing complaint IT Act India {year}",
                    "exposing personal information",
                    CATEGORY_CRIME,
                    "sit_exposing_personal_info",
                )
            )
        elif name == "sexual_assault":
            queries.append(
                _sq(
                    f"sexual assault molestation FIR IPC 354 medical legal aid India {year}",
                    "sexual assault",
                    CATEGORY_CRIME,
                    "sit_sexual_assault",
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
                f"CrPC Section 154 information police station FIR procedure India indiacode.nic.in indiankanoon.org {CURRENT_YEAR}",
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
    ranker=None,
    situation_threshold: float = SEMANTIC_SITUATION_THRESHOLD,
    situations: Optional[Sequence[str]] = None,
) -> List[SearchQuery]:
    """Build search queries from the dialogue situation, then entity templates."""
    entities = list(entities or [])
    if situations is None:
        situations, _meta = resolve_dialogue_situations(
            victim_text,
            ranker=ranker,
            threshold=situation_threshold,
            log=False,
        )
    else:
        situations = list(situations)
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
    built.extend(_situation_queries(victim_text, situations=situations))
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
