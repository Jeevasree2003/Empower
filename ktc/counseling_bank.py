"""Trigger-matched legal + clinical facts, kept out of Stage 2e verbalization.

KARE blobs are often off-topic, so OpenIE cannot invent missing helpline or
statute text. These sentences are optional supplemental counseling — they fire
only when a trigger token appears in the victim utterance, never as a global
always-on dump mixed into verbalized KTC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ktc.entity_extraction import (
    CATEGORY_CRIME,
    CATEGORY_LEGAL,
    CATEGORY_MENTAL_HEALTH,
)
from ktc.knowledge_item import KnowledgeCandidate

DOMAIN_LEGAL = "legal"
DOMAIN_CLINICAL = "clinical"


@dataclass(frozen=True)
class CounselingFact:
    domain: str
    text: str
    triggers: frozenset
    url: str = ""
    emergency: bool = False


_CRISIS = frozenset(
    {
        "dying",
        "suicide",
        "suicidal",
        "self harm",
        "self-harm",
        "kill myself",
        "scared",
        "trauma",
        "depression",
        "anxiety",
        "stress",
        "insane",
        "mental health",
    }
)
_VIOLENCE = frozenset(
    {
        "murder",
        "kill",
        "rape",
        "assault",
        "domestic violence",
        "threat",
        "threaten",
        "threat to life",
        "harassment",
        "abuse",
        "stalking",
        "torture",
    }
)
_PROCEDURE = frozenset(
    {"fir", "complaint", "police", "ncw", "legal aid", "protection order"}
)
_FAMILY_LAW = frozenset({"bigamy", "desertion", "dowry", "domestic violence", "kicked"})
_MISSING = frozenset({"missing person"})
_WORKPLACE = frozenset({"workplace", "posh", "terminate", "employer"})
_DEBT = frozenset({"loan", "recovery", "invested", "refund", "scam", "fraud"})
_CYBER = frozenset(
    {
        "cyberstalking",
        "cyberbullying",
        "phishing",
        "scam",
        "fraud",
        "identity theft",
        "revenge porn",
    }
)
_GENERAL_SUPPORT = frozenset({"general_support"})
_SITUATION_LEGAL = frozenset(
    {
        "child_exploitation",
        "online_harassment",
        "identity_theft",
        "online_bullying",
        "matrimonial_fraud",
        "intimate_content_sharing",
        "financial_scam",
        "acid_attack",
        "trafficking",
        "cyberstalking",
        "exposing_personal_information",
        "sexual_assault",
        "homicide_threat",
        "rape",
        "gang_rape",
        "torture",
        "kicked_out",
        "workplace_harassment",
        "loan_recovery",
        "investment_fraud",
        "missing_person",
        "desertion_bigamy",
        "domestic_violence",
    }
)
_SITUATION_CLINICAL = frozenset({"social_exclusion", "suicide_crisis", "help_seeking"})
_GENERAL_HELP_PATTERNS = (
    r"\bhelpline\b",
    r"\bsupport\b",
    r"\bhelp me\b",
    r"\bwho can i talk to\b",
    r"\bwho can i contact\b",
    r"\burgent help\b",
    r"\bneed some help\b",
    r"\bi need help\b",
    r"\bneed help\b",
)


def _facts() -> List[CounselingFact]:
    return [
        CounselingFact(
            DOMAIN_CLINICAL,
            "KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India.",
            _CRISIS | _GENERAL_SUPPORT,
            "https://www.mohfw.gov.in/",
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "iCall psychosocial helpline 9152987821 provides confidential counseling for people in emotional distress.",
            _CRISIS,
            "https://icallhelpline.org/",
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "You can ask for help even if you are not sure what the problem is called; a counselor can listen first.",
            _CRISIS,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "If you do not know which office to visit, a helpline can tell you the next safe step without requiring a diagnosis.",
            _CRISIS,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "If you feel unsafe or at immediate risk of harm, contact emergency services 112 in India.",
            frozenset({"dying", "suicide", "suicidal", "self harm", "self-harm", "murder", "kill", "assault", "threat to life"}),
            "https://www.mha.gov.in/",
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "You do not have to face this alone; speaking with a trusted person or counselor can reduce isolation during a crisis.",
            _CRISIS,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "If someone has threatened violence, move to a safer place if you can and avoid confronting the person who made the threat.",
            frozenset({"murder", "kill", "threat", "threaten", "threat to life"}),
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Murder and attempted murder are cognizable offences; report the threat and any evidence to the local police station immediately.",
            frozenset({"murder", "kill"}),
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "A threat to kill can be reported as criminal intimidation; police protection can be requested without delay.",
            frozenset({"murder", "kill", "threat", "threaten", "threat to life"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Rape is a cognizable offence under IPC Section 376; a survivor can file an FIR and seek medical and legal aid without delay.",
            frozenset({"rape", "gang rape", "gang-rape"}),
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Gang rape is an aggravated offence under IPC Section 376D; a delayed FIR is still valid and a survivor can request a medical examination and police protection.",
            frozenset({"gang rape", "gang-rape", "rape"}),
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "After sexual assault, a survivor can go to the nearest hospital for medical care and forensic examination; treatment should not be refused for want of a police report.",
            frozenset({"rape", "gang rape", "gang-rape"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Marrying again while a previous marriage is still legally valid can be an offence of bigamy under IPC Section 494.",
            frozenset({"bigamy"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Cruelty by a husband or his relatives can be reported under IPC Section 498A and the Protection of Women from Domestic Violence Act.",
            _FAMILY_LAW | frozenset({"domestic violence", "abuse"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "A missing-person complaint can be given at the local police station; do not wait to file information if someone has not returned.",
            _MISSING,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "A woman who is thrown out of the shared household can seek protection and residence orders under the Protection of Women from Domestic Violence Act, and can ask about maintenance.",
            _FAMILY_LAW | frozenset({"kicked", "husband"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Workplace sexual harassment can be reported to the Internal Complaints Committee under the POSH Act; keep copies of messages, calls, and any termination mail.",
            _WORKPLACE,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Loan recovery agents cannot threaten, shame, or call at odd hours; abusive recovery can be reported to the lender and to police if there is intimidation.",
            _DEBT,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "You do not have to file a police case before getting emotional support; 181 or NALSA legal aid can help if you later need legal information or protection.",
            _CRISIS,
            "https://nalsa.gov.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "A victim can file an FIR at the nearest police station; information to police is recorded under CrPC Section 154.",
            _VIOLENCE | _PROCEDURE | _MISSING,
            "https://www.indiacode.nic.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Women in distress in India can call the National Commission for Women helpline 181 for support and referral.",
            _VIOLENCE | _PROCEDURE | _FAMILY_LAW | _CRISIS | _GENERAL_SUPPORT,
            "https://www.ncw.nic.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Cybercrime including online harassment can be reported at the National Cyber Crime Reporting Portal cybercrime.gov.in.",
            _CYBER | frozenset({"complaint", "online_harassment", "cyberstalking", "identity_theft", "financial_scam", "intimate_content_sharing", "online_bullying"}),
            "https://cybercrime.gov.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Sexual offences against children are covered by the POCSO Act; report immediately to police and the Child Welfare Committee.",
            frozenset({"child_exploitation", "trafficking"}),
            "https://wcd.nic.in/",
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Childline 1098 is the 24x7 emergency helpline for children in distress in India and can connect you to local protection services.",
            frozenset({"child_exploitation", "trafficking"}),
            "https://www.childlineindia.org/",
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Publishing or transmitting obscene or sexually explicit material electronically can be an offence under IT Act Sections 67 and 67A; preserve screenshots and report at cybercrime.gov.in.",
            frozenset({"online_harassment", "intimate_content_sharing"}),
            "https://cybercrime.gov.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Identity theft and impersonation using your name or documents can be reported to local police and the National Cyber Crime Reporting Portal.",
            frozenset({"identity_theft"}),
            "https://cybercrime.gov.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "A fraudulent marriage or matrimonial cheating complaint can be filed with police; keep chats, payments, and the marriage advertisement as evidence.",
            frozenset({"matrimonial_fraud"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Non-consensual sharing of intimate images or videos can be reported on cybercrime.gov.in and to the local cyber cell; ask platforms to take the content down.",
            frozenset({"intimate_content_sharing"}),
            "https://cybercrime.gov.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Financial and UPI scams can be reported on cybercrime.gov.in and to the bank's fraud helpline so the transaction can be flagged quickly.",
            frozenset({"financial_scam", "investment_fraud"}),
            "https://cybercrime.gov.in/",
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "Social exclusion and ostracism can be isolating; KIRAN 1800-599-0019 and iCall can listen and help you plan a next safe step.",
            frozenset({"social_exclusion"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "An acid attack is a grave offence; seek emergency medical care, preserve evidence, and file an FIR without delay.",
            frozenset({"acid_attack"}),
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Human trafficking of women or children can be reported to police, Childline 1098, and anti-trafficking units; do not confront traffickers alone.",
            frozenset({"trafficking", "child_exploitation"}),
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Cyberstalking is an offence under IPC Section 354D; block the accounts, save messages, and report at cybercrime.gov.in.",
            frozenset({"cyberstalking", "online_harassment"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Publishing someone's private phone number, address, or photos online without consent can be reported as a cyber offence and to the platform.",
            frozenset({"exposing_personal_information"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Sexual assault and molestation can be reported as a cognizable offence; a survivor can file an FIR and seek medical and legal aid.",
            frozenset({"sexual_assault", "rape"}),
        ),
    ]


def _trigger_keys(
    entities: Sequence[Dict[str, str]],
    victim_text: str,
    situations: Optional[Sequence[str]] = None,
) -> Set[str]:
    keys = {e.get("text", "").strip().lower() for e in entities if e.get("text")}
    lower = (victim_text or "").lower()
    for token in (
        "dying",
        "suicide",
        "murder",
        "kill",
        "rape",
        "fir",
        "complaint",
        "missing",
        "insane",
        "scared",
        "husband",
        "kicked",
        "harassment",
        "loan",
        "torture",
        "frustrated",
    ):
        if re.search(r"\b" + re.escape(token) + r"\b", lower):
            keys.add(token)
    if "missing" in lower or "not returned" in lower:
        keys.add("missing person")
    if "another marriage" in lower or "not divorced" in lower or "left me" in lower:
        keys.add("desertion")
        keys.add("bigamy")
    if "kicked" in lower or "out of the house" in lower:
        keys.add("kicked")
        keys.add("desertion")
    if "office" in lower or "terminate" in lower or "employer" in lower:
        keys.add("workplace")
    if "loan" in lower or "recovery agent" in lower:
        keys.add("loan")
    if re.search(r"\btortur", lower):
        keys.add("torture")
        keys.add("abuse")
    if re.search(r"raped by\s+\d|gang\s+rape", lower):
        keys.add("gang rape")
        keys.add("rape")
    if any(re.search(pattern, lower) for pattern in _GENERAL_HELP_PATTERNS):
        keys.add("general_support")
    for name in situations or ():
        token = (name or "").strip().lower()
        if token:
            keys.add(token)
    return {k for k in keys if k}


def content_need_domains(
    entities: Sequence[Dict[str, str]],
    victim_text: str,
    situations: Optional[Sequence[str]] = None,
) -> Set[str]:
    """Domains evidenced in the victim text — used to search the KARE blob."""
    cats = {e.get("category") for e in entities}
    keys = _trigger_keys(entities, victim_text, situations=situations)
    domains: Set[str] = set()
    if cats & {CATEGORY_MENTAL_HEALTH} or keys & _CRISIS or keys & _SITUATION_CLINICAL:
        domains.add(DOMAIN_CLINICAL)
    if cats & {CATEGORY_CRIME, CATEGORY_LEGAL} or keys & (
        _VIOLENCE | _PROCEDURE | _FAMILY_LAW | _MISSING | _CYBER | _WORKPLACE | _DEBT | _GENERAL_SUPPORT | _SITUATION_LEGAL
    ):
        domains.add(DOMAIN_LEGAL)
    if keys & _GENERAL_SUPPORT:
        domains.add(DOMAIN_CLINICAL)
        domains.add(DOMAIN_LEGAL)
    return domains


def victim_needs_domains(
    entities: Sequence[Dict[str, str]],
    victim_text: str,
    situations: Optional[Sequence[str]] = None,
) -> Set[str]:
    """Only domains evidenced in the victim text — no blanket clinical/legal fill."""
    return content_need_domains(entities, victim_text, situations=situations)


def counseling_candidates(
    entities: Sequence[Dict[str, str]],
    victim_text: str,
    per_domain: int = 3,
    situations: Optional[Sequence[str]] = None,
) -> List[KnowledgeCandidate]:
    if not (victim_text or "").strip():
        return []
    keys = _trigger_keys(entities, victim_text, situations=situations)
    needed = victim_needs_domains(entities, victim_text, situations=situations)
    selected: List[KnowledgeCandidate] = []
    seen: Set[str] = set()

    def try_add(fact: CounselingFact) -> bool:
        norm = fact.text.lower()
        if norm in seen:
            return False
        seen.add(norm)
        selected.append(
            KnowledgeCandidate(
                text=fact.text,
                source="counseling_bank",
                url=fact.url or None,
                query="counseling_bank",
                domain=fact.domain,
            )
        )
        return True

    for fact in _facts():
        if fact.domain not in needed:
            continue
        specific = fact.triggers
        if fact.emergency and specific and keys & specific:
            try_add(fact)

    for domain in (DOMAIN_CLINICAL, DOMAIN_LEGAL):
        if domain not in needed:
            continue
        taken = sum(1 for item in selected if item.domain == domain)
        for fact in _facts():
            if fact.domain != domain or fact.emergency:
                continue
            if fact.triggers and keys & fact.triggers:
                if try_add(fact):
                    taken += 1
                if taken >= per_domain:
                    break
    return selected


def merge_turn_knowledge(
    ranked: Iterable[KnowledgeCandidate],
    bank: Sequence[KnowledgeCandidate],
    top_k: int = 8,
) -> List[KnowledgeCandidate]:
    """Keep ranked KTC items first; bank is optional and must not replace them."""
    merged: List[KnowledgeCandidate] = []
    seen: Set[str] = set()

    def add(candidate: KnowledgeCandidate) -> None:
        key = candidate.text.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        merged.append(candidate)

    for item in ranked:
        add(item)
        if len(merged) >= top_k:
            return merged[:top_k]
    for item in bank:
        add(item)
        if len(merged) >= top_k:
            break
    return merged[:top_k]
