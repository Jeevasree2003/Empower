"""Grounded legal + clinical counseling sentences for live victim turns.

KARE knowledge blobs are often off-topic (yoga vs abandonment, romance-scam
pages vs a homicide threat). OpenIE on that text cannot invent the missing
domain. These items are short, India-facing counselor facts used when a victim
has spoken, so verbalized knowledge covers both legal help-seeking and
clinical safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

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


_ALWAYS = frozenset({"*"})
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
    }
)
_PROCEDURE = frozenset(
    {"fir", "complaint", "helpline", "police", "ncw", "legal aid", "protection order"}
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


def _facts() -> List[CounselingFact]:
    return [
        CounselingFact(
            DOMAIN_CLINICAL,
            "KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India.",
            _CRISIS | _ALWAYS,
            "https://www.mohfw.gov.in/",
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "iCall psychosocial helpline 9152987821 provides confidential counseling for people in emotional distress.",
            _CRISIS | _ALWAYS,
            "https://icallhelpline.org/",
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "You can ask for help even if you are not sure what the problem is called; a counselor can listen first.",
            _CRISIS | _ALWAYS,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "If you do not know which office to visit, a helpline can tell you the next safe step without requiring a diagnosis.",
            _CRISIS | _ALWAYS,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "If you feel unsafe or at immediate risk of harm, contact emergency services 112 in India.",
            frozenset(
                {
                    "dying",
                    "suicide",
                    "suicidal",
                    "self harm",
                    "self-harm",
                    "murder",
                    "kill",
                    "rape",
                    "assault",
                    "threat to life",
                }
            ),
            "https://www.mha.gov.in/",
            emergency=True,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "You do not have to face this alone; speaking with a trusted person or counselor can reduce isolation during a crisis.",
            _CRISIS | _ALWAYS,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "If someone has threatened violence, move to a safer place if you can and avoid confronting the person who made the threat.",
            frozenset({"murder", "kill", "rape", "assault", "threat", "threaten", "threat to life"}),
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
            _VIOLENCE,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Rape is a cognizable offence under IPC Section 376; a survivor can file an FIR and seek medical and legal aid without delay.",
            frozenset({"rape", "gang rape", "gang-rape"}),
            emergency=True,
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
            _WORKPLACE | frozenset({"harassment"}),
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Loan recovery agents cannot threaten, shame, or call at odd hours; abusive recovery can be reported to the lender and to police if there is intimidation.",
            _DEBT,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "You do not have to file a police case before getting emotional support; 181 or NALSA legal aid can help if you later need legal information or protection.",
            _CRISIS | _ALWAYS,
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
            _VIOLENCE | _PROCEDURE | _FAMILY_LAW | _CRISIS,
            "https://www.ncw.nic.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Cybercrime including online harassment can be reported at the National Cyber Crime Reporting Portal cybercrime.gov.in.",
            _CYBER | frozenset({"complaint"}),
            "https://cybercrime.gov.in/",
        ),
    ]


def _trigger_keys(entities: Sequence[Dict[str, str]], victim_text: str) -> Set[str]:
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
        "workplace",
    ):
        if token in lower:
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
    if "invested" in lower or "not refund" in lower:
        keys.add("scam")
    return {k for k in keys if k}


def content_need_domains(entities: Sequence[Dict[str, str]], victim_text: str) -> Set[str]:
    """Domains evidenced in the victim text — used to search the KARE blob."""
    cats = {e.get("category") for e in entities}
    keys = _trigger_keys(entities, victim_text)
    domains: Set[str] = set()
    if cats & {CATEGORY_MENTAL_HEALTH} or keys & _CRISIS:
        domains.add(DOMAIN_CLINICAL)
    if cats & {CATEGORY_CRIME, CATEGORY_LEGAL} or keys & (
        _VIOLENCE | _PROCEDURE | _FAMILY_LAW | _MISSING | _CYBER | _WORKPLACE | _DEBT
    ):
        domains.add(DOMAIN_LEGAL)
    return domains


def victim_needs_domains(entities: Sequence[Dict[str, str]], victim_text: str) -> Set[str]:
    """Domains the counselor brief should cover once the victim has spoken."""
    domains = content_need_domains(entities, victim_text)
    if victim_text.strip():
        domains.add(DOMAIN_CLINICAL)
        domains.add(DOMAIN_LEGAL)
    return domains


def counseling_candidates(
    entities: Sequence[Dict[str, str]],
    victim_text: str,
    per_domain: int = 3,
) -> List[KnowledgeCandidate]:
    if not (victim_text or "").strip():
        return []
    keys = _trigger_keys(entities, victim_text)
    needed = victim_needs_domains(entities, victim_text)
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
                domain=fact.domain,
            )
        )
        return True

    for fact in _facts():
        if fact.domain not in needed:
            continue
        specific = fact.triggers - _ALWAYS
        if fact.emergency and specific and keys & specific:
            try_add(fact)

    for domain in (DOMAIN_CLINICAL, DOMAIN_LEGAL):
        if domain not in needed:
            continue
        taken = sum(1 for item in selected if item.domain == domain)
        for fact in _facts():
            if fact.domain != domain or fact.emergency:
                continue
            specific = fact.triggers - _ALWAYS
            if specific and keys & specific:
                if try_add(fact):
                    taken += 1
                if taken >= per_domain:
                    break
        if taken < per_domain:
            for fact in _facts():
                if fact.domain != domain:
                    continue
                if "*" not in fact.triggers:
                    continue
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
    """Guarantee legal+clinical bank facts, then fill with gated ranked items."""
    merged: List[KnowledgeCandidate] = []
    seen: Set[str] = set()

    def add(candidate: KnowledgeCandidate) -> None:
        key = candidate.text.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        merged.append(candidate)

    for item in bank:
        add(item)
        if len(merged) >= top_k:
            return merged[:top_k]
    for item in ranked:
        add(item)
        if len(merged) >= top_k:
            break
    return merged[:top_k]
