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
_FAMILY_LAW = frozenset({"bigamy", "desertion", "dowry", "domestic violence"})
_MISSING = frozenset({"missing person"})
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
            "If you feel unsafe or at immediate risk of harm, contact emergency services 112 in India.",
            _CRISIS | _VIOLENCE | _ALWAYS,
            "https://www.mha.gov.in/",
        ),
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
            "You do not have to face this alone; speaking with a trusted person or counselor can reduce isolation during a crisis.",
            _CRISIS | _ALWAYS,
        ),
        CounselingFact(
            DOMAIN_CLINICAL,
            "If someone has threatened violence, move to a safer place if you can and avoid confronting the person who made the threat.",
            _VIOLENCE,
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Murder and attempted murder are cognizable offences; report the threat and any evidence to the local police station immediately.",
            frozenset({"murder", "kill"}),
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
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Marrying again while a previous marriage is still legally valid can be an offence of bigamy under IPC Section 494.",
            _FAMILY_LAW,
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
            "A victim can file an FIR at the nearest police station; information to police is recorded under CrPC Section 154.",
            _VIOLENCE | _PROCEDURE | _MISSING | _ALWAYS,
            "https://www.indiacode.nic.in/",
        ),
        CounselingFact(
            DOMAIN_LEGAL,
            "Women in distress in India can call the National Commission for Women helpline 181 for support and referral.",
            _VIOLENCE | _PROCEDURE | _FAMILY_LAW | _ALWAYS,
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
        "scared",
        "husband",
    ):
        if token in lower:
            keys.add(token)
    if "missing" in lower or "not returned" in lower:
        keys.add("missing person")
    if "another marriage" in lower or "not divorced" in lower or "left me" in lower:
        keys.add("desertion")
        keys.add("bigamy")
    return {k for k in keys if k}


def victim_needs_domains(entities: Sequence[Dict[str, str]], victim_text: str) -> Set[str]:
    """Return {legal, clinical} domains that this victim turn should cover."""
    cats = {e.get("category") for e in entities}
    keys = _trigger_keys(entities, victim_text)
    domains: Set[str] = set()
    if cats & {CATEGORY_MENTAL_HEALTH} or keys & _CRISIS:
        domains.add(DOMAIN_CLINICAL)
    if cats & {CATEGORY_CRIME, CATEGORY_LEGAL} or keys & (
        _VIOLENCE | _PROCEDURE | _FAMILY_LAW | _MISSING | _CYBER
    ):
        domains.add(DOMAIN_LEGAL)
    if victim_text.strip():
        domains.add(DOMAIN_CLINICAL)
        domains.add(DOMAIN_LEGAL)
    return domains


def counseling_candidates(
    entities: Sequence[Dict[str, str]],
    victim_text: str,
    per_domain: int = 2,
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

    for domain in (DOMAIN_CLINICAL, DOMAIN_LEGAL):
        if domain not in needed:
            continue
        taken = 0
        for fact in _facts():
            if fact.domain != domain:
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
