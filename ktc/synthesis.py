"""LLM-3 — Evidence synthesis over ranked knowledge candidates.

Turns a ranked list of KnowledgeCandidate items into one grounded passage
for the response-generation model. Defaults to an OpenAI-compatible chat
backend (local Ollama included) with concatenation fallback when the key
is missing, the model drifts, or the output fails a numeric grounding check.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from ktc.knowledge_item import KnowledgeCandidate
from ktc.verbalization import _make_llm_client

logger = logging.getLogger(__name__)

DEFAULT_SYNTHESIS_MODEL = "qwen2.5:3b-instruct"


@dataclass
class SynthesisResult:
    """Passage plus whether it came from a successful LLM call (not concat fallback)."""

    text: str
    used_llm: bool = False


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SECTION_RE = re.compile(
    r"\b(?:(?:ipc|crpc)\s*(?:section\s*)?|section\s+)(?P<code>\d+[a-z]?)\b",
    re.IGNORECASE,
)
_NAMED_ANCHORS_RE = re.compile(r"\b(?:kiran|icall|nalsa|tiss)\b", re.IGNORECASE)
_HYPHEN_TRANSLATE = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
)
_CONTENT_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "with",
        "from",
        "by",
        "as",
        "if",
        "you",
        "your",
        "can",
        "may",
        "must",
        "not",
        "do",
        "does",
        "will",
        "would",
        "should",
        "have",
        "has",
        "had",
        "it",
        "its",
    }
)
_CONTENT_COVER_RATIO = 0.55
_META_PREFIXES = (
    "note:",
    "here is",
    "here's",
    "sure,",
    "as an ai",
    "i cannot",
    "the passage",
    "summary:",
    "synthesized",
)

_SYSTEM_PROMPT = """You merge grounded knowledge candidates into ONE coherent English passage for a counselor's reply model.

You are given a numbered list of candidate facts below. Your job is to rewrite them into a single, well-connected passage — NOT to select a subset. Every fact in the list that is relevant to the victim's situation must be represented in your output in some form. Do not omit a fact just because it seems less important than others; instead, order the passage so the most urgent/relevant information comes first and secondary information follows.

You may merge closely related facts into one sentence for fluency, but do not drop any fact's core content (numbers, section references, helpline names, and specific entitlements must all still appear somewhere in the output).

Hard rules (violations are unacceptable):
1. Do not invent legal section numbers, helpline numbers, URLs, dates, or percentages that are not in the candidates.
2. Keep specific names and numbers exactly as written. If a candidate says "KIRAN helpline 1800-599-0019", you must keep "KIRAN" and "1800-599-0019". Do not genericize to "a mental health helpline".
3. The dialogue is only context for what the victim asked. Do not treat the dialogue as a source of legal, medical, or helpline facts.
4. Do not copy legal citation or docket noise (for example 2026:JHHC:16350-DB). Personal names and contact PII were already stripped upstream; do not add new ones.
5. Output ONLY the synthesized passage. No title, no "Note:", no bullet labels, no markdown, no reasoning.

Worked example:
Candidates:
1. [counseling_bank] KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India. url=https://www.mohfw.gov.in/
2. [static_dataset] A victim can file an FIR at the nearest police station.

Correct output (covers every candidate):
KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India. A victim can file an FIR at the nearest police station.

Incorrect output (invents a number that is not in the candidates):
KIRAN and iCall 9152987821 can help, and you should call 112.

Incorrect output (drops a real candidate fact for brevity):
KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India.
"""


def _concat_candidates(candidates: Sequence[KnowledgeCandidate]) -> str:
    parts: List[str] = []
    seen = set()
    for item in candidates:
        text = re.sub(r"\s+", " ", (item.text or "").strip())
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return " ".join(parts)


def _normalize_grounding_text(text: str) -> str:
    return (text or "").translate(_HYPHEN_TRANSLATE)


def _numbers_in(text: str) -> Set[str]:
    return set(_NUMBER_RE.findall(_normalize_grounding_text(text)))


def novel_numbers(passage: str, source_text: str) -> Set[str]:
    """Digit sequences in ``passage`` that never appear in ``source_text``."""
    return _numbers_in(passage) - _numbers_in(source_text)


def grounding_tokens(text: str) -> Set[str]:
    """Phone digits, IPC/section references, and helpline names used for coverage checks."""
    normalized = _normalize_grounding_text(text)
    found: Set[str] = {number for number in _numbers_in(normalized) if len(number) >= 3}
    for match in _SECTION_RE.finditer(normalized):
        code = (match.group("code") or "").lower()
        if code:
            found.add(code)
    for match in _NAMED_ANCHORS_RE.finditer(normalized):
        found.add(match.group(0).lower())
    return found


def _content_words(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", _normalize_grounding_text(text).lower())
    return [word for word in words if word not in _CONTENT_STOPWORDS and len(word) > 2]


def candidate_is_covered(passage: str, fact: str) -> bool:
    """True when this candidate's distinctive tokens, or tokenless key content, appear in ``passage``."""
    fact = (fact or "").strip()
    if not fact:
        return True
    fact_tokens = grounding_tokens(fact)
    passage_tokens = grounding_tokens(passage)
    if fact_tokens:
        return fact_tokens <= passage_tokens
    words = list(dict.fromkeys(_content_words(fact)))
    if not words:
        return True
    passage_lower = _normalize_grounding_text(passage).lower()
    hits = sum(1 for word in words if word in passage_lower)
    return hits / len(words) >= _CONTENT_COVER_RATIO


def completeness_report(
    passage: str, candidates: Sequence[KnowledgeCandidate]
) -> List[Tuple[int, bool, str]]:
    """Per-candidate coverage: ``(index, covered, text_preview)``."""
    rows: List[Tuple[int, bool, str]] = []
    for index, item in enumerate(candidates):
        fact = re.sub(r"\s+", " ", (item.text or "").strip())
        preview = fact if len(fact) <= 120 else fact[:117] + "..."
        covered = candidate_is_covered(passage, fact)
        rows.append((index, covered, preview))
    return rows


def coverage_gap_facts(passage: str, candidates: Sequence[KnowledgeCandidate]) -> List[str]:
    """Return candidate texts that are not covered by ``passage``."""
    gaps: List[str] = []
    for index, covered, _preview in completeness_report(passage, candidates):
        if covered:
            continue
        gaps.append(re.sub(r"\s+", " ", (candidates[index].text or "").strip()))
    return gaps


def log_completeness_check(passage: str, candidates: Sequence[KnowledgeCandidate]) -> List[int]:
    """Log per-candidate coverage against the LLM response. Returns missing indices."""
    missing: List[int] = []
    for index, covered, preview in completeness_report(passage, candidates):
        logger.info(
            "completeness_check candidate=%s covered=%s text_preview=%r",
            index,
            covered,
            preview,
        )
        if not covered:
            missing.append(index)
            logger.warning("synthesis_coverage_gap missing_fact=%r", preview)
    if missing:
        logger.warning(
            "synthesis completeness check failed; missing_indices=%s; falling back to concatenated evidence",
            missing,
        )
    return missing


def log_coverage_gaps(passage: str, candidates: Sequence[KnowledgeCandidate]) -> List[str]:
    """Warn when a candidate's distinctive content never appears in the synthesized passage."""
    missing = log_completeness_check(passage, candidates)
    return [
        re.sub(r"\s+", " ", (candidates[index].text or "").strip()) for index in missing
    ]


def _approx_tokens(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(len(stripped.split()), (len(stripped) + 3) // 4)


def _format_candidates(candidates: Sequence[KnowledgeCandidate]) -> str:
    lines: List[str] = []
    for i, item in enumerate(candidates, 1):
        text = re.sub(r"\s+", " ", (item.text or "").strip())
        if not text:
            continue
        extra = []
        if item.source:
            extra.append(item.source)
        if item.url:
            extra.append(f"url={item.url}")
        prefix = f"[{', '.join(extra)}] " if extra else ""
        lines.append(f"{i}. {prefix}{text}")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip().strip('"').strip("'").strip()


def _sanitize_synthesis_passage(text: str) -> str:
    """Squeeze whitespace and ensure terminal punctuation; keep the full passage."""
    passage = re.sub(r"\s+", " ", (text or "").strip())
    if passage and passage[-1] not in '.!?"\'':
        passage += "."
    return passage


def _passage_is_malformed(text: str) -> bool:
    passage = (text or "").strip()
    if not passage:
        return True
    if len(passage) < 12:
        return True
    lower = passage.lower()
    if lower.startswith(_META_PREFIXES):
        return True
    if "```" in passage:
        return True
    if passage[-1] not in '.!?"\'' and len(passage) >= 80:
        return True
    return False


_CONTEXT_SYSTEM_PROMPT = """No grounded facts survived retrieval for this turn. Using ONLY the conversation given, write ONE short (2-4 sentence) warm passage that acknowledges the victim's message and gives general, safe, non-specific supportive guidance (e.g. reach out to someone trusted, local police, or a helpline) WITHOUT inventing any specific phone number, legal section/IPC code, organisation name, or URL, since none were verified for this turn. Do not claim to be human or a professional; do not diagnose. Output ONLY the passage — no title, no notes, no markdown."""


def synthesize_from_context(
    dialog_history: str,
    victim_span: str,
    model: str = DEFAULT_SYNTHESIS_MODEL,
    llm_config=None,
) -> SynthesisResult:
    """One LLM call over dialogue context when retrieval produced no grounded facts."""
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        logger.warning("LLM_API_KEY not set; skipping context-only synthesis")
        return SynthesisResult(text="", used_llm=False)

    try:
        from openai import OpenAI  # noqa: F401 — checked by _make_llm_client
    except ImportError:
        logger.warning("openai package not installed; skipping context-only synthesis")
        return SynthesisResult(text="", used_llm=False)

    history = (dialog_history or "").strip() or "(none)"
    latest = (victim_span or "").strip() or "(none)"
    user_prompt = (
        "Dialogue history:\n"
        f"{history}\n\n"
        "Victim's latest message:\n"
        f"{latest}\n\n"
        "Supportive passage:"
    )
    try:
        client, _config = _make_llm_client(llm_config)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = _strip_fences(response.choices[0].message.content or "")
        passage = _sanitize_synthesis_passage(raw)
    except Exception as exc:
        logger.warning("context-only synthesis failed: %s; returning empty knowledge", exc)
        return SynthesisResult(text="", used_llm=False)

    if _passage_is_malformed(passage):
        logger.warning("context-only synthesis output malformed; returning empty knowledge")
        return SynthesisResult(text="", used_llm=False)

    invented = grounding_tokens(passage)
    if invented:
        logger.warning(
            "context-only synthesis invented grounded tokens=%s; returning empty knowledge",
            sorted(invented),
        )
        return SynthesisResult(text="", used_llm=False)
    return SynthesisResult(text=passage, used_llm=True)


def synthesize_evidence(
    candidates: List[KnowledgeCandidate],
    dialog_history: str,
    backend: str = "llm",
    model: str = DEFAULT_SYNTHESIS_MODEL,
    llm_config=None,
) -> SynthesisResult:
    """Return one grounded passage, or concatenated candidate texts on fallback."""
    items = [c for c in candidates if (c.text or "").strip()]
    fallback = _concat_candidates(items)
    if not items:
        return SynthesisResult(text="", used_llm=False)
    if backend == "template":
        return SynthesisResult(text=fallback, used_llm=False)
    if backend != "llm":
        raise ValueError(f"Unsupported synthesis backend: {backend}")

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        logger.warning("LLM_API_KEY not set; falling back to concatenated evidence")
        return SynthesisResult(text=fallback, used_llm=False)

    try:
        from openai import OpenAI  # noqa: F401 — checked by _make_llm_client
    except ImportError:
        logger.warning("openai package not installed; falling back to concatenated evidence")
        return SynthesisResult(text=fallback, used_llm=False)

    source_blob = " ".join(c.text for c in items)
    formatted = _format_candidates(items)
    user_prompt = (
        "Dialogue context (not a fact source):\n"
        f"{(dialog_history or '').strip() or '(none)'}\n\n"
        "Candidates:\n"
        f"{formatted}\n\n"
        "Rewrite every relevant candidate fact into one passage. Do not select a subset.\n\n"
        "Synthesized passage:"
    )
    prompt_chars = len(_SYSTEM_PROMPT) + len(user_prompt)
    logger.info(
        "synthesis_prompt_built candidates=%s prompt_chars=%s",
        len(items),
        prompt_chars,
    )
    try:
        client, _config = _make_llm_client(llm_config)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=2048,
        )
        raw = _strip_fences(response.choices[0].message.content or "")
        passage = _sanitize_synthesis_passage(raw)
    except Exception as exc:
        logger.warning("LLM synthesis failed: %s; falling back to concatenated evidence", exc)
        return SynthesisResult(text=fallback, used_llm=False)

    if _passage_is_malformed(passage):
        logger.warning(
            "synthesis output malformed; finish_reason=%s content_len=%s; falling back to concatenated evidence",
            getattr(response.choices[0], "finish_reason", None),
            len(raw),
        )
        return SynthesisResult(text=fallback, used_llm=False)

    invented = novel_numbers(passage, source_blob)
    if invented:
        logger.warning(
            "synthesis grounding check failed; invented numbers=%s; falling back to concatenated evidence",
            sorted(invented),
        )
        return SynthesisResult(text=fallback, used_llm=False)
    logger.info(
        "synthesis_prompt_built candidates=%s prompt_chars=%s output_chars=%s output_approx_tokens=%s",
        len(items),
        prompt_chars,
        len(passage),
        _approx_tokens(passage),
    )
    missing = log_completeness_check(passage, items)
    if missing:
        return SynthesisResult(text=fallback, used_llm=False)
    return SynthesisResult(text=passage, used_llm=True)
