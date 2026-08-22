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
from typing import List, Sequence, Set

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
1. Use ONLY facts present in the candidate texts. Never add outside knowledge. Never infer facts that are not explicitly stated.
2. Do not invent legal section numbers, helpline numbers, URLs, dates, or percentages that are not in the candidates.
3. Keep specific names and numbers exactly as written. If a candidate says "KIRAN helpline 1800-599-0019", you must keep "KIRAN" and "1800-599-0019". Do not genericize to "a mental health helpline".
4. The dialogue is only context for what the victim asked. Do not treat the dialogue as a source of legal, medical, or helpline facts.
5. Do not copy legal citation or docket noise (for example 2026:JHHC:16350-DB). Personal names and contact PII were already stripped upstream; do not add new ones.
6. Output ONLY the synthesized passage. No title, no "Note:", no bullet labels, no markdown, no reasoning.

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


def _numbers_in(text: str) -> Set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def novel_numbers(passage: str, source_text: str) -> Set[str]:
    """Digit sequences in ``passage`` that never appear in ``source_text``."""
    return _numbers_in(passage) - _numbers_in(source_text)


def grounding_tokens(text: str) -> Set[str]:
    """Phone digits, IPC/section references, and helpline names used for coverage checks."""
    found: Set[str] = set(_numbers_in(text))
    for match in _SECTION_RE.finditer(text or ""):
        code = (match.group("code") or "").lower()
        if code:
            found.add(code)
    for match in _NAMED_ANCHORS_RE.finditer(text or ""):
        found.add(match.group(0).lower())
    return found


def coverage_gap_facts(passage: str, candidates: Sequence[KnowledgeCandidate]) -> List[str]:
    """Return candidate texts whose grounding tokens are entirely absent from ``passage``."""
    passage_tokens = grounding_tokens(passage)
    gaps: List[str] = []
    for item in candidates:
        fact = re.sub(r"\s+", " ", (item.text or "").strip())
        tokens = grounding_tokens(fact)
        if tokens and tokens.isdisjoint(passage_tokens):
            gaps.append(fact)
    return gaps


def log_coverage_gaps(passage: str, candidates: Sequence[KnowledgeCandidate]) -> List[str]:
    """Warn when a candidate's distinctive tokens never appear in the synthesized passage."""
    gaps = coverage_gap_facts(passage, candidates)
    for fact in gaps:
        logger.warning("synthesis_coverage_gap missing_fact=%r", fact)
    return gaps


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
            max_tokens=1024,
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
    log_coverage_gaps(passage, items)
    return SynthesisResult(text=passage, used_llm=True)
