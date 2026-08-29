"""Stage 2e — Triplet verbalization.

The paper used few-shot GPT-J. This implementation defaults to an LLM backend
(Groq/OpenAI-compatible via ``LLM_API_KEY``) with template fallback when no key
is configured. Pass ``backend='template'`` for fully offline runs.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, List, Optional

from ktc.triplet import Triplet
from ktc.cleaning import normalize_clause_key

logger = logging.getLogger(__name__)

_VERBALIZATION_CACHE: dict = {}

_COPULA_PREFIXES = ("is ", "are ", "was ", "were ", "has been ", "have been ", "had been ")
_MODAL_PREFIXES = ("can ", "may ", "should ", "must ", "will ", "could ", "would ", "might ")
_PERFECT_PREFIXES = ("has ", "have ", "had ")
_PRESENT_PARTICIPLE = re.compile(r"^(?:is|are|was|were|has been|have been|had been) \w+ing\b")

# Few-shot examples for LLM verbalization (paper-style single-sentence output).
_LLM_FEW_SHOT_EXAMPLES = """
Example 1:
Head: Cyber Cells
Relation: are present in
Tail: every state
Sentence: Cyber Cells are present in every state.

Example 2:
Head: victim
Relation: can file
Tail: an online complaint
Sentence: A victim can file an online complaint.

Example 3:
Head: the police
Relation: was filed by
Tail: the complaint
Sentence: The complaint was filed by the police.
""".strip()


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _sanitize_llm_sentence(text: str) -> str:
    """Keep only the first fluent sentence; drop model meta-commentary."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    for marker in (" Note:", " However,", " Note that", "\nNote:", "\nHowever,"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
    return _clean(text)


def _is_passive_relation(relation_lower: str) -> bool:
    if relation_lower.endswith(" by"):
        return True
    return bool(
        re.match(r"^(?:is|are|was|were|has been|have been|had been) \w+ed\b", relation_lower)
        or re.match(r"^(?:is|are|was|were) \w+en\b", relation_lower)
    )


def verbalize_template(triplet: Triplet) -> str:
    """Convert a triplet into a natural-language sentence without naive concatenation."""
    head = triplet.head.strip()
    relation = triplet.relation.strip()
    tail = triplet.tail.strip()

    relation_lower = relation.lower()

    if _is_passive_relation(relation_lower) or relation_lower.endswith(" by"):
        sentence = f"{tail} {relation} {head}"
    elif relation_lower.startswith(_COPULA_PREFIXES):
        sentence = f"{head} {relation} {tail}"
    elif relation_lower.startswith(_MODAL_PREFIXES):
        sentence = f"{head} {relation} {tail}"
    elif relation_lower.startswith(_PERFECT_PREFIXES):
        sentence = f"{head} {relation} {tail}"
    elif _PRESENT_PARTICIPLE.match(relation_lower):
        sentence = f"{head} {relation} {tail}"
    else:
        sentence = f"{head} {relation} {tail}"

    return _clean(sentence)


def _make_llm_client(llm_config=None):
    """Build an OpenAI-compatible client (OpenAI, Groq, etc.)."""
    from openai import OpenAI

    from ktc.live_config import LiveRetrievalConfig

    config = llm_config or LiveRetrievalConfig.load()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_API_BASE", "").strip() or (config.llm_api_base or "").strip()
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), config


_BATCH_LINE_PREFIX = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*")


def _parse_batched_verbalization(
    raw: str,
    triplet_list: List[Triplet],
    fallback_to_template: bool,
) -> List[str]:
    text = (raw or "").strip()
    if len(triplet_list) == 1 and text and not _BATCH_LINE_PREFIX.match(text.splitlines()[0]):
        return [_sanitize_llm_sentence(text)]
    parsed: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = _BATCH_LINE_PREFIX.sub("", stripped, count=1)
        if stripped:
            parsed.append(_sanitize_llm_sentence(stripped))
    if len(parsed) != len(triplet_list):
        logger.warning(
            "LLM verbalization line count mismatch: got %s expected %s; using templates",
            len(parsed),
            len(triplet_list),
        )
        if fallback_to_template:
            return [verbalize_template(triplet) for triplet in triplet_list]
        raise ValueError(
            f"LLM verbalization returned {len(parsed)} lines for {len(triplet_list)} triplets"
        )
    return parsed


def verbalize_llm(
    triplets: Iterable[Triplet],
    model: Optional[str] = None,
    llm_config=None,
    fallback_to_template: bool = True,
) -> List[str]:
    """LLM verbalization via an OpenAI-compatible chat API (Groq by default)."""
    triplet_list = list(triplets)
    if not triplet_list:
        return []

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        if fallback_to_template:
            logger.warning("LLM_API_KEY not set; falling back to template verbalization")
            return [verbalize_template(t) for t in triplet_list]
        raise RuntimeError("LLM_API_KEY is required for verbalization_backend='llm'")

    try:
        from openai import OpenAI  # noqa: F401 — checked by _make_llm_client
    except ImportError as exc:
        if fallback_to_template:
            logger.warning("openai package not installed; falling back to template verbalization")
            return [verbalize_template(t) for t in triplet_list]
        raise ImportError("Install openai to use LLM verbalization: pip install openai") from exc

    client, config = _make_llm_client(llm_config)
    resolved_model = model or config.llm_model
    system_prompt = (
        "Convert each knowledge triplet into exactly one fluent English sentence. "
        "Reply with exactly one numbered line per triplet, in the same order as listed. "
        "Output ONLY those numbered sentences. No notes, no explanations, no alternative "
        "phrasing, no preamble, and no markdown. "
        "Match the style of these examples:\n\n"
        f"{_LLM_FEW_SHOT_EXAMPLES}"
    )
    listed = "\n".join(
        f"{index}. Head: {triplet.head} | Relation: {triplet.relation} | Tail: {triplet.tail}"
        for index, triplet in enumerate(triplet_list, 1)
    )
    user_prompt = (
        "Triplets:\n"
        f"{listed}\n\n"
        "Sentences:"
    )
    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max(200 * len(triplet_list), 250),
        )
        raw = response.choices[0].message.content or ""
        return _parse_batched_verbalization(raw, triplet_list, fallback_to_template)
    except Exception as exc:
        if fallback_to_template:
            logger.warning(
                "LLM verbalization failed for batch of %s triplets: %s; using template",
                len(triplet_list),
                exc,
            )
            return [verbalize_template(triplet) for triplet in triplet_list]
        raise


def _triplet_cache_key(triplet: Triplet, backend: str, model: Optional[str]) -> tuple:
    return (
        backend,
        (model or "").strip(),
        normalize_clause_key(triplet.head),
        normalize_clause_key(triplet.relation),
        normalize_clause_key(triplet.tail),
    )


def clear_verbalization_cache() -> None:
    """Reset the process-level verbalization cache (for tests)."""
    _VERBALIZATION_CACHE.clear()


def verbalize_triplets(triplets: Iterable[Triplet], backend: str = "llm", **kwargs) -> List[str]:
    triplet_list = list(triplets)
    if not triplet_list:
        return []
    model = kwargs.get("model")
    results: List[Optional[str]] = [None] * len(triplet_list)
    missing: List[int] = []
    for index, triplet in enumerate(triplet_list):
        key = _triplet_cache_key(triplet, backend, model)
        cached = _VERBALIZATION_CACHE.get(key)
        if cached is not None:
            results[index] = cached
        else:
            missing.append(index)
    if missing:
        pending = [triplet_list[i] for i in missing]
        if backend == "template":
            produced = [verbalize_template(item) for item in pending]
        elif backend == "llm":
            produced = verbalize_llm(
                pending,
                model=model,
                llm_config=kwargs.get("llm_config"),
                fallback_to_template=kwargs.get("fallback_to_template", True),
            )
        else:
            raise ValueError(f"Unsupported verbalization backend: {backend}")
        for index, sentence in zip(missing, produced):
            results[index] = sentence
            _VERBALIZATION_CACHE[_triplet_cache_key(triplet_list[index], backend, model)] = sentence
    return [sentence or "" for sentence in results]
