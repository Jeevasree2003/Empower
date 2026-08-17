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

logger = logging.getLogger(__name__)

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
    if text and text[-1] not in ".!?":
        text += "."
    return text


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
    sentences: List[str] = []
    system_prompt = (
        "Convert knowledge triplets into one fluent English sentence each. "
        "Write natural sentences, not concatenations. "
        "Match the style of these examples:\n\n"
        f"{_LLM_FEW_SHOT_EXAMPLES}"
    )
    for triplet in triplet_list:
        user_prompt = (
            f"Head: {triplet.head}\n"
            f"Relation: {triplet.relation}\n"
            f"Tail: {triplet.tail}\n"
            "Sentence:"
        )
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=80,
            )
            sentences.append(_clean(response.choices[0].message.content or ""))
        except Exception as exc:
            if fallback_to_template:
                logger.warning(
                    "LLM verbalization failed for triplet (%r, %r, %r): %s; using template",
                    triplet.head,
                    triplet.relation,
                    triplet.tail,
                    exc,
                )
                sentences.append(verbalize_template(triplet))
            else:
                raise
    return sentences


def verbalize_triplets(triplets: Iterable[Triplet], backend: str = "llm", **kwargs) -> List[str]:
    triplet_list = list(triplets)
    if backend == "template":
        return [verbalize_template(t) for t in triplet_list]
    if backend == "llm":
        return verbalize_llm(
            triplet_list,
            model=kwargs.get("model"),
            llm_config=kwargs.get("llm_config"),
            fallback_to_template=kwargs.get("fallback_to_template", True),
        )
    raise ValueError(f"Unsupported verbalization backend: {backend}")
