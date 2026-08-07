"""Stage 2e — Triplet verbalization.

The paper used few-shot GPT-J. This implementation uses template-based verbalization
by default so the pipeline runs locally without an API key. Set ``backend='llm'`` and
provide an OpenAI-compatible endpoint to use an instruction-tuned model instead.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List

from ktc.triplet import Triplet

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


def verbalize_llm(triplets: Iterable[Triplet], model: str = "gpt-4o-mini") -> List[str]:
    """Optional LLM verbalization via an OpenAI-compatible chat API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai to use LLM verbalization: pip install openai") from exc

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sentences: List[str] = []
    system_prompt = (
        "Convert knowledge triplets into one fluent English sentence each. "
        "Write natural sentences, not concatenations. "
        "Match the style of these examples:\n\n"
        f"{_LLM_FEW_SHOT_EXAMPLES}"
    )
    for triplet in triplets:
        user_prompt = (
            f"Head: {triplet.head}\n"
            f"Relation: {triplet.relation}\n"
            f"Tail: {triplet.tail}\n"
            "Sentence:"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=80,
        )
        sentences.append(_clean(response.choices[0].message.content))
    return sentences


def verbalize_triplets(triplets: Iterable[Triplet], backend: str = "template", **kwargs) -> List[str]:
    triplet_list = list(triplets)
    if backend == "template":
        return [verbalize_template(t) for t in triplet_list]
    if backend == "llm":
        return verbalize_llm(triplet_list, model=kwargs.get("model", "gpt-4o-mini"))
    raise ValueError(f"Unsupported verbalization backend: {backend}")
