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


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if text and text[-1] not in ".!?":
        text += "."
    return text


def verbalize_template(triplet: Triplet) -> str:
    """Convert a triplet into a natural-language sentence without naive concatenation."""
    head = triplet.head.strip()
    relation = triplet.relation.strip()
    tail = triplet.tail.strip()

    relation_lower = relation.lower()
    if relation_lower.startswith(("is ", "are ", "was ", "were ")):
        sentence = f"{head} {relation} {tail}"
    elif any(relation_lower.startswith(v) for v in ("can ", "may ", "should ", "must ", "will ")):
        sentence = f"{head} {relation} {tail}"
    elif relation_lower.endswith(" by"):
        sentence = f"{tail} {relation} {head}"
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
    for triplet in triplets:
        prompt = (
            "Convert the following knowledge triplet into one fluent English sentence. "
            "Do not simply concatenate the parts; write a natural sentence.\n"
            f"Head: {triplet.head}\nRelation: {triplet.relation}\nTail: {triplet.tail}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
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
