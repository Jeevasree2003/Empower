"""Local Ollama refinement. The LLM output is the only text shown to the user."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional, Tuple

from demo.fallbacks import EMPTY_KNOWLEDGE_FALLBACK, crisis_or_generic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Rakshak, a calm women-and-child safety counselling assistant for a live demo in India.
Ground helplines, statutes, and procedures ONLY in the knowledge context. Do not invent section numbers, URLs, or phone numbers.
Do not mention POCSO, IPC, BNS, or any Act unless that exact name already appears in the knowledge context.
Physical beating of a child is not automatically a sexual offence.
If the user is in immediate danger, tell them to call 112 and the nearest police. You may also mention 181, 1098, or KIRAN 1800-599-0019 when they appear in the knowledge or the person is in danger.
You are not a lawyer and must not role-play as one.
If the knowledge context is empty, give a short supportive reply and ask one clarifying question. Do not fabricate facts.
Never mention drafts, models, prefixes, or that any output was discarded.
Reply in 2–6 short sentences."""

USER_TEMPLATE = """Conversation:
{history}

Knowledge context:
{knowledge}

Draft reply from our domain-specific model:
{draft}

Here is a draft reply from our domain-specific model, grounded in the knowledge context above. If the draft is coherent, relevant to the conversation, and appropriately answers the user, refine it for clarity and tone. If the draft is repetitive, generic, off-topic, or does not make sense given the conversation, ignore it completely and write a new response directly from the knowledge context and conversation history instead. Never mention that there was a draft or that any model output was discarded.

Write the counsellor reply now."""

_HAVE_YOU_LOOP = re.compile(r"(have you.{0,40}){3,}", re.I)
_GREAT_DAY = re.compile(r"have a (great|wonderful|nice) day", re.I)


def draft_should_discard(draft: str) -> bool:
    text = (draft or "").strip()
    if not text:
        return True
    if len(text) > 80:
        return True
    if _HAVE_YOU_LOOP.search(text):
        return True
    if text.lower().startswith("this. have you"):
        return True
    if _GREAT_DAY.search(text) and "have you" in text.lower():
        return True
    if text.lower().count("have you") >= 3:
        return True
    return False


def _chat_ollama(model: str, system: str, user: str) -> Optional[str]:
    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.get("message") or {}).get("content") or ""
    except Exception as exc:
        logger.warning("ollama_package_failed err=%s", exc)
        return None


def _chat_openai_compat(model: str, system: str, user: str) -> Optional[str]:
    base = os.environ.get("LLM_API_BASE", "http://localhost:11434/v1").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "ollama")
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base, api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("openai_compat_failed err=%s", exc)
        return None


def refine(
    history: str,
    knowledge: str,
    draft: str,
    user_message: str = "",
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (final_reply, draft_status) where status is used, discarded, or fallback."""
    discarded = draft_should_discard(draft)
    shown_draft = "" if discarded else draft.strip()
    status = "discarded" if discarded else "used"
    knowledge_block = (knowledge or "").strip() or "(none)"
    prompt = USER_TEMPLATE.format(
        history=history or "(empty)",
        knowledge=knowledge_block,
        draft=shown_draft or "(empty — write from knowledge and conversation only)",
    )
    model_name = model or os.environ.get("LLM_MODEL") or "mistral"
    text = _chat_ollama(model_name, SYSTEM_PROMPT, prompt)
    if text is None:
        text = _chat_openai_compat(model_name, SYSTEM_PROMPT, prompt)
    cleaned = _strip_ungrounded_statutes((text or "").strip(), knowledge)
    if cleaned:
        return cleaned, status
    if knowledge_block == "(none)":
        return EMPTY_KNOWLEDGE_FALLBACK, "fallback"
    return crisis_or_generic(user_message), "fallback"


def _strip_ungrounded_statutes(reply: str, knowledge: str) -> str:
    if not reply:
        return ""
    know = (knowledge or "").lower()
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", reply.strip()):
        text = sentence.strip()
        if not text:
            continue
        if re.search(r"\bpocso\b", text, re.I) and "pocso" not in know:
            continue
        bad = False
        for number in re.findall(r"(?:section|ipc|bns)\s+(\d+[a-z]?)", text, re.I):
            if not re.search(rf"\b{re.escape(number)}\b", know):
                bad = True
                break
        if bad:
            continue
        kept.append(text)
    return " ".join(kept).strip()
