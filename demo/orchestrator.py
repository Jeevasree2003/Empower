"""One Gradio turn: retrieve, RDPT draft, Ollama refine, audit log."""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from demo.audit_log import append_turn
from demo.fallbacks import EMPTY_KNOWLEDGE_FALLBACK, crisis_or_generic
from demo.refine import refine
from demo.retrieve import RetrieveResult, retrieve

logger = logging.getLogger(__name__)


def _history_dicts(history) -> List[dict]:
    if not history:
        return []
    first = history[0]
    if isinstance(first, dict) and "role" in first:
        return [item for item in history if item.get("content")]
    messages: List[dict] = []
    for pair in history:
        if not pair:
            continue
        user, bot = (list(pair) + [None, None])[:2]
        if user:
            messages.append({"role": "user", "content": user})
        if bot:
            messages.append({"role": "assistant", "content": bot})
    return messages


class RakshakOrchestrator:
    def __init__(self, use_rdpt: bool = True, dialogue_id: Optional[str] = None):
        self.use_rdpt = use_rdpt
        self.dialogue_id = dialogue_id or f"demo-{uuid.uuid4().hex[:8]}"

    def respond(self, user_message: str, history=None) -> Dict[str, str]:
        text = (user_message or "").strip()
        record = {
            "dialogue_id": self.dialogue_id,
            "user_message": text,
            "retrieved_knowledge": "",
            "verbalized": [],
            "rdpt_draft": "",
            "draft_status": "skipped",
            "final_reply": "",
            "retrieve_error": "",
            "live_enabled": False,
            "live_sentences": 0,
        }
        if not text:
            record["final_reply"] = "Please tell me what happened so I can help."
            append_turn(record)
            return record

        retrieved: RetrieveResult = retrieve(
            text,
            history_messages=_history_dicts(history),
            dialogue_id=self.dialogue_id,
        )
        record["retrieve_error"] = retrieved.error
        record["retrieved_knowledge"] = retrieved.final_knowledge_text
        record["verbalized"] = retrieved.verbalized
        record["live_enabled"] = retrieved.live_enabled
        record["live_sentences"] = len(retrieved.live_verbalized)
        knowledge = retrieved.final_knowledge_text

        draft = ""
        if self.use_rdpt:
            try:
                from demo.rdpt_generate import generate_draft

                draft = generate_draft(retrieved.dialog_history, knowledge)
            except Exception:
                logger.exception("rdpt_stage_failed")
                draft = ""
        record["rdpt_draft"] = draft

        if retrieved.error and not knowledge:
            record["final_reply"] = crisis_or_generic(text)
            record["draft_status"] = "fallback"
            append_turn(record)
            return record

        try:
            final_reply, status = refine(
                retrieved.dialog_history,
                knowledge,
                draft,
                user_message=text,
            )
        except Exception:
            logger.exception("refine_failed")
            final_reply = EMPTY_KNOWLEDGE_FALLBACK if not knowledge else crisis_or_generic(text)
            status = "fallback"
        record["draft_status"] = status
        record["final_reply"] = final_reply
        append_turn(record)
        return record
