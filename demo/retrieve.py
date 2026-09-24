"""KTC retrieve: local corpus + counseling bank + optional live web search."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ktc.counseling_bank import counseling_candidates
from ktc.knowledge_item import KnowledgeCandidate
from ktc.pipeline import HybridRunResult, KnowledgeTripletPipeline
from ktc.ranking import ranking_query_from_history
from ktc.reply_knowledge import is_reply_usable

from demo.corpus_index import PassageCorpus

logger = logging.getLogger(__name__)

_PIPELINE: Optional[KnowledgeTripletPipeline] = None
_CORPUS: Optional[PassageCorpus] = None
_ENV_LOADED = False

_JUNK_KNOWLEDGE = re.compile(
    r"should take what legal|what legal actions|\b\d{1,2}\s+sep\b|"
    r"Loading\.\.\.|Download \(|\d+\.\d+\s*KB|##\s|1098Child|Help Line 14433",
    re.I,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_demo_env() -> None:
    """Load LIVE_SEARCH_API_KEY and related vars before Tavily is called."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv is not installed; .env will not be loaded")
        return
    root = _repo_root()
    load_dotenv(root / ".env")
    if not os.environ.get("LIVE_SEARCH_API_KEY", "").strip():
        load_dotenv(root / ".env.example")


def live_requested(enable_live: Optional[bool] = None) -> bool:
    if enable_live is not None:
        return bool(enable_live)
    flag = os.environ.get("RAKSHAK_OFFLINE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    return True


def get_pipeline() -> KnowledgeTripletPipeline:
    global _PIPELINE
    load_demo_env()
    if _PIPELINE is None:
        _PIPELINE = KnowledgeTripletPipeline(
            verbalization_backend="template",
            ranker_backend="auto",
            context_fallback=False,
        )
        try:
            _PIPELINE._get_ranker()
        except Exception:
            logger.warning("sbert_unavailable; using TfidfRanker")
            from ktc.ranking import get_ranker

            _PIPELINE.ranker = get_ranker("tfidf")
        logger.info(
            "ktc_pipeline yaml_live=%s api_key_set=%s",
            _PIPELINE.live_config.enable_live_retrieval,
            bool(os.environ.get("LIVE_SEARCH_API_KEY", "").strip()),
        )
    return _PIPELINE


def get_corpus() -> PassageCorpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = PassageCorpus(ranker=get_pipeline()._get_ranker())
    return _CORPUS


@dataclass
class RetrieveResult:
    dialog_history: str
    knowledge_text: str
    verbalized: List[str] = field(default_factory=list)
    final_knowledge_text: str = ""
    sources: List[str] = field(default_factory=list)
    passages: List[str] = field(default_factory=list)
    corpus_source: str = ""
    hybrid: Optional[HybridRunResult] = None
    live_enabled: bool = False
    live_verbalized: List[str] = field(default_factory=list)
    error: str = ""


def _content_to_text(content) -> str:
    """Normalize a Gradio chat message's `content` field to plain text.

    Depending on Gradio version/config, `content` may be a plain string or a
    list of content parts (e.g. [{"type": "text", "text": "..."}]). This
    handles both shapes so format_history never crashes on a list.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()


def format_history(messages: List[dict], latest_user: str) -> str:
    parts: List[str] = []
    for item in messages:
        role = (item.get("role") or "").lower()
        content = _content_to_text(item.get("content"))
        if not content:
            continue
        if role in {"user", "victim"}:
            parts.append(f"victim: {content}")
        elif role in {"assistant", "agent", "bot"}:
            parts.append(f"agent: {content}")
    latest = (latest_user or "").strip()
    if not parts or parts[-1] != f"victim: {latest}":
        if latest:
            parts.append(f"victim: {latest}")
    return " <|sep|> ".join(parts)


def _filter_knowledge(
    blob: str,
    live_sentences: List[str],
    bank_sentences: Optional[List[str]] = None,
) -> str:
    kept: List[str] = []
    seen = set()

    def add(text: str, source: str) -> None:
        piece = (text or "").strip()
        if not piece or _JUNK_KNOWLEDGE.search(piece):
            return
        key = piece.lower()
        if key in seen:
            return
        if not is_reply_usable(KnowledgeCandidate(text=piece, source=source)):
            return
        seen.add(key)
        kept.append(piece if piece.endswith((".", "!", "?")) else piece + ".")

    for piece in bank_sentences or []:
        add(piece, "counseling_bank")
    for piece in live_sentences:
        add(piece, "live_sentence_direct")
    for piece in re.split(r"(?<=[.!?])\s+", blob or ""):
        helpline = bool(
            re.search(r"\b(112|181|1098|kiran|childline|child welfare)\b", piece or "", re.I)
        )
        add(piece, "counseling_bank" if helpline else "static_dataset")
    return " ".join(kept).strip()


def retrieve(
    user_message: str,
    history_messages: Optional[List[dict]] = None,
    dialogue_id: str = "demo",
    top_passages: int = 8,
    enable_live: Optional[bool] = None,
) -> RetrieveResult:
    load_demo_env()
    history = format_history(history_messages or [], user_message)
    result = RetrieveResult(dialog_history=history, knowledge_text="")
    live_on = live_requested(enable_live)
    result.live_enabled = live_on
    if live_on and not os.environ.get("LIVE_SEARCH_API_KEY", "").strip():
        logger.warning(
            "live_search_skipped_no_key: set LIVE_SEARCH_API_KEY in .env "
            "(copy from .env.example) or pass --offline"
        )
    try:
        pipeline = get_pipeline()
        corpus = get_corpus()
        result.corpus_source = corpus.source
        query = ranking_query_from_history(history, nlp=pipeline._get_nlp(), ranker=pipeline._get_ranker())
        ranked = corpus.top_k(query or user_message, k=top_passages)
        result.passages = [text for text, _score in ranked]
        knowledge_blob = " ".join(result.passages)
        result.knowledge_text = knowledge_blob
        hybrid = pipeline.run_hybrid(
            knowledge_blob,
            history,
            enable_live=live_on,
            synthesize=False,
            dialogue_id=dialogue_id,
            context_fallback=False,
        )
        result.hybrid = hybrid
        result.verbalized = list(hybrid.verbalized or [])
        result.live_verbalized = list(hybrid.live_verbalized or [])
        result.sources = list(hybrid.final_knowledge_sources or [])
        bank_texts = [item.text for item in counseling_candidates([], user_message)]
        result.final_knowledge_text = _filter_knowledge(
            hybrid.final_knowledge_text or "",
            result.live_verbalized,
            bank_texts,
        )
    except Exception as exc:
        logger.exception("retrieve_failed")
        result.error = str(exc)
    return result