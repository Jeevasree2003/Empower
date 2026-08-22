"""Load live retrieval configuration and trusted domain allowlist."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import threading

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "live_retrieval_config.yaml"
DEFAULT_DOMAINS_PATH = PACKAGE_ROOT / "config" / "trusted_domains.yaml"


@dataclass
class LiveRetrievalConfig:
    enable_live_retrieval: bool = False
    max_live_queries_per_dialogue: int = 3
    max_api_calls_per_run: int = 100
    cache_ttl_days: int = 30
    results_per_query: int = 5
    search_provider: str = "tavily"
    # extractive = no LLM (no Groq quota); llm = Groq/OpenAI with extractive fallback on errors.
    summarize_backend: str = "extractive"
    llm_model: str = "gpt-4o-mini"
    llm_api_base: str = ""
    estimated_cost_per_dialogue_usd: float = 0.03
    page_fetch_timeout: int = 8
    search_timeout: int = 10
    failure_cache_ttl_days: int = 1
    max_concurrent_fetches: int = 4
    max_concurrent_queries: int = 3
    max_live_retrieval_seconds: float = 20
    live_sentence_top_k: int = 8
    live_sentence_candidates_per_page: int = 8
    trusted_domains: List[str] = field(default_factory=list)
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)
    domains_path: Path = field(default_factory=lambda: DEFAULT_DOMAINS_PATH)

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        domains_path: Path | None = None,
    ) -> "LiveRetrievalConfig":
        config_path = config_path or DEFAULT_CONFIG_PATH
        domains_path = domains_path or DEFAULT_DOMAINS_PATH

        data = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        trusted: List[str] = []
        if domains_path.exists():
            with domains_path.open("r", encoding="utf-8") as f:
                domain_groups = yaml.safe_load(f) or {}
            for group in domain_groups.values():
                if isinstance(group, list):
                    trusted.extend(d.lower().strip() for d in group if d)

        return cls(
            enable_live_retrieval=bool(data.get("enable_live_retrieval", False)),
            max_live_queries_per_dialogue=int(data.get("max_live_queries_per_dialogue", 3)),
            max_api_calls_per_run=int(data.get("max_api_calls_per_run", 100)),
            cache_ttl_days=int(data.get("cache_ttl_days", 30)),
            results_per_query=int(data.get("results_per_query", 5)),
            search_provider=str(data.get("search_provider", "tavily")),
            summarize_backend=str(data.get("summarize_backend", "extractive") or "extractive"),
            llm_model=str(data.get("llm_model", "gpt-4o-mini")),
            llm_api_base=str(data.get("llm_api_base", "") or ""),
            estimated_cost_per_dialogue_usd=float(data.get("estimated_cost_per_dialogue_usd", 0.03)),
            page_fetch_timeout=int(data.get("page_fetch_timeout", 8)),
            search_timeout=int(data.get("search_timeout", 10)),
            failure_cache_ttl_days=int(data.get("failure_cache_ttl_days", 1)),
            max_concurrent_fetches=int(data.get("max_concurrent_fetches", 4)),
            max_concurrent_queries=int(data.get("max_concurrent_queries", 3)),
            max_live_retrieval_seconds=float(data.get("max_live_retrieval_seconds", 20)),
            live_sentence_candidates_per_page=int(
                data.get(
                    "live_sentence_candidates_per_page",
                    data.get("live_sentence_top_k", 8),
                )
            ),
            live_sentence_top_k=int(
                data.get(
                    "live_sentence_candidates_per_page",
                    data.get("live_sentence_top_k", 8),
                )
            ),
            trusted_domains=sorted(set(trusted)),
            config_path=config_path,
            domains_path=domains_path,
        )


class ApiCallBudget:
    """Track API calls against per-run ceiling. Thread-safe for concurrent queries."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def can_call(self) -> bool:
        with self._lock:
            return self.used < self.limit

    def record(self, count: int = 1) -> bool:
        """Atomically reserve ``count`` calls. Returns False without incrementing if over limit."""
        with self._lock:
            if self.used + count > self.limit:
                return False
            self.used += count
            return True
