"""Load live retrieval configuration and trusted domain allowlist."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "live_retrieval_config.yaml"
DEFAULT_DOMAINS_PATH = PACKAGE_ROOT / "config" / "trusted_domains.yaml"


@dataclass
class LiveRetrievalConfig:
    enable_live_retrieval: bool = True
    max_live_queries_per_dialogue: int = 3
    max_api_calls_per_run: int = 100
    cache_ttl_days: int = 30
    results_per_query: int = 5
    search_provider: str = "tavily"
    llm_model: str = "gpt-4o-mini"
    llm_api_base: str = ""
    estimated_cost_per_dialogue_usd: float = 0.03
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
            enable_live_retrieval=bool(data.get("enable_live_retrieval", True)),
            max_live_queries_per_dialogue=int(data.get("max_live_queries_per_dialogue", 3)),
            max_api_calls_per_run=int(data.get("max_api_calls_per_run", 100)),
            cache_ttl_days=int(data.get("cache_ttl_days", 30)),
            results_per_query=int(data.get("results_per_query", 5)),
            search_provider=str(data.get("search_provider", "tavily")),
            llm_model=str(data.get("llm_model", "gpt-4o-mini")),
            llm_api_base=str(data.get("llm_api_base", "") or ""),
            estimated_cost_per_dialogue_usd=float(data.get("estimated_cost_per_dialogue_usd", 0.03)),
            trusted_domains=sorted(set(trusted)),
            config_path=config_path,
            domains_path=domains_path,
        )


class ApiCallBudget:
    """Track API calls against per-run ceiling."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def can_call(self) -> bool:
        return self.used < self.limit

    def record(self, count: int = 1) -> bool:
        if self.used + count > self.limit:
            return False
        self.used += count
        return True
