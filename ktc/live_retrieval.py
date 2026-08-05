"""Stage 0.75 — Live web search with domain allowlist and local caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from ktc.live_config import ApiCallBudget, LiveRetrievalConfig

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "live_cache"


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    domain: str

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "domain": self.domain,
        }


def _domain_allowed(url: str, trusted_domains: List[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in trusted_domains)


def _cache_path(query: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _read_cache(path: Path, ttl_days: int) -> Optional[List[SearchResult]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
        if age_days > ttl_days:
            return None
        return [SearchResult(**r) for r in payload.get("results", [])]
    except Exception as exc:
        logger.warning("cache_read_failed path=%s error=%s", path, exc)
        return None


def _write_cache(path: Path, results: List[SearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _search_tavily(query: str, api_key: str, max_results: int) -> List[dict]:
    import requests

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _search_serpapi(query: str, api_key: str, max_results: int) -> List[dict]:
    import requests

    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max_results,
        },
        timeout=30,
    )
    response.raise_for_status()
    organic = response.json().get("organic_results", [])
    return [
        {
            "url": item.get("link", ""),
            "title": item.get("title", ""),
            "content": item.get("snippet", ""),
        }
        for item in organic
    ]


def search_allowlisted(
    query: str,
    config: LiveRetrievalConfig,
    budget: Optional[ApiCallBudget] = None,
    cache_dir: Path | None = None,
) -> List[SearchResult]:
    """Search the web and return only allowlisted-domain results. Never raises."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_file = _cache_path(query, cache_dir)

    cached = _read_cache(cache_file, config.cache_ttl_days)
    if cached is not None:
        logger.info("cache_hit query=%r results=%d", query, len(cached))
        return cached

    api_key = os.environ.get("LIVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        logger.warning("LIVE_SEARCH_API_KEY not set; skipping live search for query=%r", query)
        return []

    if budget is not None and not budget.can_call():
        logger.warning("API call budget exhausted; skipping query=%r", query)
        return []

    try:
        if budget is not None:
            budget.record(1)

        raw_results: List[dict] = []
        if config.search_provider == "serpapi":
            raw_results = _search_serpapi(query, api_key, config.results_per_query * 2)
        else:
            raw_results = _search_tavily(query, api_key, config.results_per_query * 2)

        allowlisted: List[SearchResult] = []
        for item in raw_results:
            url = item.get("url") or item.get("link") or ""
            if not url or not _domain_allowed(url, config.trusted_domains):
                continue
            domain = urlparse(url).netloc.lower()
            allowlisted.append(
                SearchResult(
                    url=url,
                    title=(item.get("title") or "").strip(),
                    snippet=(item.get("content") or item.get("snippet") or "").strip(),
                    domain=domain,
                )
            )
            if len(allowlisted) >= config.results_per_query:
                break

        if not allowlisted:
            logger.warning("no_allowlisted_results query=%r", query)
        else:
            _write_cache(cache_file, allowlisted)

        return allowlisted
    except Exception as exc:
        logger.warning("live_search_failed query=%r error=%s", query, exc)
        return []
