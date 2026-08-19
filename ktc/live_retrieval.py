"""Stage 0.75 — Live web search with domain allowlist and local caching."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
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


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet_needs_enrichment(snippet: str) -> bool:
    """True when Tavily snippet is too thin or looks like nav/boilerplate."""
    text = snippet.strip()
    if len(text) < 500:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    nav_markers = sum(
        1
        for line in lines
        if line.startswith("+")
        or line.lower() in {"login", "register as a volunteer"}
        or "banner" in line.lower()
    )
    return nav_markers >= 3


def _page_cache_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / "pages" / f"{digest}.json"


def _read_page_cache(path: Path, ttl_days: int) -> Optional[str]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
        if age_days > ttl_days:
            return None
        return payload.get("text", "")
    except Exception as exc:
        logger.warning("page_cache_read_failed path=%s error=%s", path, exc)
        return None


def _write_page_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_page_text(
    url: str,
    timeout: int = 30,
    max_chars: int = 8000,
    cache_dir: Path | None = None,
    cache_ttl_days: int = 30,
) -> str:
    """Fetch a page and extract readable main-body text for summarization."""
    cache_dir = cache_dir or (DEFAULT_CACHE_DIR / "pages")
    cache_file = _page_cache_path(url, cache_dir)
    cached = _read_page_cache(cache_file, cache_ttl_days)
    if cached is not None:
        logger.info("page_cache_hit url=%s chars=%d", url, len(cached))
        return cached

    import requests

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; EMPOWER-KARE/1.0)"},
    )
    response.raise_for_status()
    raw_html = response.text

    raw_html = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw_html)
    content_html = raw_html
    for pattern in (
        r"(?is)<article[^>]*>(.*?)</article>",
        r"(?is)<main[^>]*>(.*?)</main>",
        r'(?is)<div[^>]+role=["\']main["\'][^>]*>(.*?)</div>',
    ):
        match = re.search(pattern, raw_html)
        if match and len(match.group(1)) > 200:
            content_html = match.group(1)
            break

    text = re.sub(r"(?is)<[^>]+>", " ", content_html)
    text = html.unescape(text)
    text = _normalize_whitespace(text)[:max_chars]
    if text:
        _write_page_cache(cache_file, text)
    return text


def enrich_search_result(result: SearchResult, cache_ttl_days: int = 30) -> SearchResult:
    """Replace thin/nav-heavy Tavily snippets with fetched page text when possible."""
    if not _snippet_needs_enrichment(result.snippet):
        return result
    try:
        page_text = fetch_page_text(result.url, cache_ttl_days=cache_ttl_days)
    except Exception as exc:
        logger.warning("page_fetch_failed url=%s error=%s", result.url, exc)
        return result
    if not page_text:
        return result
    combined = page_text
    snippet = result.snippet.strip()
    if snippet and snippet not in page_text:
        combined = f"{snippet}\n\n{page_text}"
    return SearchResult(
        url=result.url,
        title=result.title,
        snippet=combined[:8000],
        domain=result.domain,
    )


def _search_tavily(query: str, api_key: str, max_results: int, include_domains: Optional[List[str]] = None) -> List[dict]:
    import requests

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    response = requests.post(
        "https://api.tavily.com/search",
        json=payload,
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
              raw_results = _search_tavily(query, api_key, config.results_per_query * 2, include_domains=config.trusted_domains)

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
