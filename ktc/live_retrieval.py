"""Stage 0.75 — Live web search with domain allowlist and local caching."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, TypeVar
from urllib.parse import urlparse

from ktc.live_config import ApiCallBudget, LiveRetrievalConfig

logger = logging.getLogger(__name__)
# Sites like indiankanoon.org run anti-bot protection that pattern-matches
# non-browser User-Agents and blocks them outright. A self-identifying UA
# (e.g. "EMPOWER-KARE/1.0") gets a 100% block rate from some sources; a
# realistic browser header set gets treated like ordinary traffic.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "live_cache"

T = TypeVar("T")


@dataclass
class LiveDeadline:
    """Wall-clock budget for one live-retrieval turn."""

    limit_seconds: float
    started: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        if self.limit_seconds <= 0:
            return 0.0
        return max(0.0, self.limit_seconds - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def warn_if_expired(self) -> bool:
        if not self.expired():
            return False
        logger.warning("live_retrieval_time_budget_exceeded elapsed=%.1fs", self.elapsed())
        return True


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


def _failure_cache_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / "failures" / f"{digest}.json"


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


def _read_failure_record(path: Path, ttl_days: int) -> Optional[tuple]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        age_seconds = (datetime.now(timezone.utc) - fetched_at).total_seconds()
        if age_seconds > ttl_days * 86400:
            return None
        reason = str(payload.get("reason") or "unknown")
        return age_seconds, reason
    except Exception as exc:
        logger.warning("failure_cache_read_failed path=%s error=%s", path, exc)
        return None


def _write_failure_cache(path: Path, url: str, error: str, reason: str = "unknown") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "error": error,
        "reason": reason,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_fetch_failure(exc: BaseException) -> str:
    """Stable reason token for logs and the failure cache."""
    try:
        import requests
    except ImportError:
        requests = None  # type: ignore

    if requests is not None:
        if isinstance(exc, (requests.Timeout, requests.exceptions.Timeout)):
            return "timeout"
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "connection_error"
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status:
                return f"status_{status}"
            return "http_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "connection" in name:
        return "connection_error"
    return name or "unknown"


def run_io_tasks(
    tasks: Sequence[Callable[[], T]],
    max_workers: int,
    deadline: Optional[LiveDeadline] = None,
) -> List[Optional[T]]:
    """Run independent I/O callables concurrently. Abandoned tasks yield None."""
    results: List[Optional[T]] = [None] * len(tasks)
    if not tasks:
        return results
    workers = max(1, min(max_workers, len(tasks)))
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for index, fn in enumerate(tasks):
            if deadline is not None and deadline.warn_if_expired():
                break
            futures[executor.submit(fn)] = index
        pending = set(futures)
        while pending:
            timeout = None if deadline is None else deadline.remaining()
            if timeout is not None and timeout <= 0:
                deadline.warn_if_expired()
                break
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                if deadline is not None:
                    deadline.warn_if_expired()
                break
            for fut in done:
                index = futures[fut]
                try:
                    results[index] = fut.result()
                except Exception as exc:
                    logger.warning("io_task_failed index=%s error=%s", index, exc)
    finally:
        executor.shutdown(wait=False)
    return results


def fetch_page_text(
    url: str,
    timeout: int = 8,
    max_chars: int = 8000,
    cache_dir: Path | None = None,
    cache_ttl_days: int = 30,
    failure_cache_ttl_days: int = 1,
    deadline: Optional[LiveDeadline] = None,
) -> str:
    """Fetch a page and extract readable main-body text for summarization."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    fail_file = _failure_cache_path(url, cache_dir)
    cached_failure = _read_failure_record(fail_file, failure_cache_ttl_days)
    if cached_failure is not None:
        fail_age, fail_reason = cached_failure
        logger.info(
            "skipping_known_bad_url url=%s cached_failure_age=%.0fs reason=%s",
            url,
            fail_age,
            fail_reason,
        )
        return ""

    cache_file = _page_cache_path(url, cache_dir)
    cached = _read_page_cache(cache_file, cache_ttl_days)
    if cached is not None:
        logger.info("page_cache_hit url=%s chars=%d", url, len(cached))
        return cached

    if deadline is not None and deadline.warn_if_expired():
        return ""
    wait_s = timeout
    if deadline is not None:
        wait_s = min(timeout, max(0.1, deadline.remaining()))
        if wait_s < 0.1:
            return ""

    import requests

    try:
        response = requests.get(
            url,
            timeout=wait_s,
            headers=BROWSER_HEADERS,
        )
        response.raise_for_status()
        raw_html = response.text
    except Exception as exc:
        reason = classify_fetch_failure(exc)
        _write_failure_cache(fail_file, url, str(exc), reason=reason)
        logger.warning("page_fetch_failed url=%s reason=%s error=%s", url, reason, exc)
        raise

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


def enrich_search_result(
    result: SearchResult,
    cache_ttl_days: int = 30,
    timeout: int = 8,
    failure_cache_ttl_days: int = 1,
    deadline: Optional[LiveDeadline] = None,
    cache_dir: Path | None = None,
) -> SearchResult:
    """Replace thin/nav-heavy Tavily snippets with fetched page text when possible."""
    if not _snippet_needs_enrichment(result.snippet):
        return result
    if deadline is not None and deadline.expired():
        deadline.warn_if_expired()
        return result
    try:
        page_text = fetch_page_text(
            result.url,
            timeout=timeout,
            cache_ttl_days=cache_ttl_days,
            failure_cache_ttl_days=failure_cache_ttl_days,
            deadline=deadline,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        reason = classify_fetch_failure(exc)
        logger.debug("page_fetch_failed url=%s reason=%s error=%s", result.url, reason, exc)
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


def enrich_search_results(
    results: List[SearchResult],
    config: LiveRetrievalConfig,
    deadline: Optional[LiveDeadline] = None,
    cache_dir: Path | None = None,
) -> List[SearchResult]:
    """Fetch pages for a result set concurrently; preserve input order."""
    if not results:
        return []

    def _one(item: SearchResult) -> SearchResult:
        return enrich_search_result(
            item,
            cache_ttl_days=config.cache_ttl_days,
            timeout=config.page_fetch_timeout,
            failure_cache_ttl_days=config.failure_cache_ttl_days,
            deadline=deadline,
            cache_dir=cache_dir,
        )

    fetched = run_io_tasks(
        [lambda item=item: _one(item) for item in results],
        max_workers=config.max_concurrent_fetches,
        deadline=deadline,
    )
    return [got if got is not None else original for got, original in zip(fetched, results)]


def _search_tavily(
    query: str,
    api_key: str,
    max_results: int,
    timeout: int = 10,
    include_domains: Optional[List[str]] = None,
) -> List[dict]:
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
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _search_serpapi(query: str, api_key: str, max_results: int, timeout: int = 10) -> List[dict]:
    import requests

    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max_results,
        },
        timeout=timeout,
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
    deadline: Optional[LiveDeadline] = None,
) -> List[SearchResult]:
    """Search the web and return only allowlisted-domain results. Never raises."""
    if deadline is not None and deadline.warn_if_expired():
        return []

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

    if budget is not None and not budget.record(1):
        logger.warning("API call budget exhausted; skipping query=%r", query)
        return []

    timeout = config.search_timeout
    if deadline is not None:
        timeout = max(1, min(timeout, int(deadline.remaining()) or 1))

    try:
        raw_results: List[dict] = []
        if config.search_provider == "serpapi":
            raw_results = _search_serpapi(query, api_key, config.results_per_query * 2, timeout=timeout)
        else:
            raw_results = _search_tavily(
                query,
                api_key,
                config.results_per_query * 2,
                timeout=timeout,
                include_domains=config.trusted_domains,
            )

        allowlisted: List[SearchResult] = []
        for item in raw_results:
            url = item.get("url") or item.get("link") or ""
            if not url or not _domain_allowed(url, config.trusted_domains):
                continue
            if "/bitstream/" in url.lower() or url.lower().endswith(".pdf"):
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
