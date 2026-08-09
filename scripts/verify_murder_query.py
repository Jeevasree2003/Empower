#!/usr/bin/env python
"""Verify murder crime_report_india phrasing against Tavily raw top-10."""
from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from ktc.live_config import LiveRetrievalConfig
from ktc.live_retrieval import _domain_allowed, _search_tavily
from ktc.query_builder import _crime_report_query_text


def _host(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def main() -> None:
    key = os.environ.get("LIVE_SEARCH_API_KEY", "")
    if not key:
        raise SystemExit("LIVE_SEARCH_API_KEY not set")
    config = LiveRetrievalConfig.load()
    old = "How to report murder in India official procedure 2026"
    new = _crime_report_query_text("murder")
    for label, q in [("OLD", old), ("NEW", new)]:
        raw = _search_tavily(q, key, 10)
        domains = [_host(item.get("url", "")) or "(empty)" for item in raw]
        allow = sum(1 for item in raw if _domain_allowed(item.get("url", ""), config.trusted_domains))
        print(f"=== {label}: {q!r} ===")
        print(f"  raw_count={len(raw)} allowlisted={allow}")
        print(f"  domains={domains}")


if __name__ == "__main__":
    main()
