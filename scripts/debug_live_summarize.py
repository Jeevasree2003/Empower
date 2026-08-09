#!/usr/bin/env python
"""Step-1 diagnostic: trace text sent to Groq and raw LLM responses per URL."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import requests

from ktc.live_config import LiveRetrievalConfig
from ktc.live_retrieval import SearchResult
from ktc.live_summarize import (
    NO_RELEVANT_INFO,
    SUMMARIZE_SYSTEM_PROMPT,
    _make_llm_client,
    _parse_sentences,
)

# URL -> representative search query from validation runs
CASES = [
    {
        "url": "https://www.who.int/india/health-topics/mental-health",
        "dialogues": "1, 100",
        "query": "current treatment for emotional distress and suicidal ideation in India 2026",
    },
    {
        "url": "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders",
        "dialogues": "1000",
        "query": "What are the symptoms of anxiety and fear?",
    },
    {
        "url": "https://cybercrime.gov.in",
        "dialogues": "3000",
        "query": "How to report murder or homicide threat in India official procedure 2026",
    },
    {
        "url": "https://www.who.int/news-room/fact-sheets/detail/violence-against-women",
        "dialogues": "4500",
        "query": "How to report threat to life and domestic violence in India official procedure 2026",
    },
]


def _find_cached_result(url: str, cache_dir: Path) -> SearchResult | None:
    for path in cache_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in payload.get("results", []):
            if item.get("url", "").rstrip("/") == url.rstrip("/"):
                return SearchResult(
                    url=item["url"],
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    domain=item.get("domain", ""),
                )
    return None


def _fetch_http(url: str) -> tuple[int, int, str]:
    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; EMPOWER-KARE-debug/1.0)"},
        )
        text = resp.text or ""
        return resp.status_code, len(resp.content), text
    except Exception as exc:
        return -1, 0, f"<request failed: {exc}>"


def _build_user_prompt(query: str, result: SearchResult) -> str:
    return (
        f"Search query: {query}\n\n"
        "Summarize only what this source states that helps answer the query.\n\n"
        f"Source ({result.domain}):\n"
        f"Title: {result.title}\n"
        f"Text: {result.snippet}"
    )


def _classify(raw_llm: str, parsed: list[str], snippet: str) -> str:
    snippet_stripped = snippet.strip()
    if len(snippet_stripped) < 50:
        return "(a) near-empty Tavily snippet — extraction/input problem"
    upper = raw_llm.upper().replace(" ", "_")
    if NO_RELEVANT_INFO in upper or not raw_llm.strip():
        return "(b) substantial snippet but Groq returned NO_RELEVANT_INFO or empty"
    if raw_llm.strip() and not parsed:
        return "(c) Groq returned text but _parse_sentences() dropped all lines"
    if parsed:
        return "OK — summaries parsed successfully"
    return "(b) unknown empty outcome"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cache_dir = root / "data" / "live_cache"
    config = LiveRetrievalConfig.load()
    client = _make_llm_client(config)

    print("=" * 80)
    print("STEP 1 DIAGNOSIS — live summarize pipeline")
    print("=" * 80)
    print(
        "\nNOTE: live_retrieval.py does NOT fetch page HTML for summarization; "
        "Groq receives Tavily search snippets only (result.snippet).\n"
    )

    for case in CASES:
        url = case["url"]
        query = case["query"]
        print("\n" + "=" * 80)
        print(f"URL: {url}")
        print(f"Dialogues: {case['dialogues']}")
        print(f"Query: {query}")
        print("=" * 80)

        status, content_len, html = _fetch_http(url)
        print(f"\n(a) Raw HTTP: status={status}, content-length={content_len} bytes")

        cached = _find_cached_result(url, cache_dir)
        if not cached:
            print("\n(b) Tavily snippet: NOT FOUND in live_cache for this URL")
            print("\nClassification: (a) no cached snippet available")
            continue

        snippet = cached.snippet
        print(f"\n(b) Text sent to Groq (Tavily snippet, {len(snippet)} chars):")
        print("-" * 40)
        print(snippet)
        print("-" * 40)

        user_prompt = _build_user_prompt(query, cached)
        print(f"\n(c) Exact user prompt to Groq ({len(user_prompt)} chars):")
        print("-" * 40)
        print(user_prompt)
        print("-" * 40)

        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw_llm = (response.choices[0].message.content or "").strip()
        parsed = _parse_sentences(raw_llm)

        print(f"\n(d) Raw Groq response ({len(raw_llm)} chars):")
        print("-" * 40)
        print(repr(raw_llm) if not raw_llm else raw_llm)
        print("-" * 40)

        print(f"\n(e) After _parse_sentences() ({len(parsed)} lines, min_len>20 filter):")
        if parsed:
            for i, line in enumerate(parsed, 1):
                print(f"  {i}. [{len(line)} chars] {line}")
        else:
            # Show what was dropped
            for line in raw_llm.splitlines():
                cleaned = re.sub(r"^[-*\d.]+\s*", "", line.strip())
                if cleaned and len(cleaned) <= 20:
                    print(f"  DROPPED (len={len(cleaned)}): {cleaned!r}")

        classification = _classify(raw_llm, parsed, snippet)
        print(f"\nClassification: {classification}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
