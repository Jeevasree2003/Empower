#!/usr/bin/env python
"""Diagnose empty allowlisted retrieval: raw Tavily results before domain filter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from ktc.entity_extraction import extract_entities
from ktc.live_config import LiveRetrievalConfig
from ktc.live_retrieval import _domain_allowed, _search_tavily, search_allowlisted
from ktc.query_builder import SearchQuery, build_queries

ROLE_MAP = {"bot": "agent", "agent": "agent", "user": "victim", "victim": "victim"}

# Queries that logged no_allowlisted_results during eval_ranking_sample (seed=42).
KNOWN_EMPTY_QUERIES = [
    "How to report rape in India official procedure 2026",
    "What is complaint in Indian law 2026?",
    "How to report murder in India official procedure 2026",
]

# Indiacode-oriented probes — cheaper fix if these return allowlisted hits.
INDIACODE_PROBE_TEMPLATES = [
    "IPC Section {section} indiacode.nic.in",
    "{crime} Indian Penal Code section indiacode",
    "indiacode.nic.in {crime} punishment India",
]


def _safe_print(text: str = "") -> None:
    """Print without crashing on Windows cp1252 consoles."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
    sys.stdout.buffer.flush()


def load_dialogue(path: Path, dialogue_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record["dialogue_id"]) == str(dialogue_id):
                return record
    raise SystemExit(f"Dialogue {dialogue_id} not found")


def first_turn_victim(dialogue: dict) -> str:
    utterances = sorted(dialogue["utterances"], key=lambda u: int(u["utterance_no"]))
    for utterance in utterances:
        role = ROLE_MAP.get(utterance["author_role"], utterance["author_role"])
        if role == "victim":
            return utterance["utterance"].strip()
    raise SystemExit(f"No victim utterance in dialogue {dialogue['dialogue_id']}")


def _host(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _classify_failure(
    query: str,
    raw_results: list[dict],
    trusted_domains: list[str],
) -> str:
    if not raw_results:
        return "(c) tavily_empty — Tavily returned zero results for this query"

    allowlisted_in_raw = [
        item
        for item in raw_results
        if _domain_allowed(item.get("url") or item.get("link") or "", trusted_domains)
    ]
    if allowlisted_in_raw:
        return (
            "(b) query_phrasing — allowlisted domain(s) appear in raw Tavily results "
            "but were not selected (ranking/result cap) or query targets procedural pages "
            "while indiacode returns statute text"
        )

    domains = sorted({_host(item.get("url", "")) for item in raw_results if item.get("url")})
    indiacode_like = [d for d in domains if "indiacode" in d or d.endswith(".gov.in")]
    if indiacode_like:
        return (
            f"(b) query_phrasing — near-miss govt/legal domains in raw results: {indiacode_like[:5]}"
        )
    return (
        "(a) missing_allowlist_coverage — raw Tavily results contain no trusted domains; "
        f"top domains: {domains[:6]}"
    )


def _print_raw_results(raw_results: list[dict], trusted_domains: list[str], limit: int = 8) -> None:
    if not raw_results:
        _safe_print("    (Tavily returned 0 results)")
        return
    for i, item in enumerate(raw_results[:limit], 1):
        url = item.get("url") or item.get("link") or ""
        host = _host(url)
        allowed = _domain_allowed(url, trusted_domains)
        title = (item.get("title") or "").strip()
        snippet = (item.get("content") or item.get("snippet") or "").strip()
        tag = "ALLOWLISTED" if allowed else "filtered_out"
        _safe_print(f"\n    [{i}] [{tag}] {host}")
        _safe_print(f"        url: {url}")
        _safe_print(f"        title: {title[:120]}")
        _safe_print(f"        snippet: {snippet[:200]}{'...' if len(snippet) > 200 else ''}")


def diagnose_query(
    query: SearchQuery | str,
    config: LiveRetrievalConfig,
    api_key: str,
    *,
    run_probes: bool = False,
    crime_hint: str | None = None,
) -> None:
    if isinstance(query, SearchQuery):
        query_text = query.text
        meta = f"template={query.template} entity={query.entity_text!r} category={query.entity_category}"
    else:
        query_text = query
        meta = "known_empty_from_eval"

    _safe_print("\n" + "=" * 88)
    _safe_print(f"QUERY: {query_text!r}")
    _safe_print(f"  ({meta})")
    _safe_print("=" * 88)

    raw = _search_tavily(query_text, api_key, config.results_per_query * 2)
    allowlisted = search_allowlisted(query_text, config)  # may hit cache from eval

    _safe_print(f"\n  Raw Tavily count: {len(raw)} | After allowlist: {len(allowlisted)}")
    _safe_print(f"  Diagnosis: {_classify_failure(query_text, raw, config.trusted_domains)}")
    _safe_print("\n  --- RAW TAVILY (before allowlist filter) ---")
    _print_raw_results(raw, config.trusted_domains)

    if allowlisted:
        _safe_print("\n  --- ALLOWLISTED (what pipeline kept) ---")
        for i, r in enumerate(allowlisted, 1):
            _safe_print(f"    [{i}] {r.domain} - {r.url}")

    if run_probes and crime_hint:
        section_map = {"rape": "376", "murder": "302", "complaint": "190"}
        section = section_map.get(crime_hint.lower())
        if section:
            _safe_print(f"\n  --- INDIACODE PROBE (section {section}, not used in production) ---")
            for tmpl in INDIACODE_PROBE_TEMPLATES:
                probe = tmpl.format(section=section, crime=crime_hint)
                probe_raw = _search_tavily(probe, api_key, config.results_per_query * 2)
                hits = sum(
                    1
                    for item in probe_raw
                    if _domain_allowed(item.get("url", ""), config.trusted_domains)
                )
                _safe_print(f"\n    Probe: {probe!r} -> {len(probe_raw)} raw, {hits} allowlisted")
                _print_raw_results(probe_raw, config.trusted_domains, limit=3)


def diagnose_dialogue(
    dialogue_id: str,
    input_path: Path,
    config: LiveRetrievalConfig,
    api_key: str,
    nlp,
    *,
    run_probes: bool,
) -> list[str]:
    dialogue = load_dialogue(input_path, dialogue_id)
    victim = first_turn_victim(dialogue)
    entities = extract_entities(victim, nlp=nlp)
    queries = build_queries(entities, max_queries=config.max_live_queries_per_dialogue)

    _safe_print("\n" + "#" * 88)
    _safe_print(f"DIALOGUE {dialogue_id}")
    _safe_print("#" * 88)
    _safe_print(f"Victim: {victim!r}")
    _safe_print("Entities:")
    for ent in entities:
        _safe_print(f"  - [{ent['category']}] {ent['text']!r}")

    empty_queries: list[str] = []
    crime_hint = next((e["text"] for e in entities if e["category"] == "crime"), None)

    for query in queries:
        allowlisted = search_allowlisted(query.text, config)
        if not allowlisted:
            empty_queries.append(query.text)
        diagnose_query(query, config, api_key, run_probes=run_probes, crime_hint=crime_hint)

    return empty_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose empty allowlisted live retrieval.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--dialogue_ids",
        nargs="+",
        default=["913", "3457", "3437", "245", "4932"],
        help="Dialogues that logged no_allowlisted_results in eval (seed=42).",
    )
    parser.add_argument(
        "--also_known_queries",
        action="store_true",
        help="Re-run the 3 unique failing query strings even if not in dialogue build.",
    )
    parser.add_argument(
        "--indiacode_probes",
        action="store_true",
        help="Run indiacode-oriented probe queries for crime entities.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("LIVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("LIVE_SEARCH_API_KEY not set")

    import spacy

    nlp = spacy.load("en_core_web_sm")
    config = LiveRetrievalConfig.load()

    all_empty: dict[str, list[str]] = {}
    for did in args.dialogue_ids:
        empty = diagnose_dialogue(
            did, args.input, config, api_key, nlp, run_probes=args.indiacode_probes
        )
        all_empty[did] = empty

    if args.also_known_queries:
        _safe_print("\n" + "#" * 88)
        _safe_print("KNOWN EMPTY QUERIES (from eval log, standalone)")
        _safe_print("#" * 88)
        crime_hints = {"rape": "rape", "murder": "murder", "complaint": "complaint"}
        for qtext in KNOWN_EMPTY_QUERIES:
            hint = next((v for k, v in crime_hints.items() if k in qtext.lower()), None)
            diagnose_query(qtext, config, api_key, run_probes=args.indiacode_probes, crime_hint=hint)

    _safe_print("\n" + "=" * 88)
    _safe_print("SUMMARY")
    _safe_print("=" * 88)
    for did, queries in all_empty.items():
        if queries:
            _safe_print(f"  Dialogue {did}: empty allowlist for {queries!r}")
        else:
            _safe_print(
                f"  Dialogue {did}: all built queries returned allowlisted hits "
                "(may differ from eval if cache/budget changed)"
            )


if __name__ == "__main__":
    main()
