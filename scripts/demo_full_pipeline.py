"""
EMPOWER-KARE — Knowledge Triplets Construction (KTC) full pipeline demo.

Prints the complete output at every stage:
  2a  OpenIE-style extraction (raw triplets)
  2c  Coreference resolution (pronoun heads resolved)
  2b  Rule-based filtering (paper filtering rules a-e)
  2d  SBERT ranking (top-K candidates, static + optional live, cosine similarity)
  2e  Verbalization (template or LLM few-shot, paper-style GPT-J substitute)

Usage:
    python scripts\\demo_full_pipeline.py
    python scripts\\demo_full_pipeline.py --dialogue-index 3 --enable-live
    python scripts\\demo_full_pipeline.py --verbalization template
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from ktc.cleaning import clean_knowledge_text
from ktc.extraction import extract_triplets
from ktc.coreference import resolve_coreferences
from ktc.filtering import filter_triplets
from ktc.pipeline import KnowledgeTripletPipeline
from ktc.entity_extraction import extract_entities
from ktc.query_builder import build_queries
from ktc.live_knowledge import fetch_live_knowledge, victim_utterance_from_history
from ktc.live_config import LiveRetrievalConfig, ApiCallBudget
import spacy

DATA_FILE = Path(__file__).resolve().parents[1] / "dataset" / "KARE-Sample.json"
ROLE_MAP = {"user": "victim", "bot": "agent"}


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def build_history(dialogue: dict, max_turns: int) -> str:
    utterances = dialogue["utterances"][:max_turns]
    return " ".join(
        f"{ROLE_MAP.get(u['author_role'], u['author_role'])}: {u['utterance']}"
        for u in utterances
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and print every KTC stage for one KARE dialogue.")
    parser.add_argument("--input", type=Path, default=DATA_FILE, help="Path to KARE JSONL/sample file")
    parser.add_argument("--dialogue-index", type=int, default=0, help="Index into the file (0-based)")
    parser.add_argument("--max-turns", type=int, default=10, help="How many dialogue turns to use as history")
    parser.add_argument("--top-k", type=int, default=26, help="Top-K triplets to keep after ranking (paper default: 26)")
    parser.add_argument(
        "--verbalization", choices=["template", "llm"], default="llm",
        help="template = offline mechanical sentences; llm = Groq few-shot (paper used GPT-J), falls back to template if LLM_API_KEY missing",
    )
    parser.add_argument("--enable-live", action="store_true", help="Also fetch live web knowledge (needs LIVE_SEARCH_API_KEY)")
    parser.add_argument("--show-limit", type=int, default=30, help="Max items to print per stage (avoid flooding terminal)")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        dialogues = [json.loads(l) for l in f if l.strip()]

    if args.dialogue_index >= len(dialogues):
        raise SystemExit(f"Only {len(dialogues)} dialogues in file; index {args.dialogue_index} out of range")

    dialogue = dialogues[args.dialogue_index]
    knowledge_text = dialogue["knowledge"]
    history = build_history(dialogue, args.max_turns)

    hr("INPUT")
    print(f"dialogue_id      : {dialogue['dialogue_id']}")
    print(f"dialogue history : {history[:400]}{' ...' if len(history) > 400 else ''}")
    print(f"raw knowledge len: {len(knowledge_text)} chars")
    print(f"raw knowledge    : {knowledge_text[:300]}{' ...' if len(knowledge_text) > 300 else ''}")

    nlp = spacy.load("en_core_web_sm")

    # ---------------- Stage: cleaning (pre-2a) ----------------
    hr("STAGE 2.pre — CLEANING")
    cleaned = clean_knowledge_text(knowledge_text)
    print(f"cleaned length: {len(cleaned)} chars")
    print(f"cleaned preview: {cleaned[:400]}{' ...' if len(cleaned) > 400 else ''}")

    # ---------------- Stage 2a: OpenIE extraction ----------------
    hr("STAGE 2a — OPENIE-STYLE TRIPLET EXTRACTION (spaCy dependency parse)")
    raw_triplets = extract_triplets(cleaned, backend="spacy", nlp=nlp)
    print(f"raw triplets extracted: {len(raw_triplets)}")
    for t in raw_triplets[: args.show_limit]:
        print(f"  ({t.head!r}, {t.relation!r}, {t.tail!r})")
    if len(raw_triplets) > args.show_limit:
        print(f"  ... ({len(raw_triplets) - args.show_limit} more)")

    # ---------------- Stage 2c: coreference resolution ----------------
    hr("STAGE 2c — COREFERENCE RESOLUTION (heuristic pronoun-head resolution)")
    resolved = resolve_coreferences(raw_triplets, cleaned, nlp=nlp, backend="heuristic")
    changed = [
        (r_orig, r_new)
        for r_orig, r_new in zip(raw_triplets, resolved)
        if r_orig.head != r_new.head
    ]
    print(f"triplets with head resolved from a pronoun: {len(changed)}")
    for before, after in changed[: args.show_limit]:
        print(f"  {before.head!r}  ->  {after.head!r}   (relation={after.relation!r}, tail={after.tail!r})")
    if not changed:
        print("  (no pronoun-headed triplets found in this passage)")

    # ---------------- Stage 2b: rule-based filtering ----------------
    hr("STAGE 2b — RULE-BASED FILTERING (paper rules a-e + extras)")
    filtered = filter_triplets(resolved)
    print(f"filtered triplets: {len(filtered)} / {len(resolved)} survived "
          f"({len(resolved) - len(filtered)} removed, "
          f"{100 * (len(resolved) - len(filtered)) / max(len(resolved), 1):.0f}% reduction)")
    for t in filtered[: args.show_limit]:
        print(f"  ({t.head!r}, {t.relation!r}, {t.tail!r})")
    if len(filtered) > args.show_limit:
        print(f"  ... ({len(filtered) - args.show_limit} more)")

    # ---------------- Stage 0-0.9: live retrieval detail (entities -> queries -> search -> summarize) ----------------
    if args.enable_live:
        hr("STAGE 0 — VICTIM UTTERANCE EXTRACTED FROM HISTORY (input to live retrieval)")
        victim_text = victim_utterance_from_history(history)
        print(f"victim_utterance: {victim_text!r}")
        if not victim_text:
            print("  WARNING: empty victim utterance -> no entities -> no queries -> no live results.")
            print("  History must contain 'victim:' or 'user:' prefixed turns for this to work.")

        hr("STAGE 0.5 — ENTITY EXTRACTION (from victim utterance)")
        entities = extract_entities(victim_text, nlp=nlp)
        print(f"entities found: {len(entities)}")
        for e in entities:
            print(f"  {e}")

        hr("STAGE 0.6 — QUERY BUILDING (entities -> search query templates)")
        live_config = LiveRetrievalConfig.load()
        queries = build_queries(entities, max_queries=live_config.max_live_queries_per_dialogue)
        print(f"queries built: {len(queries)}")
        for q in queries:
            print(f"  {q}")

        hr("STAGE 0.75-0.9 — LIVE SEARCH + SUMMARIZATION (raw retrieved knowledge per query)")
        budget = ApiCallBudget(live_config.max_api_calls_per_run)
        live_candidates, live_queries, live_sentences = fetch_live_knowledge(
            victim_text, live_config, budget, nlp=nlp
        )
        print(f"live sentences retrieved: {len(live_sentences)}")
        for s in live_sentences[: args.show_limit]:
            print(f"  query={s.query!r}")
            print(f"    source: {s.source_url}")
            print(f"    sentence: {s.sentence}")
        if not live_sentences:
            print("  No live sentences retrieved for any query.")
            print("  Check: (1) LIVE_SEARCH_API_KEY set, (2) queries above are non-empty,")
            print("  (3) config/trusted_domains.yaml covers relevant sites for these queries,")
            print("  (4) data/live_cache/ for any cached-but-empty results within the TTL window.")

    # ---------------- Stage 2d + 2e: ranking + verbalization (full pipeline) ----------------
    hr(f"STAGE 2d — SBERT RANKING (top-{args.top_k} by cosine similarity to dialogue history)")
    pipeline = KnowledgeTripletPipeline(
        top_k=args.top_k,
        coref_backend="heuristic",
        verbalization_backend=args.verbalization,
    )
    result = pipeline.inspect(knowledge_text, history, enable_live=args.enable_live)

    ranked = result["ranked_candidates"]
    static_n = sum(1 for c in ranked if c.get("source") == "static_dataset")
    live_n = sum(1 for c in ranked if c.get("source") == "live_api")
    print(f"live retrieval enabled : {result['live_retrieval_enabled']}")
    print(f"top-ranked candidates  : {len(ranked)}  (static={static_n}, live={live_n})")
    print(f"top-1 similarity score : {result['top1_similarity_score']:.4f}")
    for c in ranked[: args.show_limit]:
        tag = "[LIVE]" if c.get("source") == "live_api" else "[static]"
        print(f"  {tag} {c.get('text', '')[:120]}")
    if len(ranked) > args.show_limit:
        print(f"  ... ({len(ranked) - args.show_limit} more)")

    hr(f"STAGE 2e — VERBALIZATION (backend={args.verbalization})")
    verbalized = result["verbalized"]
    empty_count = sum(1 for s in verbalized if not s.strip())
    print(f"verbalized sentences: {len(verbalized)}  (empty: {empty_count})")
    if empty_count:
        print(f"  WARNING: {empty_count} sentence(s) came back empty from the '{args.verbalization}' backend.")
        print("  Run scripts\\debug_llm.py to see the raw API response, or re-run with --verbalization template.")
    for i, (c, s) in enumerate(zip(ranked, verbalized), 1):
        tag = "[LIVE]" if c.get("source") == "live_api" else "[static]"
        shown = s if s.strip() else "<<EMPTY>>"
        print(f"  {i:2d}. {tag} {shown}")

    hr("FINAL KNOWLEDGE-ATTRIBUTED PROMPT INPUT (what feeds into KDPL, Section IV-B2)")
    print(" ".join(verbalized))

    hr("SUMMARY")
    print(f"raw_triplets      : {len(raw_triplets)}")
    print(f"after_coreference : {len(resolved)}  ({len(changed)} pronoun heads resolved)")
    print(f"after_filtering   : {len(filtered)}  ({100 * (len(resolved) - len(filtered)) / max(len(resolved), 1):.0f}% removed)")
    print(f"top_k_ranked      : {len(ranked)}  (static={static_n}, live={live_n})")
    print(f"final_verbalized  : {len(verbalized)}")


if __name__ == "__main__":
    main()