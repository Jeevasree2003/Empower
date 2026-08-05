#!/usr/bin/env python
"""Investigate coref, relevance, and ranking issues."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ktc.cleaning import clean_knowledge_text
from ktc.coreference import _collect_noun_phrases
from ktc.pipeline import KnowledgeTripletPipeline
from ktc.ranking import SentenceBertRanker
from ktc.triplet import Triplet

DATA = Path(r"C:\Users\pg\Downloads\pro\KARE-data\KARE\Data\KARE.jsonl")


def load(did: str) -> dict:
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r["dialogue_id"]) == did:
                return r
    raise KeyError(did)


def history_at(did: str, turn: int = 0) -> str:
    r = load(did)
    utts = sorted(r["utterances"], key=lambda u: int(u["utterance_no"]))
    history = []
    bt = 0
    for u in utts:
        role = u["author_role"]
        text = f"{role}: {u['utterance'].strip()}"
        if role in {"bot", "agent"} and history:
            if bt == turn:
                return " ".join(history)
            bt += 1
        history.append(text)
    return ""


def main():
    import spacy

    nlp = spacy.load("en_core_web_sm")

    print("=" * 72)
    print("ISSUE 1: COREFERENCE — what noun phrase is chosen?")
    print("=" * 72)
    for did in ["100", "3000", "4500"]:
        r = load(did)
        cleaned = clean_knowledge_text(r.get("knowledge", "") or "")
        phrases = _collect_noun_phrases(cleaned, nlp)
        last = phrases[-1] if phrases else "(none)"
        print(f"\nDialogue {did}: resolves ALL pronouns to phrases[-1] = {last!r}")
        print(f"  Total noun phrases collected: {len(phrases)}")
        print(f"  Last 5 phrases: {phrases[-5:]}")

    print("\n" + "=" * 72)
    print("ISSUE 1b: Raw pre-coref triplets that became bad substitutions")
    print("=" * 72)
    from ktc.extraction import extract_triplets
    from ktc.coreference import resolve_coreferences

    examples = {
        "100": [("it", "is", "best"), ("They", "will tell", "you"), ("he", "will be", "able")],
        "3000": [("It", "is", "the small step"), ("they", "will ask", "you"), ("it", "is", "true love")],
        "4500": [("it", "becomes", "easy to locate"), ("It", "is", "the basic"), ("She", "reported", "the incident")],
    }
    for did, _ in examples.items():
        r = load(did)
        cleaned = clean_knowledge_text(r.get("knowledge", "") or "")
        raw = extract_triplets(cleaned, nlp=nlp)
        resolved = resolve_coreferences(raw, cleaned, nlp=nlp)
        print(f"\n--- Dialogue {did} ---")
        shown = 0
        for raw_t, res_t in zip(raw, resolved):
            if raw_t.head != res_t.head and raw_t.head.strip().split()[0].lower() in {
                "it", "they", "he", "she", "we", "i", "you"
            }:
                print(f"  BEFORE: ({raw_t.head}) | {raw_t.relation} | ({raw_t.tail[:60]}...)")
                print(f"  AFTER:  ({res_t.head}) | {res_t.relation} | ({res_t.tail[:60]}...)")
                print()
                shown += 1
                if shown >= 3:
                    break

    print("=" * 72)
    print("ISSUE 2: RAW KNOWLEDGE — first 500 chars")
    print("=" * 72)
    for did in ["3000", "4500"]:
        k = load(did).get("knowledge", "") or ""
        print(f"\n--- Dialogue {did} (len={len(k)}) ---")
        print(k[:500])
        lower = k.lower()
        terms = ["domestic", "husband", "murder", "violence", "dowry", "protection order", "kill me"]
        print("\nKeyword scan:")
        for term in terms:
            print(f"  {term!r}: {'FOUND at ' + str(lower.find(term)) if term in lower else 'not found'}")

    print("\n" + "=" * 72)
    print("ISSUE 2b: Dialogue 3000 — context around 'kill' in raw knowledge")
    print("=" * 72)
    k3000 = load("3000").get("knowledge", "") or ""
    idx = k3000.lower().find("kill")
    if idx >= 0:
        print(k3000[max(0, idx - 120) : idx + 200])

    print("\n" + "=" * 72)
    print('ISSUE 3: SBERT scores for "My dad beats..." triplet')
    print("=" * 72)
    target = Triplet("My dad", "beats", "me and my sister like wild animals")
    ranker = SentenceBertRanker()
    target_emb = ranker.model.encode(target.as_text(), normalize_embeddings=True)

    labels = [
        ("dialogue 1 turn 0", history_at("1", 0)),
        ("dialogue 1 turn 8", history_at("1", 8)),
        ("dialogue 100 turn 0", history_at("100", 0)),
        ("dialogue 1000 turn 0", history_at("1000", 0)),
        ("dialogue 3000 turn 0", history_at("3000", 0)),
        ("unrelated (taxes)", "victim: I need help with my taxes and accounting."),
    ]
    print("\nDirect cosine similarity (triplet vs history):")
    for label, hist in labels:
        emb = ranker.model.encode(hist, normalize_embeddings=True)
        score = float(np.dot(target_emb, emb))
        print(f"  {label:22s}  {score:.4f}")

    pipeline = KnowledgeTripletPipeline()
    print("\nCalibration — all filtered triplet scores for dialogue 3000 turn 0:")
    filtered = pipeline.get_filtered_triplets(load("3000").get("knowledge", "") or "")
    hist = history_at("3000", 0)
    hist_emb = ranker.model.encode(hist, normalize_embeddings=True)
    scores = []
    for t in filtered:
        emb = ranker.model.encode(t.as_text(), normalize_embeddings=True)
        scores.append((float(np.dot(emb, hist_emb)), t))
    scores.sort(reverse=True)
    for sc, t in scores[:10]:
        mark = " <-- MY DAD" if t.head == "My dad" else ""
        print(f"  {sc:.4f}  ({t.head}) | {t.relation} | {t.tail[:45]}...{mark}")

    print("\nHow many knowledge blobs contain 'My dad beats'?")
    count = 0
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            if "My dad beats" in json.loads(line).get("knowledge", ""):
                count += 1
    print(f"  {count} / 4999 dialogues")


if __name__ == "__main__":
    main()
