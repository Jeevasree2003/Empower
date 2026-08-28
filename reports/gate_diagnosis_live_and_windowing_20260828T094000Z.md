# KTC gate diagnosis — passage windowing and live retrieval

Same 50 turns as `reports/gate_diagnosis_20260828T090809Z.md` (seed=42). Candidate cosine gate stays **0.38**. Production `DEFAULT_PASSAGE_TOP_N` stays **3**. Passage floor (`passage_min_score`) is now a separate pipeline field from the candidate gate (`min_cosine`); both default to 0.38.

Configs on the identical sample:

| config | passage `top_n` | passage floor | live | trace |
|---|---:|---:|---|---|
| baseline (existing report) | 3 | 0.38 | off | `reports/trace_20260828T090809Z.jsonl` |
| Part 1 windowing | **5** | 0.38 | off | `reports/trace_topn5_static_20260828T093614Z.jsonl` |
| Part 2 live | 3 | 0.38 | **on** | `reports/trace_live_topn3_20260828T093713Z.jsonl` |
| both | **5** | 0.38 | **on** | `reports/trace_live_topn5_20260828T093911Z.jsonl` |
| Part 3 floor | **5** | **0.15** | off | `reports/trace_floor015_topn5_static_20260828T095151Z.jsonl` |

## Side-by-side (identical 50 turns)

| metric | static top_n=3 floor=0.38 | static top_n=5 floor=0.38 | static top_n=5 floor=0.15 | static+live top_n=3 | live+top_n=5 |
|---|---:|---:|---:|---:|---:|
| **gate_passed mean** | **0.30** | **0.42** | **0.54** | **1.32** | **1.44** |
| turns with gate_passed > 0 | 8 | 8 | **13** | 18 | 18 |
| verbalized in `final_knowledge_sources` | 8 | 8 | 13 | 18 | 18 |
| supplemental_counseling only | 32 | 32 | 29 | 22 | 22 |
| **fully empty turns** | **10** | **10** | **8** | **10** | **10** |
| zero static candidates retrieved | 30 | 30 | **1** | 30 | 30 |
| no passage selected | 21 | 21 | **0** | 21 | 21 |
| passage but OpenIE empty | 9 | 9 | 1 | 9 | 9 |
| mean candidates in ranker pool | 1.74 | 2.68 | **7.78** | 7.02 | 7.96 |
| mean passages used | 1.32 | 1.78 | **4.86** | 1.32 | — |
| live_sentences (sum) | 0 | 0 | 0 | 583 | 583 |
| live_sentence_relevance (sum) | 0 | 0 | 0 | 247 | 247 |
| live OpenIE triplets (sum) | 0 | 0 | 0 | 176 | 176 |
| turns with any live candidate in pool | 0 | 0 | 0 | 29 | 29 |

## Part 1 — widening `top_n` 3 → 5 (static, floor still 0.38)

Window count alone **does not** shrink empty pools. Zero-candidate turns stay **30/50**; **21** still have no passage ≥ 0.38; **9** still select a passage and extract zero triplets. Empty-turn count stays **10**. Turns that already pass the candidate gate stay **8**.

What did change: when at least four windows already scored ≥ 0.38, the 4th/5th window is no longer dropped. Mean retrieved candidates **1.74 → 2.68**; mean `gate_passed` **0.30 → 0.42**. That is more triplets on turns that were already retrieving, not new turns entering the funnel.

The Kannada-couple blob is the cautionary example: dialogue **1178** turn **9** went from 3 passages / 6 candidates to **5 passages / 16 candidates**, including `a Kannada couple.(Rajesh Kamat and Suvarna Kamat…)` at cosine ~0.11 (still below the candidate gate). Widening `top_n` while keeping a 0.38 passage floor therefore cannot fix unrelated KARE knowledge attachments; it can add more noise from extra on-threshold windows.

**Do not make top_n=5 the production default yet.** The gain is modest and concentrated on turns that already had passing windows. The 21/30 “no passage ≥ 0.38” leak is a **floor** problem, not a **count** problem.

## Part 2 — live retrieval on the same 50 turns (top_n=3)

Live search did run (Tavily allowlist; many queries still returned `no_allowlisted_results`). Summarize backend remained extractive.

- Mean live_sentences / turn: 11.66 (583 total)
- Mean live_sentence_relevance / turn: 4.94 (247 total)
- Mean live OpenIE triplets / turn: 3.52 (176 total)
- Ranker pool mean: 1.74 → **7.02**
- `gate_passed` mean: 0.30 → **1.32**
- Turns with any verbalized KT: 8 → **18**

Of the original **30** zero-static-candidate turns:

| after live | count |
|---|---:|
| gained live candidates in the ranker pool | 16 |
| live funnel counts > 0 (sentences/relevance/OpenIE) | 17 |
| `final_knowledge_sources` nonempty | 21 (17 counseling-only + 4 verbalized) |
| newly verbalized (vs counseling/empty at baseline) | **4** |
| still fully empty | **9** (same empty dialogues as baseline except one of the 10 empty turns was not in the 30) |

Baseline split of those 30: 21 already had counseling-bank only, 9 were empty. Live flipped **4** of the 21 counseling-only turns to verbalized; it **did not** reduce the 10 fully empty turns.

Live candidates often still die at the **same 0.38 candidate gate** (iCALL pages, cybercrime.gov.in boilerplate, IPC 376 snippets scoring 0.02–0.37). So live helps coverage into the pool more than it helps survival through the gate.

## Empty after both (10 / 50 = 20%)

The empty-turn set is **identical** across baseline, top_n=5, live, and live+top_n=5:

`4908/3, 2202/2, 4216/6, 3211/5, 1382/0, 2451/3, 1545/7, 4466/5, 1978/2, 3249/4`

Not all 10 are “KARE blob is unrelated.” Split:

**Have retrieved candidates that fail the 0.38 candidate gate (plausible next fix: gate/ranking, not more windows):**

- 2202, 3211, 3249 — live candidates in the pool, counseling bank did not fire
- 2451 — top_n=5 selected 5 passages / 6 static triplets; all below 0.38 (anecdote-like dating story vs knowledge)

**No pool and no counseling (closest to “no plausible static+live fix without new queries/bank rules”): 6/50 = 12%**

- 1382 — ranking query `I am looking for help.`; no entities
- 4908, 1978 — NER/noun-chunk only, no lexicon crime term, no live hits
- 4216 — `youtube` lexicon, agent asking about video comments; no live hits
- 1545 — spaCy `XYZ` only; topless-contract turn
- 4466 — spaCy `Rajasthan` only; medical-aid turn

The recurring **Kannada couple** passage is **not** in this empty set; it is counseling-only (1178/9). It is evidence of mismatched KARE `knowledge` text, but the counseling bank still fills the turn.

## Recommendation

1. **Widening `top_n` alone does not materially help** the diagnosis target (empty pools / empty turns). Keep production `DEFAULT_PASSAGE_TOP_N=3`. If you change passage selection next, change the **floor** (`passage_min_score`, e.g. 0.15) rather than the count — that is the 21/30 leak. Measure that in a later pass; this run isolated window count only.

2. **Live retrieval alone does materially help gate_passed** (0.30 → 1.32) and verbalized turns (8 → 18), mostly by stuffing the ranker pool. It does **not** shrink fully empty turns (still 10/50). Many live sentences then fail the 0.38 candidate gate.

3. **After both (window + live, floor still 0.38), 10/50 turns (20%) are still empty.** About **4** of those have candidates sitting under 0.38 (threshold/ranking). About **6** (12%) have no static passage, no live candidate, and no counseling-bank hit.

## Part 3 — passage floor 0.38 → 0.15 (static, top_n=5, candidate gate still 0.38)

This is the missing variable from Part 1. Production `DEFAULT_PASSAGE_MIN_SCORE` remains **0.38**.

Of the original **21** “no passage ≥ 0.38” turns:

| | count |
|---|---:|
| now select a passage at floor=0.15 | **21 / 21** |
| of those, OpenIE puts candidates in the ranker pool | **21 / 21** |
| of those, at least one candidate clears the unchanged **0.38** candidate gate | **1 / 21** (dlg 3249 turn 4: `Someone keeps on a social media platform.` score 0.4157) |
| of those, pool is larger but everything fails 0.38 | **20 / 21** |

So loosening the floor **does** get windows into extraction. It does **not** get those windows through the candidate gate: 20/21 new pools are sub-0.38 noise (typical top scores 0.11–0.31). Dialogue 704 turn 6 now ranks the Kannada-couple passage first at **0.2777** — still below 0.38, same failure mode as predicted.

Empty turns 10 → **8** (3249 newly verbalized from the 21; 2451 was already in the “had candidates below 0.38” bucket and now passes with `Your lover is stalking you.` at 0.437). Mean pool 1.74 → **7.78** with only **+5** turns clearing the candidate gate (8 → 13), of which only **+1** came from the original 21 no-passage turns.

## Recommendation (go / no-go)

1. **`top_n=5`:** no-go for production (Part 1 unchanged).
2. **`passage_min_score=0.15`:** **no-go for production.** It fills empty pools (30 → 1 zero-candidate turns) but almost entirely with candidates that fail the 0.38 gate (20/21 of the original no-passage turns). Empty turns only drop 10 → 8. That is the Kannada-couple pattern at scale: more retrieval, same gate, more noise.
3. **Live retrieval** remains the only intervention that materially raises `gate_passed` mean (0.30 → 1.32) and verbalized turns (8 → 18). It still leaves 10 empty turns.
4. Promoting 0.15 would couple poorly with live (even larger pools of sub-gate sentences). If the candidate gate is ever lowered, re-measure the floor then; do not ship 0.15 while the candidate gate stays 0.38.

Production defaults stay `passage_top_n=3`, `passage_min_score=0.38`, `min_cosine=0.38`.
