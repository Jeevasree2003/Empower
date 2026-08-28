# KTC gate diagnosis

Sample: 50 turns, seed=42, static pipeline (no live retrieval).
Candidate gate: per-candidate cosine >= 0.38, then `is_ktc_usable`, then substring dedup.
Funnel `gate_passed` is the count after those filters (mean 0.30 in this sample, vs 0.118 on the 500-dialogue funnel).

Instrumentation monkeypatched `ktc.pipeline.rank_candidates` in-process only. Production gate thresholds, ranking, and extraction were not changed.

## Cross-tab (step 2)

```
group                         n  gate_pass%  mean_retr  mean_ranked  n_pass
--------------------------------------------------------------------------
lexicon_only                  9        22.2      1.444        1.444       2
ner_or_chunk_no_lexicon      15        13.3      1.533        1.533       2
history_fallback             16         6.2      1.688        1.688       1
no_history_fallback          34        20.6      1.765        1.765       7
```

Additional entity mix in the same 50 turns (not requested, for context): 25 `lexicon_plus_ner_or_chunk`, 1 `no_entities`. Mean retrieved candidates is ~1.4–1.8 in every bucket — the pool is tiny before the candidate gate even runs.

## NER/noun-chunk vs lexicon

Lexicon-only gate pass rate is **22.2%** (n=9); NER/noun-chunk-with-no-lexicon is **13.3%** (n=15). Weak-source turns pass less often, which is consistent with a confidence-aware threshold *if* the pool were large. In this sample both groups retrieve about the same number of candidates (~1.5), so NER is not flooding the ranker with junk so much as still failing a near-empty pool.

History fallback is worse (6.2% vs 20.6% without it). That is expected: fallback fires when the last two victim turns have no confident entities, i.e. the ranking query is already weak.

**Extraction coverage is not the main reason `gate_passed` mean is 0.118.** Widening NER/noun chunks did not create a large ranked pool.

## Empty / counseling-only: retrieval vs gate

This is the distinction that matters.

| slice | n | zero candidates retrieved | of those: no passage ≥ 0.38 | of those: passage but OpenIE empty | retrieved but failed candidate gate | gate passed |
|---|---:|---:|---:|---:|---:|---:|
| main sample (50) | 50 | 30 (60%) | 21 | 9 | 12 (24%) | 8 |
| supplemental_counseling-only | 20 | 13 (65%) | — | — | 7 (35%) | 0 |
| fully empty | 20 | 16 (80%) | — | — | 4 (20%) | 0 |

Among the 12 sample turns that *did* retrieve candidates and still failed, every top-ranked reject reason was `cosine < 0.38` (typical top scores ~0.13–0.34). One extra empty-bucket turn failed *after* clearing 0.38 because `is_ktc_usable` tagged the top hit as an anecdote (`the in-laws for mental harassment...`).

Counseling-only examples with a non-empty pool still die at the candidate cosine gate (e.g. dlg 1178 top=0.2995; dlg 4248 top=0.3189 `fraudulent impersonation is a statutory offence`). Empty examples are almost all `retr=0` / `passages=0` even when entities exist (`whatsapp` lexicon, `Rakshak` spaCy PERSON).

**Primary bottleneck: retrieval coverage**, specifically the *passage* cosine gate (min_cosine=0.38 against the last-1–2 victim turns) and empty OpenIE on the few passages that survive. 21/30 empty pools never selected a knowledge passage; the other 9 selected passages but extracted zero triplets.

The candidate-level 0.38 gate is a secondary killer (12/50 turns). Lowering *only* that gate cannot fix the 60% of turns whose ranker pool is already empty.

## Recommendation

**Fix retrieval coverage first** (passage selection / OpenIE yield), not the candidate score threshold:

1. Passage cosine 0.38 against a short victim span is dropping the knowledge blob before OpenIE runs. That is the same numeric threshold as the candidate gate, applied one stage earlier, and it explains most `no_passages_used` / counseling-bank-only turns.
2. When a passage does survive, OpenIE often returns 0 triplets (9/30 empty pools). Extraction yield on selected windows is the next leak.
3. Only after a non-empty pool is typical should the candidate gate (0.38) or a source-aware (lexicon vs NER) threshold be tuned. NER/noun-chunk turns do pass the candidate gate somewhat less often, but they are not the reason mean `gate_passed` is 0.118.

Do not lower the candidate gate in isolation: it would not help turns with `candidates_retrieved=0`.
