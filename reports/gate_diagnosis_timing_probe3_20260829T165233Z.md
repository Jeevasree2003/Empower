# KTC gate diagnosis

Sample: 3 turns, seed=42, static pipeline (no live retrieval).
Candidate gate: per-candidate cosine >= 0.38, then `is_ktc_usable`, then substring dedup.
Funnel `gate_passed` is the count after those filters.

## Cross-tab (step 2)

```
group                         n  gate_pass%  mean_retr  mean_ranked  n_pass
--------------------------------------------------------------------------
lexicon_only                  0         n/a      0.000        0.000       0
ner_or_chunk_no_lexicon       3        33.3      2.333        2.333       1
history_fallback              2        50.0      3.500        3.500       1
no_history_fallback           1         0.0      0.000        0.000       0
```

## NER/noun-chunk vs lexicon

Not enough turns in one of the entity-source buckets to compare NER/noun-chunk vs lexicon gate pass rates.

## Empty / counseling-only: retrieval vs gate

main sample: n=3; zero retrieved=1 (33% [no passage survived 0.38=0, passage but OpenIE empty=1]); retrieved but failed cosine/usability gate=1 (33%); gate passed=1
supplemental_counseling-only (20): n=20; zero retrieved=4 (20% [no passage survived 0.38=2, passage but OpenIE empty=2]); retrieved but failed cosine/usability gate=16 (80%); gate passed=0
fully empty (20): n=20; zero retrieved=7 (35% [no passage survived 0.38=4, passage but OpenIE empty=3]); retrieved but failed cosine/usability gate=13 (65%); gate passed=0

**Primary bottleneck: threshold/ranking.**

## Recommendation

Fix the candidate gate/ranking first: most counseling-only/empty turns do retrieve candidates, but they fail min_cosine=0.38 or the post-gate usability filter.

## Notes

- `candidates_retrieved` is the static OpenIE pool (`static_candidates_from_triplets`) before ranking.
  An empty pool usually means no knowledge passage cleared the *passage* cosine gate (also 0.38)
  or OpenIE extracted zero triplets from the passages that did.
- Ranker scores every retrieved candidate; `candidates_after_ranking` equals the pool size.
- Counseling-bank facts are injected *after* the gate and are not in `candidates_retrieved`.
