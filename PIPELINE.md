# EMPOWER-KARE Pipeline

This document mirrors the expected project pipeline for reproducing
**EMPOWER-KARE: Deep Prompt Learning for Knowledge-Aware Response Generation in
Clinical Counseling and Legal Support Conversations** (IEEE TAI, 2026).

## Setup

For **hybrid live knowledge retrieval** (optional API calls in `ktc/live_retrieval.py` and
`ktc/live_summarize.py`), copy the example env file and add your keys locally:

```bash
cp .env.example .env
# Edit .env and replace placeholders with real keys
```

- `.env` is listed in `.gitignore` and is **never committed** — safe for secrets.
- Keys are read via `os.environ` (`LIVE_SEARCH_API_KEY`, `LLM_API_KEY`, optional `LLM_API_BASE`).
- Summarization uses an **OpenAI-compatible client** pointed at Groq by default
  (`llm_api_base: https://api.groq.com/openai/v1`, `llm_model: llama-3.3-70b-versatile` in
  `config/live_retrieval_config.yaml`). Set `LLM_API_KEY` to your Groq API key.
- Entry scripts call `load_dotenv()` **before** importing `ktc` live modules, so the
  environment is populated when `LiveRetrievalConfig.load()` or `search_allowlisted()` run.
- Standard preprocessing (`scripts/preprocess_kare.py`) uses static KTC with LLM
  verbalization by default; live retrieval only runs when `run_hybrid(enable_live=True)`
  is used with keys present.

Install dependencies (includes `python-dotenv`):

```bash
pip install -r requirements.txt
```

## Dataset

- **Full dataset:** https://www.iitp.ac.in/~ai-nlp-ml/resources/data/KARE.zip
- **Local copy:** place `KARE.jsonl` at `KARE-data/KARE/Data/KARE.jsonl` (5000 dialogues)
- **Sample only:** `dataset/KARE-Sample.json` (not sufficient for training)

Each dialogue record contains:

```json
{
  "dialogue_id": "...",
  "utterances": [{"utterance_no": "0", "author_role": "bot|user", "utterance": "..."}],
  "knowledge": "raw domain knowledge text ..."
}
```

## Pipeline overview

| Stage | Script / module | Output |
|-------|-----------------|--------|
| 0 | fixes in `EMPOWER-MODEL/` | runnable GPT-2 path |
| 1 | raw `KARE.jsonl` | unchanged |
| 2 | `ktc/` | verbalized knowledge per turn |
| 3 | `scripts/preprocess_kare.py` | `train.json`, `valid.json`, `test.json` |
| 4 | `scripts/kdpt.sh` | `best_pt1` checkpoint |
| 5 | `scripts/rdpt.sh` | `best_pt2` + DDKM checkpoint |
| 6 | `scripts/gen.sh` | generated test responses |
| 7 | `EMPOWER-MODEL/automatic_evaluation.py` | BLEU, ROUGE-L, F1, KF1 |

## Stage 0 fixes

- **BART removed:** `module.py` now supports GPT-2 only (`gpt2`, etc.). The missing
  `prefixBart.py` is no longer imported.
- **ParlAI removed:** `automatic_evaluation.py` uses `rouge_score` and `metrics.py`
  instead of `parlai`.

## Stage 2 — Knowledge Triplets Construction (KTC)

Implemented in `ktc/` with five testable sub-modules:

| Step | Module | Method |
|------|--------|--------|
| 2a | `ktc/extraction.py` | spaCy dependency OpenIE — **all clause verbs**, not ROOT-only |
| 2b | `ktc/filtering.py` | Paper filtering rules (a–e) |
| 2c | `ktc/coreference.py` | **coreferee** (default) or heuristic pronoun resolution |
| 2d | `ktc/ranking.py` | Sentence-BERT cosine similarity, top **26** triplets |
| 2e | `ktc/verbalization.py` | **LLM few-shot verbalization** (default; Groq via `LLM_API_KEY`) |

### Verbalization choice (Stage 2e)

The paper used few-shot GPT-J. This repo now defaults to **LLM verbalization** via the
same OpenAI-compatible Groq endpoint as live summarization (`LLM_API_KEY`,
`config/live_retrieval_config.yaml` → `llama-3.3-70b-versatile`). If no API key is
present, verbalization **falls back to template** with a warning so unit tests and
offline smoke runs still work.

For fully offline preprocessing:

```bash
python scripts/preprocess_kare.py --verbalization_backend template --coref_backend heuristic
```

### Extraction coverage (Stage 2a)

OpenIE runs on **every clause verb** in a sentence (relative clauses, coordinated
verbs, subordinate clauses), not only the ROOT verb. Auxiliary tokens (`can`, `was`)
attached to another verb are skipped to avoid empty duplicate passes.

### Coreference (Stage 2c)

Default backend is **`model`** (spaCy **coreferee**). Install with
`pip install coreferee` and `python -m coreferee install en`. Use
`--coref_backend heuristic` when coreferee is unavailable.

### Live retrieval query policy (Stage 0.5)

Decisions for `ktc/query_builder.py` — documented before hardcoding crime→section
mappings or changing legal templates.

#### IPC vs BNS section numbering

**Decision: use IPC section numbers for `crime_statute_indiacode` retrieval queries.**

India's Bharatiya Nyaya Sanhita (BNS) replaced most IPC offences on **1 July 2024**
(BNSS replaced CrPC for procedure). BNS is the operative code for new cases.

We still target **IPC** in search queries because:

1. **Empirical retrieval** — empty-retrieval diagnosis (Aug 2026) showed Tavily +
   `indiacode.nic.in` reliably returns allowlisted hits for queries like
   `IPC Section 376 indiacode.nic.in` and `IPC Section 302 indiacode.nic.in`; BNS
   section probes were not equivalently indexed at the time of testing.
2. **India Code URL structure** — penal statute pages on indiacode.nic.in still use IPC
   act IDs and section ordinals in `show-data` URLs (e.g. Sections 375/376, 302).
3. **Counselor-facing use** — retrieved sentences support victim counseling, not court
   filings; IPC labels remain widely recognized in training data and public discourse.

**Future:** add parallel `crime_statute_bns` templates once indiacode BNS pages are
verified to rank in Tavily top results (e.g. BNS §63 rape, §103 murder). Do not mix
IPC and BNS section numbers in a single query string.

| Offence (canonical) | IPC (retrieval) | BNS (in force; not used in queries yet) |
|---------------------|-----------------|----------------------------------------|
| rape                | 376             | 63                                     |
| murder              | 302             | 103                                    |
| criminal intimidation | 506           | 351                                    |

#### Legal entity `complaint` (dialogue 245)

**Decision: target CrPC Section 154 (FIR / information to police), not a generic
definitional query.**

Dialogue 245 victim text: *"I need help to lodge a complaint"* — procedural intent to
file with police, not *"what is a legal complaint?"* as an abstract concept. The
failed `legal_general` template surfaced consumer-court blogs and indiankanoon, not
government procedure.

Use template `legal_fir_procedure` with CrPC §154 / police-station wording for
`complaint` (and close variants). BNSS §173 is the BNS-era analogue; same IPC-style
retrieval rationale applies — CrPC §154 for indiacode hooks until BNSS indexing is
verified.

`crime_report_india` is **kept alongside** `crime_statute_indiacode` (supplement, not
replacement) so eval scripts can compare both templates.

Run KTC tests:

```bash
python -m unittest ktc.test_ktc
```

## Stage 3 — Preprocessed dataset

```bash
cd EMPOWER-KARE
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python scripts/preprocess_kare.py \
  --input ../../KARE-data/KARE/Data/KARE.jsonl \
  --output_dir data/preprocessed
```

This writes one JSON object per line into `train.json`, `valid.json`, and
`test.json` using the paper's **4000 / 500 / 500** dialogue split.

Each turn record:

```json
{
  "history": ["victim: ...", "agent: ..."],
  "knowledge": ["verbalized text __knowledge__ eval text"],
  "response": "gold agent response"
}
```

Ablation (**EMPOWER − KTC**):

```bash
python scripts/preprocess_kare.py --knowledge_mode raw --output_dir data/preprocessed_raw
```

Smoke test on a subset:

```bash
python scripts/preprocess_kare.py --max_dialogues 10 --output_dir data/smoke
```

## Stages 4–7 — Training and evaluation

Point the shell scripts at the preprocessed directory:

```bash
# Stage 4 — KDPL
bash scripts/kdpt.sh   # set --data_dir to data/preprocessed

# Stage 5 — RDPL + DDKM
bash scripts/rdpt.sh   # set --pfxKlgModel_name_or_path to best_pt1

# Stage 6 — Inference
bash scripts/gen.sh    # beam=5, min_length=20 (configure in finetune.py hparams)

# Stage 7 — Evaluation
python EMPOWER-MODEL/automatic_evaluation.py \
  --pred_file output/preds.txt \
  --test_data data/preprocessed/test.json \
  --eval_metric kf1
```

## Backbone

The paper uses **GPT-2 medium**. The training code expects `--model_name_or_path gpt2`
(or another GPT-2 checkpoint). LLaMA support would require porting DDKM embedding
lookups (`model.transformer.wte.weight` → `model.model.embed_tokens.weight`) and
using Hugging Face PEFT prefix tuning separately.
