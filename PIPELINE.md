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
- Standard preprocessing (`scripts/preprocess_kare.py`) uses static KTC by default; live
  retrieval only runs when `run_hybrid(enable_live=True)` is used with keys present.

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
| 2a | `ktc/extraction.py` | spaCy dependency OpenIE (Stanford OpenIE-compatible interface) |
| 2b | `ktc/filtering.py` | Paper filtering rules (a–e) |
| 2c | `ktc/coreference.py` | Pronoun head resolution via spaCy noun phrases |
| 2d | `ktc/ranking.py` | Sentence-BERT cosine similarity, top **26** triplets |
| 2e | `ktc/verbalization.py` | **Template-based** verbalization (default) |

### Verbalization choice (Stage 2e)

The paper used few-shot GPT-J. This repo defaults to **template-based verbalization**
so preprocessing runs locally without API keys. For closer replication, run:

```bash
export OPENAI_API_KEY=...
python scripts/preprocess_kare.py --verbalization_backend llm
```

Template verbalization produces fluent single sentences rather than naive
`head + relation + tail` concatenation.

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
