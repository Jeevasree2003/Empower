"""
evaluate_all_metrics.py

Computes every metric reported in Table III of the EMPOWER-KARE paper:
    PPL, BLEU-4, Avg. BLEU, F1, Knowledge-F1, BERTScore-F1,
    Embedding Average (EA), Vector Extrema (VE), Greedy Matching (GM)

Run from EMPOWER-MODEL/, with your venv active.

--------------------------------------------------------------------
ONE-TIME SETUP (run once):
--------------------------------------------------------------------
    pip install bert-score --break-system-packages
    pip install spacy --break-system-packages
    python -m spacy download en_core_web_md

--------------------------------------------------------------------
USAGE:
--------------------------------------------------------------------
    # Text-based metrics only (fast, uses an existing preds.txt):
    python evaluate_all_metrics.py --pred_file output_gen_v1/preds.txt --test_data ../data/preprocessed_v2/test.json

    # Everything, including PPL (slower — reloads the model, runs over the test set):
    python evaluate_all_metrics.py --pred_file output_gen_v1/preds.txt --test_data ../data/preprocessed_v2/test.json --compute_ppl

    # PPL on a quick subset first, to sanity check before committing to a full run:
    python evaluate_all_metrics.py --pred_file output_gen_v1/preds.txt --test_data ../data/preprocessed_v2/test.json --compute_ppl --ppl_n_examples 300

Note on EA/VE/GM: the paper computes these with Word2Vec embeddings via the
nlg-eval toolkit. This script uses spaCy's en_core_web_md vectors (300d,
GloVe-derived) instead, since that avoids a ~1.5GB Word2Vec download. Values
are the same *kind* of metric but are not numerically identical to the paper's
toolkit — say so if you quote these numbers, don't present them as a literal
match to Table III's EA/VE/GM columns.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

KNOWLEDGE_SEP = "__knowledge__"
NO_PASSAGE_USED = "no_passages_used"


# --------------------------------------------------------------------------
# Loading preds.txt / test_data (same format your automatic_evaluation.py uses)
# --------------------------------------------------------------------------

def load_preds(pred_file: str):
    hyps, refs = [], []
    with open(pred_file, mode="r", encoding="utf-8") as rf:
        for line in rf.readlines():
            parts = line.split("|||")
            if len(parts) != 2:
                continue
            hyps.append(parts[0].strip())
            refs.append(parts[1].strip())
    return hyps, refs


def load_knowledge(test_data: str):
    knowledge_list = []
    with open(test_data, mode="r", encoding="utf-8") as rf:
        for line in rf:
            record = json.loads(line)
            knowledge_field = record["knowledge"][0]
            if KNOWLEDGE_SEP in knowledge_field:
                knowledge = knowledge_field.split(KNOWLEDGE_SEP, 1)[1].strip()
            else:
                knowledge = knowledge_field.strip()
            knowledge_list.append(knowledge)
    return knowledge_list


# --------------------------------------------------------------------------
# F1, Knowledge-F1, ROUGE-L, BLEU  (reusing your existing metrics.py)
# --------------------------------------------------------------------------

def compute_text_metrics(hyps, refs, knowledge_list):
    from metrics import bleu_metric, f1_metric
    from rouge_score import rouge_scorer

    results = {}

    results["F1"] = float(f1_metric(hyps, refs))

    if knowledge_list is not None and len(knowledge_list) == len(hyps):
        hyp_kf1, know_kf1 = [], []
        for hyp, know in zip(hyps, knowledge_list):
            if know != NO_PASSAGE_USED:
                hyp_kf1.append(hyp)
                know_kf1.append(know)
        results["Knowledge-F1"] = float(f1_metric(hyp_kf1, know_kf1)) if hyp_kf1 else float("nan")
    else:
        results["Knowledge-F1"] = float("nan")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl_scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hyps, refs)]
    results["ROUGE-L"] = float(np.mean(rl_scores))

    b1, b2, b3, b4 = bleu_metric(hyps, refs)
    results["BLEU-1"] = b1 * 100
    results["BLEU-2"] = b2 * 100
    results["BLEU-3"] = b3 * 100
    results["BLEU-4"] = b4 * 100
    results["Avg. BLEU"] = float(np.mean([b1, b2, b3, b4])) * 100

    return results


# --------------------------------------------------------------------------
# BERTScore-F1
# --------------------------------------------------------------------------

def compute_bertscore(hyps, refs, model_type=None, batch_size=32):
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("[skip] bert-score not installed. Run: pip install bert-score --break-system-packages")
        return float("nan")

    kwargs = dict(lang="en", verbose=False, batch_size=batch_size)
    if model_type:
        kwargs = dict(model_type=model_type, verbose=False, batch_size=batch_size)
    _, _, f1 = bert_score_fn(hyps, refs, **kwargs)
    return float(f1.mean())


# --------------------------------------------------------------------------
# Embedding Average / Vector Extrema / Greedy Matching  (spaCy vectors)
# --------------------------------------------------------------------------

def _sentence_vectors(nlp, text: str):
    doc = nlp(text)
    vecs = [tok.vector for tok in doc if tok.has_vector and not tok.is_space]
    return vecs


def _embedding_average(vecs):
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _vector_extrema(vecs):
    if not vecs:
        return None
    arr = np.stack(vecs)
    max_vals = arr.max(axis=0)
    min_vals = arr.min(axis=0)
    extrema = np.where(np.abs(max_vals) >= np.abs(min_vals), max_vals, min_vals)
    return extrema


def _cos(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _greedy_match_one_direction(vecs_a, vecs_b):
    if not vecs_a or not vecs_b:
        return 0.0
    sims = []
    for va in vecs_a:
        best = max(_cos(va, vb) for vb in vecs_b)
        sims.append(best)
    return float(np.mean(sims))


def compute_embedding_metrics(hyps, refs, spacy_model="en_core_web_md"):
    try:
        import spacy
    except ImportError:
        print("[skip] spaCy not installed. Run: pip install spacy --break-system-packages")
        return {"EA": float("nan"), "VE": float("nan"), "GM": float("nan")}

    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        print(f"[skip] spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
        return {"EA": float("nan"), "VE": float("nan"), "GM": float("nan")}

    ea_scores, ve_scores, gm_scores = [], [], []
    for hyp, ref in zip(hyps, refs):
        hv = _sentence_vectors(nlp, hyp)
        rv = _sentence_vectors(nlp, ref)
        if not hv or not rv:
            continue

        ea_h, ea_r = _embedding_average(hv), _embedding_average(rv)
        ea_scores.append(_cos(ea_h, ea_r))

        ve_h, ve_r = _vector_extrema(hv), _vector_extrema(rv)
        ve_scores.append(_cos(ve_h, ve_r))

        gm_hr = _greedy_match_one_direction(hv, rv)
        gm_rh = _greedy_match_one_direction(rv, hv)
        gm_scores.append((gm_hr + gm_rh) / 2.0)

    return {
        "EA": float(np.mean(ea_scores)) if ea_scores else float("nan"),
        "VE": float(np.mean(ve_scores)) if ve_scores else float("nan"),
        "GM": float(np.mean(gm_scores)) if gm_scores else float("nan"),
    }


# --------------------------------------------------------------------------
# Perplexity — reloads the actual trained KDPT + RDPT checkpoints
# --------------------------------------------------------------------------

def compute_perplexity(data_dir: str, rdpt_dir: str, kdpt_pt1: str, ckpt: str = None,
                        batch_size: int = 8, n_examples: int = -1, device_str: str = None):
    import pickle
    import torch
    from torch.utils.data import DataLoader

    from module import PrefixDialogModule
    from dataset import KnowledgeSeq2SeqDataset

    rdpt_dir = Path(rdpt_dir)
    hparams_path = rdpt_dir / "hparams.pkl"
    if not hparams_path.exists():
        print(f"[skip PPL] no hparams.pkl at {hparams_path}")
        return float("nan")
    with hparams_path.open("rb") as f:
        hparams = pickle.load(f)

    hparams.pfxKlgModel_name_or_path = str(kdpt_pt1)
    hparams.output_dir = str(rdpt_dir)

    if ckpt is None:
        ckpts = sorted(rdpt_dir.glob("f1=*.ckpt")) or sorted(rdpt_dir.glob("loss=*.ckpt"))
        if not ckpts:
            print(f"[skip PPL] no checkpoint found in {rdpt_dir}")
            return float("nan")
        ckpt = str(ckpts[0])
    print(f"[PPL] loading checkpoint: {ckpt}")

    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location="cpu")
    state = payload.get("state_dict", payload)

    model = PrefixDialogModule(hparams)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[PPL] warning: {len(missing)} missing keys on load (first few: {missing[:5]})")

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    dataset = KnowledgeSeq2SeqDataset(
        tokenizer=model.tokenizer,
        data_dir=str(Path(data_dir)),
        max_source_length=hparams.max_source_length,
        max_target_length=hparams.max_target_length,
        max_knowledge_length=hparams.max_knowledge_length,
        type_path="test",
        n_obs=n_examples if n_examples > 0 else None,
        preseqlen=hparams.preseqlen,
        tuning_mode="pt2",
        model_type="gpt2",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    total_loss, n_batches = 0.0, 0
    print(f"[PPL] running forward pass over {len(dataset)} test examples, batch_size={batch_size} ...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                outputs = model._step(batch)
            except Exception as exc:
                print(f"[PPL] batch {i} failed: {exc}")
                continue
            ce_loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if hasattr(ce_loss, "item"):
                total_loss += ce_loss.item()
                n_batches += 1
            if (i + 1) % 20 == 0:
                running_ppl = float(np.exp(total_loss / max(n_batches, 1)))
                print(f"[PPL]   batch {i + 1}/{len(loader)}  running PPL={running_ppl:.3f}")

    if n_batches == 0:
        return float("nan")
    mean_loss = total_loss / n_batches
    return float(np.exp(mean_loss))


# --------------------------------------------------------------------------
# Paper's published EMPOWER numbers, for a side-by-side printout
# --------------------------------------------------------------------------

PAPER_EMPOWER = {
    "PPL": 7.11, "BLEU-4": 3.37, "Avg. BLEU": 8.81, "F1": 0.20,
    "Knowledge-F1": 0.09, "BERTScore-F1": 0.86, "EA": 0.89, "VE": 0.48, "GM": 0.77,
}


def print_report(results: dict):
    print("\n" + "=" * 60)
    print(f"{'Metric':<16}{'Yours':>12}{'Paper (EMPOWER)':>20}")
    print("-" * 60)
    order = ["PPL", "BLEU-4", "Avg. BLEU", "F1", "Knowledge-F1",
              "BERTScore-F1", "EA", "VE", "GM", "ROUGE-L", "BLEU-1", "BLEU-2", "BLEU-3"]
    for key in order:
        if key not in results:
            continue
        val = results[key]
        val_str = f"{val:.4f}" if val == val else "n/a"  # NaN check
        paper_str = f"{PAPER_EMPOWER[key]:.4f}" if key in PAPER_EMPOWER else "—"
        print(f"{key:<16}{val_str:>12}{paper_str:>20}")
    print("=" * 60)
    print("Note: PPL lower is better; all other metrics higher is better.")
    print("EA/VE/GM here use spaCy vectors, not the paper's Word2Vec toolkit — see script docstring.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bertscore_model", default=None, help="e.g. distilbert-base-uncased for a faster/lighter run")
    parser.add_argument("--spacy_model", default="en_core_web_md")
    parser.add_argument("--compute_ppl", action="store_true")
    parser.add_argument("--data_dir", default="../data/preprocessed_v2")
    parser.add_argument("--rdpt_dir", default="output_rdpt")
    parser.add_argument("--kdpt_pt1", default="output/best_pt1")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--ppl_batch_size", type=int, default=8)
    parser.add_argument("--ppl_n_examples", type=int, default=-1, help="-1 = full test set")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    hyps, refs = load_preds(args.pred_file)
    print(f"Loaded {len(hyps)} prediction/reference pairs from {args.pred_file}")

    knowledge_list = None
    try:
        knowledge_list = load_knowledge(args.test_data)
        if len(knowledge_list) != len(hyps):
            print(f"[warn] test_data has {len(knowledge_list)} rows, preds has {len(hyps)} — "
                  f"Knowledge-F1 will be skipped (row counts must match).")
            knowledge_list = None
    except Exception as exc:
        print(f"[warn] could not load knowledge from test_data: {exc}")

    results = compute_text_metrics(hyps, refs, knowledge_list)

    print("Computing BERTScore-F1 (downloads a model the first time this runs)...")
    results["BERTScore-F1"] = compute_bertscore(hyps, refs, model_type=args.bertscore_model)

    print("Computing EA / VE / GM via spaCy vectors...")
    results.update(compute_embedding_metrics(hyps, refs, spacy_model=args.spacy_model))

    if args.compute_ppl:
        results["PPL"] = compute_perplexity(
            data_dir=args.data_dir,
            rdpt_dir=args.rdpt_dir,
            kdpt_pt1=args.kdpt_pt1,
            ckpt=args.ckpt,
            batch_size=args.ppl_batch_size,
            n_examples=args.ppl_n_examples,
        )
    else:
        print("[skip] PPL not requested — pass --compute_ppl to include it (reloads the model, slower).")
        results["PPL"] = float("nan")

    print_report(results)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved full results to {args.out_json}")


if __name__ == "__main__":
    sys.exit(main())