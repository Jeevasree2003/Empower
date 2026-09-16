"""RDPT draft generation. Drafts are never shown raw to the user."""

from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch

from demo.paths import BEST_RDPT_CKPT, EMPOWER_MODEL, KDPT_PT1, RDPT_DIR

logger = logging.getLogger(__name__)

_MODEL = None
_DEVICE = None

MAX_NEW_TOKENS = 40
MAX_SOURCE = 384


def _ensure_model_path() -> None:
    model_dir = str(EMPOWER_MODEL)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


def _pick_ckpt(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit and explicit.exists():
        return explicit
    if BEST_RDPT_CKPT.exists():
        return BEST_RDPT_CKPT
    ckpts = sorted(RDPT_DIR.glob("f1=*.ckpt"))
    if not ckpts:
        return None
    best = None
    best_score = -1.0
    for path in ckpts:
        name = path.name
        try:
            score = float(name.split("f1=")[1].split("-")[0])
        except (IndexError, ValueError):
            score = -1.0
        if score > best_score:
            best_score = score
            best = path
    return best


def load_rdpt(ckpt: Optional[Path] = None):
    """Load PrefixDialogModule once. Returns None if checkpoints are missing."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ckpt_path = _pick_ckpt(ckpt)
    if ckpt_path is None or not KDPT_PT1.exists():
        logger.warning("rdpt_missing ckpt=%s pt1=%s", ckpt_path, KDPT_PT1)
        return None
    _ensure_model_path()
    os.chdir(str(EMPOWER_MODEL))
    from module import PrefixDialogModule

    hparams_path = RDPT_DIR / "hparams.pkl"
    hparams = None
    if hparams_path.exists():
        with hparams_path.open("rb") as handle:
            hparams = pickle.load(handle)
    if hparams is not None:
        hparams.pfxKlgModel_name_or_path = str(KDPT_PT1)
        hparams.eval_max_gen_length = MAX_NEW_TOKENS
        hparams.eval_min_gen_length = 2
        hparams.output_dir = str(RDPT_DIR)
    try:
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload.get("state_dict", payload)
    if hparams is None:
        hparams = payload.get("hyper_parameters")
    model = PrefixDialogModule(hparams)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("rdpt_missing_keys n=%s sample=%s", len(missing), missing[:8])
    if unexpected:
        logger.warning("rdpt_unexpected_keys n=%s", len(unexpected))
    model.hparams.eval_max_gen_length = MAX_NEW_TOKENS
    model.hparams.eval_min_gen_length = 2
    model.eval_max_length = MAX_NEW_TOKENS
    model.eval_min_length = 2
    model.tokenizer.padding_side = "left"
    model.tokenizer.truncation_side = "left"
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    logger.info("rdpt_loaded ckpt=%s device=%s", ckpt_path, _DEVICE)
    return _MODEL


@torch.no_grad()
def generate_draft(history: str, knowledge: str = "") -> str:
    """Return a short RDPT string, or empty on any failure.

    Knowledge is used only as extra context prepended to history when present.
    Do not append BOS: GPT-2 pad/eos/bos share one id.
    """
    try:
        model = load_rdpt()
        if model is None:
            return ""
        tokenizer = model.tokenizer
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        source = history.strip()
        if knowledge.strip():
            source = f"{knowledge.strip()} {tokenizer.sep_token} {source}"
        encoded = tokenizer(
            [source],
            max_length=MAX_SOURCE,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = _DEVICE
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        bsz = input_ids.size(0)
        klg_prompt_dict = model.prefix_model.get_prompt(bsz=bsz, return_dict=True)
        emb_weight = model.model.transformer.wte.weight
        prefix_past_kv_list, prefix_key_padding_mask, _ = model.prefix_model_2.get_prompt_2(
            bsz, klg_prompt_dict, emb_weight
        )
        prompt_len = input_ids.size(-1)
        generated_ids = model.model.generate(
            input_ids,
            pref_past_kv_list=prefix_past_kv_list,
            pref_key_padding_mask=prefix_key_padding_mask,
            attention_mask=attention_mask,
            use_cache=True,
            max_length=prompt_len + MAX_NEW_TOKENS,
            min_length=prompt_len + 2,
            pad_token_id=model.pad,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            no_repeat_ngram_size=3,
        )
        generated_ids = generated_ids[:, prompt_len:]
        text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return " ".join(text.split()).strip()
    except Exception:
        logger.exception("rdpt_generate_failed")
        return ""
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
