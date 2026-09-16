"""Repo-relative paths for the Rakshak demo."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KARE_JSONL = REPO_ROOT / "KARE-data" / "KARE" / "Data" / "KARE.jsonl"
PREPROCESSED_V2 = REPO_ROOT / "data" / "preprocessed_v2"
EMPOWER_MODEL = REPO_ROOT / "EMPOWER-MODEL"
RDPT_DIR = EMPOWER_MODEL / "output_rdpt_v3"
KDPT_PT1 = EMPOWER_MODEL / "output_kdpt_v2" / "best_pt1"
BEST_RDPT_CKPT = RDPT_DIR / "f1=0.0884-step_count=0.ckpt"
AUDIT_LOG = REPO_ROOT / "reports" / "demo_turns.jsonl"
