"""Install-time NLTK data locations. Filtering must not download at runtime."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import nltk

# Download names used by scripts/setup_nltk.py (covers old and new NLTK layouts).
NLTK_RESOURCES: Tuple[Tuple[str, str], ...] = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
)

# Filtering only tokenizes and POS-tags; either generation of each pair is enough.
_FILTER_PUNKT_PATHS = ("tokenizers/punkt", "tokenizers/punkt_tab")
_FILTER_TAGGER_PATHS = (
    "taggers/averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng",
)


def _has_any(paths: Sequence[str]) -> bool:
    for find_path in paths:
        try:
            nltk.data.find(find_path)
            return True
        except LookupError:
            continue
    return False


def missing_nltk_resources() -> List[str]:
    missing: List[str] = []
    for find_path, name in NLTK_RESOURCES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            missing.append(name)
    return missing


def missing_filter_resources() -> List[str]:
    missing: List[str] = []
    if not _has_any(_FILTER_PUNKT_PATHS):
        missing.extend(["punkt", "punkt_tab"])
    if not _has_any(_FILTER_TAGGER_PATHS):
        missing.extend(["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"])
    return missing


def download_nltk_resources(quiet: bool = False) -> List[str]:
    """Download missing resources. Returns names that were requested."""
    downloaded: List[str] = []
    for find_path, name in NLTK_RESOURCES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(name, quiet=quiet)
            downloaded.append(name)
    return downloaded


def setup_command() -> str:
    return "python scripts/setup_nltk.py"
