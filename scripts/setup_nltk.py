#!/usr/bin/env python
"""Download NLTK corpora required by KTC filtering. Run once after pip install."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ktc.nltk_setup import download_nltk_resources, missing_nltk_resources


def main() -> int:
    before = missing_nltk_resources()
    downloaded = download_nltk_resources(quiet=False)
    still = missing_nltk_resources()
    if still:
        print("Still missing after download:", ", ".join(still), file=sys.stderr)
        return 1
    if downloaded:
        print("Downloaded:", ", ".join(downloaded))
    elif before:
        print("Downloaded previously missing resources.")
    else:
        print("NLTK data already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
