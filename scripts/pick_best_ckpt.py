"""Pick the highest-F1 Lightning ckpt in an output dir and print its path."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

F1_RE = re.compile(r"f1[=:]([0-9.]+)", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    ckpts = list(args.output_dir.glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no .ckpt files in {args.output_dir}")
    scored = []
    for p in ckpts:
        m = F1_RE.search(p.name)
        score = float(m.group(1)) if m else -1.0
        scored.append((score, p.stat().st_mtime, p))
    scored.sort(key=lambda t: (t[0], t[1]))
    print(scored[-1][2].as_posix())


if __name__ == "__main__":
    main()
