"""Append one JSON object per chat turn for the panel trustworthiness trail."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from demo.paths import AUDIT_LOG


def append_turn(record: Dict[str, Any], path: Optional[Path] = None) -> Path:
    target = path or AUDIT_LOG
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target
