"""JSON summaries and timing helpers for presentation suite runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_run_summary(output_path: Path, payload: dict[str, Any]) -> None:
    """Write ``run_summary.json`` with a consistent baseline schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "suite": "v2_paper_2026",
        "generated_utc": utc_now_iso(),
        **payload,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
