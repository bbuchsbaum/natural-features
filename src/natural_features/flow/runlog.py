"""Run logging helpers."""

from __future__ import annotations

import json
from pathlib import Path

from natural_features.flow.engine import FlowRunResult


def write_run_json(result: FlowRunResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return p

