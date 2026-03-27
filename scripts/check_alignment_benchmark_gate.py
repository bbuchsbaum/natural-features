#!/usr/bin/env python3
"""Evaluate alignment benchmark reports against soft/hard thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _check_metric(name: str, value: float | None, rule: dict[str, Any]) -> tuple[list[str], list[str]]:
    warns: list[str] = []
    fails: list[str] = []
    if value is None:
        warns.append(f"{name}: missing value; skipping threshold check")
        return warns, fails

    if "soft_max" in rule and value > float(rule["soft_max"]):
        warns.append(f"{name}: {value:.6g} > soft_max {float(rule['soft_max']):.6g}")
    if "hard_max" in rule and value > float(rule["hard_max"]):
        fails.append(f"{name}: {value:.6g} > hard_max {float(rule['hard_max']):.6g}")

    if "soft_min" in rule and value < float(rule["soft_min"]):
        warns.append(f"{name}: {value:.6g} < soft_min {float(rule['soft_min']):.6g}")
    if "hard_min" in rule and value < float(rule["hard_min"]):
        fails.append(f"{name}: {value:.6g} < hard_min {float(rule['hard_min']):.6g}")
    return warns, fails


def evaluate_gate(report: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary", {}))
    rules = dict(thresholds.get("metrics", {}))
    warnings: list[str] = []
    failures: list[str] = []
    for metric, rule in rules.items():
        raw = summary.get(metric)
        value = float(raw) if raw is not None else None
        w, f = _check_metric(metric, value, dict(rule))
        warnings.extend(w)
        failures.extend(f)
    return {
        "passed": len(failures) == 0,
        "warnings": warnings,
        "failures": failures,
        "summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True, help="Benchmark JSON report path")
    ap.add_argument(
        "--thresholds",
        type=Path,
        default=Path("tests/benchmarks/thresholds/alignment_quality_gate.json"),
        help="Threshold configuration JSON path",
    )
    ap.add_argument("--json", action="store_true", help="Emit full JSON gate result")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    gate = evaluate_gate(report, thresholds)

    if args.json:
        print(json.dumps(gate, indent=2, sort_keys=True))
    else:
        for w in gate["warnings"]:
            print(f"WARN: {w}")
        for f in gate["failures"]:
            print(f"FAIL: {f}")
        print(f"passed={gate['passed']}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
