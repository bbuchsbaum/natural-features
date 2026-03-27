from __future__ import annotations

import importlib.util
from pathlib import Path


def _evaluate_gate():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "check_alignment_benchmark_gate.py"
    spec = importlib.util.spec_from_file_location("check_alignment_benchmark_gate", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load check_alignment_benchmark_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_gate


def test_evaluate_gate_soft_warning_only() -> None:
    evaluate_gate = _evaluate_gate()
    report = {"summary": {"fallback_rate": 0.2, "token_f1_mean": 0.95}}
    thresholds = {
        "metrics": {
            "fallback_rate": {"soft_max": 0.15, "hard_max": 0.4},
            "token_f1_mean": {"soft_min": 0.9, "hard_min": 0.75},
        }
    }
    out = evaluate_gate(report, thresholds)
    assert out["passed"] is True
    assert any("fallback_rate" in w for w in out["warnings"])
    assert out["failures"] == []


def test_evaluate_gate_hard_failure() -> None:
    evaluate_gate = _evaluate_gate()
    report = {"summary": {"boundary_mae_ms_mean": 1800.0}}
    thresholds = {
        "metrics": {
            "boundary_mae_ms_mean": {"soft_max": 900.0, "hard_max": 1500.0},
        }
    }
    out = evaluate_gate(report, thresholds)
    assert out["passed"] is False
    assert any("hard_max" in f for f in out["failures"])
