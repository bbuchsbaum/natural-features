from __future__ import annotations

import json
from pathlib import Path

from natural_features.features.speech.benchmark import BenchmarkConfig, run_alignment_benchmark


def test_tier_a_alignment_manifest_exists_and_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "tests" / "benchmarks" / "manifests" / "tier_a_alignment_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["manifest_version"] == 1
    assert isinstance(payload["items"], list) and payload["items"]
    item = payload["items"][0]
    assert "audio_path" in item
    assert "reference_ctm" in item or "reference_textgrid" in item


def test_tier_a_alignment_manifest_runs_with_backend_none() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "tests" / "benchmarks" / "manifests" / "tier_a_alignment_manifest.json"
    report = run_alignment_benchmark(
        manifest,
        config=BenchmarkConfig(backend="none", continue_on_error=False),
    )
    assert report["manifest_items"] == 1
    assert report["summary"]["n_success"] == 1
    assert report["summary"]["n_failed"] == 0
