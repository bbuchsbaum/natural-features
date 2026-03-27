from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import importlib.util

import pytest


def test_tier_b_manifest_shape_and_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "tests" / "stimuli" / "tier_b" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["manifest_version"] == 1
    assert isinstance(payload["entries"], list)
    for e in payload["entries"]:
        for key in ["id", "enabled", "source_url", "path", "sha256", "diagnostic_tags"]:
            assert key in e
        assert e["enabled"] is False


def test_fetch_tier_b_script_no_enabled_entries() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "fetch_tier_b_stimuli.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--allow-missing-sha"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Fetched 0 Tier B entries" in proc.stdout


def test_prepare_tier_b_script_no_raw_files() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "prepare_tier_b_clips.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Prepared 0 Tier B clips" in proc.stdout


def test_fetch_tier_b_rejects_non_https_urls(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "fetch_tier_b_stimuli.py"
    spec = importlib.util.spec_from_file_location("fetch_tier_b_stimuli", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load fetch_tier_b_stimuli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Only https:// URLs are allowed"):
        module.download("http://example.com/file.wav", tmp_path / "x.wav")
