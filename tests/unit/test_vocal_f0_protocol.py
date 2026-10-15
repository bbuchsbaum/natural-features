from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _protocol_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "qualify_vocal_f0.py"
    spec = importlib.util.spec_from_file_location("qualify_vocal_f0", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load qualification script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_negatives_score_every_native_center_once_without_missing_end():
    from types import SimpleNamespace
    protocol = _protocol_module()
    annotation_times = np.array([0, 0.0058, 0.0116, 0.0174])
    annotation_hz = np.array([0, 220, 220, 0])
    native = SimpleNamespace(times_s=np.array([0, 0.01, 0.02]))
    times, hz = protocol._scoring_reference("backing", annotation_times, annotation_hz, native, 0.02)
    np.testing.assert_array_equal(times, [0, 0.01])
    np.testing.assert_array_equal(hz, [0, 0])
    positive_times, positive_hz = protocol._scoring_reference("solo", annotation_times, annotation_hz, native, 0.02)
    np.testing.assert_array_equal(positive_times, annotation_times)
    np.testing.assert_array_equal(positive_hz, annotation_hz)


def test_reference_mask_and_mixture_use_reference_voiced_samples_only() -> None:
    protocol = _protocol_module()
    ref_times = np.array([0.0, 0.01, 0.02])
    ref_hz = np.array([0.0, 220.0, np.nan])
    mask = protocol._reference_mask(ref_times, ref_hz, 480)
    source = np.ones(480, dtype=np.float32)
    backing = np.full(480, 2.0, dtype=np.float32)

    mixed, recipe = protocol._mixture(source, backing, mask, 0.0)

    assert mask.sum() == 160
    assert recipe["backing_scale"] == pytest.approx(0.5)
    np.testing.assert_allclose(mixed, 0.95)


def test_macro_requires_all_six_non_missing_rows() -> None:
    protocol = _protocol_module()
    rows = [{"metrics": {"raw_pitch_accuracy": 0.9}} for _ in range(5)]

    assert protocol._macro(rows, ["raw_pitch_accuracy"]) == {"raw_pitch_accuracy": None}


def test_minus_six_db_recipe_makes_backing_louder_than_vocal() -> None:
    protocol = _protocol_module()
    source = np.ones(160, dtype=np.float32)
    backing = np.full(160, 2.0, dtype=np.float32)
    _, recipe = protocol._mixture(source, backing, np.ones(160, dtype=bool), -6.0)

    assert recipe["target_vocal_to_backing_db"] == -6.0
    assert recipe["backing_scale"] == pytest.approx(10 ** (6 / 20) / 2.0)
