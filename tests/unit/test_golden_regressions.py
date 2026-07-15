from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from natural_features.testing_helpers import build_tier_a_golden_reference


def _assert_numeric_summary_close(observed: dict[str, object], expected: dict[str, object]) -> None:
    assert observed["shape"] == expected["shape"]
    for key in ("time_start_s", "time_end_s"):
        if key in expected:
            assert observed[key] == expected[key]

    observed_oracle = observed["numeric_oracle"]
    expected_oracle = expected["numeric_oracle"]
    assert isinstance(observed_oracle, dict)
    assert isinstance(expected_oracle, dict)
    assert observed_oracle.keys() == expected_oracle.keys()
    assert observed_oracle["landmark_count"] == expected_oracle["landmark_count"]

    for key in expected_oracle.keys() - {"landmark_count"}:
        np.testing.assert_allclose(
            observed_oracle[key],
            expected_oracle[key],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Numeric oracle differs for {key}",
        )


def test_tier_a_golden_reference_matches() -> None:
    root = Path(__file__).resolve().parents[2]
    expected_path = root / "tests" / "fixtures" / "golden_reference_v1.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    observed = build_tier_a_golden_reference(root)

    assert observed["reference_version"] == expected["reference_version"]
    assert observed["tier"] == expected["tier"]
    assert observed["stimulus"] == expected["stimulus"]
    assert expected["runtime"]["numpy"]
    assert expected["runtime"]["python"]

    assert observed["visual_energy"] == expected["visual_energy"]
    # The exact MFCC fingerprint is diagnostic only: FFT/matrix backends can
    # differ by a few float32 ULPs across platforms. The numeric oracle keeps
    # the regression check strict enough to detect meaningful output drift.
    _assert_numeric_summary_close(observed["mfcc"], expected["mfcc"])
    assert observed["acoustic_phonetics"] == expected["acoustic_phonetics"]
    assert observed["multiscale_language"] == expected["multiscale_language"]
    _assert_numeric_summary_close(
        observed["audio_batch"]["matrix"], expected["audio_batch"]["matrix"]
    )
    _assert_numeric_summary_close(
        observed["audio_batch"]["collapsed"], expected["audio_batch"]["collapsed"]
    )
