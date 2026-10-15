from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import FeatureSeries
from natural_features.features.audio.vocal_f0_benchmark import score_vocal_f0


_NAMES = [
    "confidence",
    "missing",
    "f0_hz",
    "voiced",
    "source_support",
    "octave_jump",
    "source_supported",
    "range_edge",
]


def _estimate(times: list[float], f0: list[float], voiced: list[float]) -> FeatureSeries:
    values = np.column_stack(
        [
            np.full(len(times), 0.9),
            np.zeros(len(times)),
            np.asarray(f0),
            np.asarray(voiced),
            np.ones(len(times)),
            np.zeros(len(times)),
            np.ones(len(times)),
            np.zeros(len(times)),
        ]
    )
    return FeatureSeries(
        values=values,
        times_s=np.asarray(times),
        coords={"feature": _NAMES},
        metadata={"extractor_id": "test", "params_hash": "test"},
    )


def test_scores_octaves_misses_and_unmatched_without_extrapolation() -> None:
    estimate = _estimate([0.0, 0.01, 0.02], [220.0, 440.0, np.nan], [1, 1, 1])
    metrics = score_vocal_f0(
        [0.0, 0.01, 0.02, 0.04], [220.0, 220.0, 220.0, 220.0], estimate
    )

    assert metrics["n_reference"] == 4
    assert metrics["n_scored"] == 4
    assert metrics["n_unmatched"] == 1
    assert metrics["n_joint_voiced"] == 2
    assert metrics["raw_pitch_accuracy"] == pytest.approx(0.25)
    assert metrics["voicing_recall"] == pytest.approx(0.5)
    assert metrics["median_abs_cents_error"] == pytest.approx(600.0)
    assert metrics["octave_error_rate"] == pytest.approx(0.5)
    assert metrics["gross_pitch_error_rate"] == pytest.approx(0.5)


def test_unvoiced_false_alarm_uses_valid_voiced_f0_only_and_nan_references_are_unknown() -> None:
    estimate = _estimate([0.0, 0.01, 0.02], [-100.0, 110.0, np.nan], [1, 1, 1])
    metrics = score_vocal_f0([0.0, 0.01, 0.02], [0.0, np.nan, 0.0], estimate)

    assert metrics["n_scored"] == 2
    assert metrics["n_reference_unvoiced"] == 2
    assert metrics["n_unmatched"] == 0
    assert metrics["voicing_false_alarm_rate"] == pytest.approx(0.0)
    assert metrics["n_joint_voiced"] == 0
    assert metrics["raw_pitch_accuracy"] is None
    assert metrics["voicing_recall"] is None
    assert metrics["median_abs_cents_error"] is None
    assert metrics["octave_error_rate"] is None
    assert metrics["gross_pitch_error_rate"] is None


def test_nearest_timestamp_tie_uses_earlier_estimate() -> None:
    estimate = _estimate([0.0, 0.01], [220.0, 440.0], [1, 1])
    metrics = score_vocal_f0([0.005], [220.0], estimate, max_time_error_s=0.005)

    assert metrics["raw_pitch_accuracy"] == pytest.approx(1.0)
    assert metrics["octave_error_rate"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("times", "hz"),
    [([0.0, 0.0], [220.0, 220.0]), ([0.0], [-1.0]), ([np.inf], [220.0])],
)
def test_rejects_invalid_reference_arrays(times: list[float], hz: list[float]) -> None:
    with pytest.raises(ValueError):
        score_vocal_f0(times, hz, _estimate([0.0], [220.0], [1]))


def test_rejects_mismatched_reference_shapes() -> None:
    with pytest.raises(ValueError):
        score_vocal_f0([0.0], [220.0, 220.0], _estimate([0.0], [220.0], [1]))
