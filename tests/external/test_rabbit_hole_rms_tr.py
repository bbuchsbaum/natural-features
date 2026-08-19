from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from natural_features import (
    build_experiment_grid,
    extract_features,
    query_feature_window_tr,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAV = (
    REPO_ROOT / "data" / "stimuli" / "DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav"
)
WAV_ENV = "NF_RABBIT_HOLE_WAV"
TR_S = 2.0
STIM_ONSET_S = 0.67


def _wav_path() -> Path | None:
    env = os.environ.get(WAV_ENV, "").strip()
    candidate = Path(env).expanduser() if env else DEFAULT_WAV
    return candidate if candidate.is_file() else None


WAV = _wav_path()
pytestmark = [pytest.mark.external, pytest.mark.media]


@pytest.mark.skipif(
    WAV is None, reason=f"Rabbit Hole WAV not found ({WAV_ENV} or {DEFAULT_WAV})"
)
def test_rabbit_hole_rms_samples_onto_2s_tr_grid() -> None:
    assert WAV is not None
    result = extract_features(WAV, features=["audio.rms"])
    rms = result.features["audio.rms"]
    audio = result.inputs["audio"]
    duration_s = float(audio.samples.shape[0] / audio.sr_hz)
    n_trs = int(np.ceil((STIM_ONSET_S + duration_s) / TR_S))

    grid = build_experiment_grid(
        tr_s=TR_S,
        n_trs_by_run=[n_trs],
        run_starts_s=[0.0],
        feature_t0_s=STIM_ONSET_S,
    )
    sampled = query_feature_window_tr(
        rms,
        grid,
        run_index=1,
        t_start_s=0.0,
        t_end_s=n_trs * TR_S,
        relative_to_run=True,
        method="mean",
        output_time="run_relative",
    )

    assert pytest.approx(duration_s, abs=0.02) == 743.659
    assert n_trs == 373
    assert sampled.values.shape == (373, 1)
    np.testing.assert_allclose(sampled.times_s[:3], np.array([0.0, 2.0, 4.0]))
    np.testing.assert_allclose(sampled.times_s[-1], 744.0)
    assert np.all(np.isfinite(sampled.values))

    energy = sampled.values[:, 0]
    # The soundtrack is near-silent for ~20 s, then rises. Scan time 20 s is
    # stimulus time 19.33 s, still in the quiet pad; TR 12 (24 s scan / 23.33 s
    # stimulus) is into the audible section.
    quiet = float(np.mean(energy[:8]))
    later = float(np.mean(energy[12:24]))
    assert later > quiet
    assert later > 0.0
