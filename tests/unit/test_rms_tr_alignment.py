from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from natural_features import (
    build_experiment_grid,
    extract_features,
    query_feature_window_tr,
)
from natural_features.core.stimulus import AudioStimulus


def _block_tone(*, duration_s: float = 8.0, sr_hz: int = 8000) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr_hz)), dtype=np.float32) / sr_hz
    envelope = np.full(t.shape, 0.02, dtype=np.float32)
    envelope[(t >= 2.0) & (t < 6.0)] = 0.4
    return (envelope * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sr_hz: int) -> None:
    pcm = np.asarray(np.clip(samples, -1.0, 1.0) * 32767.0, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr_hz)
        wav.writeframes(pcm.tobytes())


def test_rms_energy_aligns_to_tr_grid_with_scan_onset_offset() -> None:
    sr_hz = 8000
    samples = _block_tone(duration_s=8.0, sr_hz=sr_hz)
    audio = AudioStimulus.from_array(samples, sr_hz=sr_hz)
    result = extract_features(audio, features=["audio.rms"])
    rms = result.features["audio.rms"]

    tr_s = 2.0
    stim_onset_s = 0.67
    duration_s = samples.shape[0] / sr_hz
    n_trs = int(np.ceil((stim_onset_s + duration_s) / tr_s))

    grid = build_experiment_grid(
        tr_s=tr_s,
        n_trs_by_run=[n_trs],
        run_starts_s=[0.0],
        feature_t0_s=stim_onset_s,
    )
    sampled = query_feature_window_tr(
        rms,
        grid,
        run_index=1,
        t_start_s=0.0,
        t_end_s=n_trs * tr_s,
        relative_to_run=True,
        method="mean",
        output_time="run_relative",
    )
    sampled_feature = query_feature_window_tr(
        rms,
        grid,
        run_index=1,
        t_start_s=0.0,
        t_end_s=n_trs * tr_s,
        relative_to_run=True,
        method="mean",
        output_time="feature",
    )

    expected_times = np.arange(n_trs, dtype=np.float64) * tr_s
    np.testing.assert_allclose(sampled.times_s, expected_times)
    np.testing.assert_allclose(sampled_feature.times_s, expected_times - stim_onset_s)
    assert sampled.values.shape == (n_trs, 1)
    assert np.all(np.isfinite(sampled.values))

    energy = sampled.values[:, 0]
    # Stimulus 2-6 s is loud; that interval is scan 2.67-6.67 s, so TRs at 4 s
    # and 6 s sit inside the loud block. TR 0 s is before / at stimulus onset.
    assert float(energy[2]) > float(energy[0])
    assert float(energy[3]) > float(energy[0])


def test_extract_features_accepts_wav_path_for_rms(tmp_path: Path) -> None:
    sr_hz = 8000
    samples = _block_tone(duration_s=4.0, sr_hz=sr_hz)
    path = tmp_path / "clip.wav"
    _write_wav(path, samples, sr_hz)

    result = extract_features(path, features=["audio.rms"])
    rms = result.features["audio.rms"]
    assert rms.values.ndim == 2
    assert rms.values.shape[0] > 1
    assert rms.coords["feature"] == ["rms"]
