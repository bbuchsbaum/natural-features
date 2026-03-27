from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from natural_features.workflows.audio_batch import extract_audio_dir, extract_audio_files


def _write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _make_clip(seconds: float, hz: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_extract_audio_files_matrices_and_df(tmp_path) -> None:
    p1 = tmp_path / "a.wav"
    p2 = tmp_path / "b.wav"
    _write_wav(p1, _make_clip(2.0, 220))
    _write_wav(p2, _make_clip(3.0, 330))

    out = extract_audio_files(
        [p1, p2],
        resolution_s=1.0,
        selected_features=["rms", "spectral_stats", "vad"],
        as_dataframe=True,
    )
    assert set(out.files.keys()) == {"a", "b"}
    assert out.files["a"].matrix.shape[1] > 0
    assert out.files["b"].matrix.shape[1] == out.files["a"].matrix.shape[1]
    assert out.long_dataframe is not None
    assert {"file_id", "time_s"}.issubset(set(out.long_dataframe.columns))


def test_extract_audio_dir_resolution_and_pattern(tmp_path) -> None:
    p1 = tmp_path / "c.wav"
    p2 = tmp_path / "d.wav"
    _write_wav(p1, _make_clip(2.5, 250))
    _write_wav(p2, _make_clip(5.0, 300))
    out = extract_audio_dir(
        tmp_path,
        resolution_s=0.5,
        selected_features=["rms", "mfcc"],
        as_dataframe=False,
    )
    assert set(out.files.keys()) == {"c", "d"}
    t_c = out.files["c"].times_s
    if len(t_c) > 1:
        dt = float(np.median(np.diff(t_c)))
        assert abs(dt - 0.5) < 1e-6
    assert out.long_dataframe is None


def test_extract_audio_files_with_collapse_stats(tmp_path) -> None:
    p1 = tmp_path / "e.wav"
    p2 = tmp_path / "f.wav"
    _write_wav(p1, _make_clip(2.0, 210))
    _write_wav(p2, _make_clip(4.0, 320))
    out = extract_audio_files(
        [p1, p2],
        resolution_s=1.0,
        selected_features=["rms", "vad"],
        as_dataframe=True,
        collapse="mean+sd",
    )
    a = out.files["e"]
    assert a.collapsed_vector is not None
    assert a.collapsed_feature_names is not None
    assert len(a.collapsed_vector) == 2 * len(a.feature_names)
    assert all(n.endswith(".mean") or n.endswith(".sd") for n in a.collapsed_feature_names)
    assert out.collapsed_dataframe is not None
    assert "file_id" in out.collapsed_dataframe.columns


def test_extract_audio_files_with_single_collapse_min_or_max(tmp_path) -> None:
    p = tmp_path / "g.wav"
    _write_wav(p, _make_clip(3.0, 250))
    out_min = extract_audio_files(
        [p],
        resolution_s=1.0,
        selected_features=["rms", "mfcc"],
        as_dataframe=False,
        collapse="min",
    )
    fmin = out_min.files["g"]
    assert fmin.collapsed_vector is not None
    assert len(fmin.collapsed_vector) == len(fmin.feature_names)
    assert out_min.collapsed_dataframe is not None

    out_max = extract_audio_files(
        [p],
        resolution_s=1.0,
        selected_features=["rms", "mfcc"],
        as_dataframe=False,
        collapse="max",
    )
    fmax = out_max.files["g"]
    assert fmax.collapsed_vector is not None
    assert len(fmax.collapsed_vector) == len(fmax.feature_names)
