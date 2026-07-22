from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import ClockMap, TemporalContext
from natural_features.workflows.acoustic_phonetics import extract_acoustic_phonetics


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_extract_acoustic_phonetics_native_and_resampled() -> None:
    a = _audio()
    native = extract_acoustic_phonetics(
        a,
        hop_s=0.02,
        posterior_backend="ctc",
        ctc_local_files_only=True,
        ctc_strict_dependency=False,
    )
    assert native.posteriorgrams.values.ndim == 2
    assert native.articulatory.values.ndim == 2
    assert native.posteriorgrams.values.shape[0] == native.articulatory.values.shape[0]
    names = list(native.articulatory.coords.get("feature", []))
    assert "bilabial" in names
    assert "posterior_entropy" in names

    res = extract_acoustic_phonetics(
        a,
        hop_s=0.02,
        posterior_backend="acoustic",
        resolution_s=0.5,
    )
    assert res.posteriorgrams.values.shape[0] < native.posteriorgrams.values.shape[0]
    assert res.articulatory.values.shape[0] == res.posteriorgrams.values.shape[0]


def test_acoustic_phonetics_preserves_input_clock_and_context() -> None:
    base = _audio()
    context = TemporalContext((ClockMap("stimulus", "scan:run-01", offset_s=-23.0),))
    audio = AudioStimulus.from_array(
        base.samples,
        sr_hz=base.sr_hz,
        clock="scan:run-01",
        temporal_context=context,
    )

    result = extract_acoustic_phonetics(
        audio,
        posterior_backend="acoustic",
        hop_s=0.1,
    )

    assert result.posteriorgrams.clock == "scan:run-01"
    assert result.articulatory.clock == "scan:run-01"
    assert result.posteriorgrams.temporal_context == context
