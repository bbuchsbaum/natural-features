from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import rms
from natural_features.features.speech.align import whisperx_align
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.fmri.design import add_lags, concat_feature_series
from natural_features.fmri.render import render_events
from natural_features.fmri.resample import build_tr_grid, resample_feature_series


@pytest.mark.media
def test_aligned_words_to_design_matrix_e2e() -> None:
    sr = 8000
    t = np.arange(sr * 6, dtype=np.float32) / sr
    wav = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = AudioStimulus.from_array(wav, sr_hz=sr)

    asr = whisper_transcribe(audio, transcript_text="HH AH0 L OW1 W ER1 L D", strict_dependency=False)
    aligned = whisperx_align(audio, asr["words"], backend="none", strict_dependency=False)
    words = aligned["words"]

    tr = 1.5
    duration_s = audio.samples.shape[0] / audio.sr_hz
    grid = build_tr_grid(duration_s=duration_s, tr_s=tr, start_s=0.0)
    events = render_events(words, grid, mode="impulse", value="count")
    events_lag = add_lags(events, [0, 1, 2])

    energy = rms(audio, hop_s=0.02, win_s=0.03)
    energy_tr = resample_feature_series(energy, tr_s=tr, method="mean", time_grid_s=grid)
    x = concat_feature_series([events_lag, energy_tr], standardize=True, add_intercept=True)

    assert x.values.shape[0] == len(grid)
    assert x.values.shape[1] >= 2
    assert np.all(np.isfinite(x.values))
