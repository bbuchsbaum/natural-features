"""All composite columns must describe identical, sample-derived windows."""
import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import rms, spectral_stats
from natural_features.features.audio.prosody import audio_pitch, prosody_features


@pytest.mark.parametrize("sr,hop,win,n", [(8000, .01, .025, 8000),
                                          (11025, .01003, .04003, 22050),
                                          (8000, .01, .025, 80)])
def test_composite_uses_common_sample_windows(sr, hop, win, n):
    t = np.arange(n) / sr
    x = (np.sin(2*np.pi*200*t) * (.1 + .3*(t > .5))).astype(np.float32)
    stimulus = AudioStimulus.from_array(x, sr_hz=sr, start_offset_s=1.7)
    out = prosody_features(stimulus, hop_s=hop, win_s=win)
    hop_n, win_n = max(1, round(sr*hop)), round(sr*max(.04, win))
    count = 1 + max(0, (n-win_n)//hop_n)
    expected_starts = 1.7 + np.arange(count)*hop_n/sr
    np.testing.assert_allclose(out.time_bounds_s[:, 0], expected_starts)
    np.testing.assert_allclose(out.time_bounds_s[:, 1], expected_starts+win_n/sr)
    np.testing.assert_allclose(out.times_s, expected_starts+win_n/(2*sr))
    effective = dict(hop_s=hop_n/sr, win_s=win_n/sr)
    r, p, s = rms(stimulus, **effective), audio_pitch(stimulus, **effective), spectral_stats(stimulus, **effective)
    np.testing.assert_array_equal(out.values[:, 0], r.values[:, 0])
    np.testing.assert_array_equal(out.values[:, 2:4], p.values)
    np.testing.assert_array_equal(out.values[:, 4], s.values[:, 0])
    np.testing.assert_array_equal(out.values[:, 5], s.values[:, 4])
    assert out.timebase.hop_s == hop_n/sr
    assert out.timebase.window_s == win_n/sr
    assert out.metadata["right_padding_samples"] == max(0, win_n-n)


def test_pitch_known_tone_offset_and_silence():
    sr = 16000
    t = np.arange(sr)/sr
    stimulus = AudioStimulus.from_array(np.sin(2*np.pi*200*t).astype(np.float32), sr_hz=sr)
    pitch = audio_pitch(stimulus)
    assert np.all(np.abs(pitch.values[:, 0]-200) < 3)
    silence = AudioStimulus.from_array(np.zeros(sr, np.float32), sr_hz=sr)
    np.testing.assert_array_equal(audio_pitch(silence).values, 0)
