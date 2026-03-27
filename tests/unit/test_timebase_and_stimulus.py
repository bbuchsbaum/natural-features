from __future__ import annotations

import numpy as np

from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.core.timebase import is_monotonic_non_decreasing, times_from_hop, times_from_rate


def test_times_from_rate() -> None:
    t = times_from_rate(3, 2.0, start_offset_s=1.0)
    np.testing.assert_allclose(t, np.array([1.0, 1.5, 2.0]))


def test_times_from_hop_centered() -> None:
    t = times_from_hop(2, 0.5, center=True, window_s=1.0)
    np.testing.assert_allclose(t, np.array([0.5, 1.0]))


def test_video_frame_stream_chunks() -> None:
    frames = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    v = VideoStimulus.from_array(frames, fps=5.0)
    chunks = list(v.frame_stream(chunk_size=4))
    assert len(chunks) == 3
    assert chunks[0][1].shape[0] == 4
    assert is_monotonic_non_decreasing(v.frame_times_s)


def test_audio_stream_windowing() -> None:
    samples = np.arange(20, dtype=np.float32)
    a = AudioStimulus.from_array(samples, sr_hz=10)
    windows = list(a.audio_stream(window_s=0.5, hop_s=0.5))
    assert len(windows) == 4
    assert windows[0][0] == 0.0


def test_audio_stream_overlapping_keeps_tail_window() -> None:
    samples = np.arange(10, dtype=np.float32)
    a = AudioStimulus.from_array(samples, sr_hz=1)
    windows = list(a.audio_stream(window_s=5.0, hop_s=3.0))
    starts = [t for t, _ in windows]
    assert starts == [0.0, 3.0, 6.0, 9.0]
    assert windows[-1][1].shape[0] == 1
