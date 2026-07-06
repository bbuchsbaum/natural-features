from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus, ImageStimulus, MultiModalStimulus, VideoStimulus
from natural_features.core.timebase import is_monotonic_non_decreasing, times_from_hop, times_from_rate
from natural_features.features.vision.lowlevel import visual_energy


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


def test_image_stimulus_normalizes_and_wraps_one_frame() -> None:
    img = np.array([[[0, 128, 255], [64, 32, 16]], [[255, 0, 128], [255, 255, 255]]], dtype=np.uint8)
    stim = ImageStimulus.from_array(img, onset_s=2.0, duration_s=0.5)
    assert stim.image.shape == (2, 2, 3)
    assert stim.image.dtype == np.float32
    assert float(stim.image.max()) <= 1.0
    np.testing.assert_allclose(stim.frame_times_s, np.array([2.0]))
    assert stim.as_frames().shape == (1, 2, 2, 3)
    assert MultiModalStimulus(image=stim).start_offset_s == 2.0


def test_image_stimulus_defaults_to_zero_onset() -> None:
    stim = ImageStimulus.from_array(np.ones((2, 2), dtype=np.float32))
    np.testing.assert_allclose(stim.frame_times_s, np.array([0.0]))
    assert MultiModalStimulus(image=stim).start_offset_s == 0.0


def test_image_from_file_requires_pillow_or_loads(tmp_path) -> None:
    pil = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    data = np.zeros((3, 4, 3), dtype=np.uint8)
    data[:, :, 1] = 255
    pil.fromarray(data).save(path)

    stim = ImageStimulus.from_file(path, onset_s=1.0)
    assert stim.source == str(path.resolve())
    assert stim.image.shape == (3, 4, 3)
    np.testing.assert_allclose(stim.frame_times_s, np.array([1.0]))


def test_visual_energy_accepts_image_stimulus() -> None:
    img = np.zeros((3, 4, 3), dtype=np.float32)
    img[:, :, 0] = 0.25
    img[:, :, 1] = 0.5
    img[:, :, 2] = 0.75
    out = visual_energy(ImageStimulus.from_array(img, onset_s=1.5))
    assert out.values.shape == (1, 8)
    np.testing.assert_allclose(out.times_s, np.array([1.5]))
    assert out.timebase.sampling_rate_hz is None


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
