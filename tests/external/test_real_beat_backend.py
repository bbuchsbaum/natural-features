"""Opt-in parity with a locally provisioned, content-identified checkpoint."""
from __future__ import annotations

import os

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.beat import music_beat_activations

pytestmark = [pytest.mark.external, pytest.mark.nightly]


@pytest.mark.parametrize("seconds", [2, 31])
def test_real_beat_this_matches_direct_upstream_frames(seconds, monkeypatch):
    checkpoint = os.environ.get("NF_TEST_BEAT_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set NF_TEST_BEAT_CHECKPOINT to a local Beat This checkpoint")
    torch = pytest.importorskip("torch")
    from beat_this.inference import Audio2Frames

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url",
                        lambda *a, **k: pytest.fail("offline parity attempted a download"))
    sr = 22050
    t = np.arange(sr * seconds) / sr
    envelope = np.exp(-35 * (t % 0.5))
    waveform = (0.2 * envelope * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 330 * t))).astype(np.float32)
    stimulus = AudioStimulus.from_array(waveform, sr, start_offset_s=7, clock="music")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        direct = Audio2Frames(checkpoint_path=checkpoint, device="cpu")
        expected = np.column_stack([x.sigmoid().detach().numpy() for x in direct(waveform, sr)])
        observed = music_beat_activations(stimulus, checkpoint=checkpoint)
    finally:
        torch.set_num_threads(old_threads)
    np.testing.assert_allclose(observed.values, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(observed.times_s, 7 + np.arange(len(expected)) / 50)
    assert observed.values.shape == (1 + len(waveform) // 441, 2)
    assert observed.clock == "music"
    assert observed.metadata["fallback_used"] is False
    context = np.asarray(observed.metadata["computation_context"]["dependency_bounds_s"])
    assert context.shape == (len(expected), 2)
    assert np.all(context[:, 0] >= 7) and np.all(context[:, 1] <= 7 + seconds)
    if seconds == 31:
        assert len(np.unique(context, axis=0)) > 1
