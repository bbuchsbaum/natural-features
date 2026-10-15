"""Opt-in checks against the exact local separator and CREPE assets."""
import os

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.vocal_f0 import _load_runtime, audio_vocal_f0

pytestmark = [pytest.mark.external, pytest.mark.nightly]


def test_real_models_preserve_clocks_missingness_and_rng():
    checkpoint = os.environ.get("NF_TEST_VOCAL_F0_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set NF_TEST_VOCAL_F0_CHECKPOINT to the pinned local Demucs asset")
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        runtime = _load_runtime(checkpoint, None, True, "cpu")
        audio = (0.2 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
        first, confidence = runtime.pitch(audio, 160, 50, 1000)
        second, other_confidence = runtime.pitch(audio, 160, 50, 1000)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(confidence, other_confidence)
        assert np.median(np.abs(1200 * np.log2(first[5:-5] / 220))) < 50
        result = audio_vocal_f0(
            AudioStimulus.from_array(np.zeros(16001, np.float32), 16000, start_offset_s=7.5),
            separator_checkpoint=checkpoint,
        )
        assert np.isnan(result.values[:, 0]).all()
        assert not result.values[:, 4].any()
        np.testing.assert_allclose(result.times_s, 7.5 + np.arange(101) / 100)
        assert torch.equal(torch_state, torch.random.get_rng_state())
        after = np.random.get_state()
        assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
        np.testing.assert_array_equal(numpy_state[1], after[1])
    finally:
        torch.set_num_threads(threads)
