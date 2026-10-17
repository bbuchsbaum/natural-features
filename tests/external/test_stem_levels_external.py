"""Opt-in pinned-model execution and gain oracle; not separator qualification."""
import os

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.stem_levels import audio_stem_levels

pytestmark = [pytest.mark.external, pytest.mark.nightly]


def test_real_stem_model_preserves_gain_clock_and_paired_outputs():
    checkpoint = os.environ.get("NF_TEST_VOCAL_F0_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set NF_TEST_VOCAL_F0_CHECKPOINT to the pinned local Demucs asset")
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sr = 44100
        t = np.arange(3 * sr) / sr
        x = np.column_stack([.1 * np.sin(2 * np.pi * 220 * t), .08 * np.sin(2 * np.pi * 330 * t)])
        outputs = [audio_stem_levels(
            AudioStimulus.from_array((gain * x).astype(np.float32), sr, start_offset_s=7.25, clock="song"),
            separator_checkpoint=checkpoint,
        ) for gain in [1, 2]]
        base, doubled = outputs[0]["default"], outputs[1]["default"]
        assert base.values.shape == (28, 4)
        assert np.isfinite(base.values).all()
        assert base.clock == "song"
        np.testing.assert_allclose(base.times_s[[0, -1]], [7.375, 10.075])
        np.testing.assert_allclose(doubled.values - base.values, 20 * np.log10(2), atol=.02)
        diag = outputs[0]["diagnostics"]
        assert base.metadata["extractor_id"] != diag.metadata["extractor_id"]
        assert base.metadata["output_pair_id"] == diag.metadata["output_pair_id"]
        assert base.metadata["computation_context"]["dependency_bounds_s"] == [[7.25, 10.25]]
    finally:
        torch.set_num_threads(threads)
