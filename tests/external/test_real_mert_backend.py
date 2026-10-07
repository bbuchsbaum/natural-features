"""Offline parity check for a deliberately provisioned, pinned MERT checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.mert import audio_music_mert


_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


@pytest.mark.external
def test_pinned_mert_matches_direct_upstream_for_selected_layers() -> None:
    """No download: this runs only when an explicit local pinned asset is selected."""
    raw_path = os.environ.get("NF_TEST_MERT_MODEL")
    if not raw_path:
        pytest.skip("set NF_TEST_MERT_MODEL to a pinned local MERT snapshot")
    root = Path(raw_path)
    if not (root / "config.json").is_file() or os.environ.get("NF_TEST_MERT_REVISION") != _REVISION:
        pytest.skip("NF_TEST_MERT_MODEL must be accompanied by the expected pinned revision")

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    sr_hz = 24_000
    samples = (0.1 * np.sin(2.0 * np.pi * 440.0 * np.arange(sr_hz) / sr_hz)).astype(np.float32)
    stimulus = AudioStimulus.from_array(samples, sr_hz=sr_hz)
    selected = [0, 6, 12]
    result = audio_music_mert(
        stimulus, model=root, layers=selected, trust_remote_code=True, local_files_only=True
    )

    processor = AutoFeatureExtractor.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(str(root), local_files_only=True, trust_remote_code=True)
    model.eval()
    inputs = processor(samples, sampling_rate=sr_hz, return_tensors="pt")
    with torch.inference_mode():
        direct = model(**inputs, output_hidden_states=True).hidden_states
    expected = np.stack(
        [direct[layer].detach().cpu().numpy()[0].astype(np.float32) for layer in selected], axis=1
    )
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=0.0)
    assert result.coords["layer"] == selected
    assert result.values.shape == (74, 3, 768)
    np.testing.assert_allclose(result.times_s[[0, -1]], [199.5 / sr_hz, 23_559.5 / sr_hz])
