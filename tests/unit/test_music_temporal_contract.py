from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_bundle import FeatureBundle, in_clock
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import ClockMap, TemporalContext
from natural_features.features.audio._music_contract import (
    asset_digest, computation_context, mono_audio, music_series,
)
from natural_features.flow.cache import cache_fingerprint
from natural_features.storage.readers import read_feature_series
from natural_features.storage.writers import write_feature_series


def _feature():
    stimulus = AudioStimulus.from_array(
        np.ones(100, dtype=np.float32), 100, start_offset_s=4,
        clock="recording", temporal_context=TemporalContext((ClockMap("recording", "scan", 2, -3),)),
    )
    context = computation_context(
        stimulus, policy="bidirectional", bounds_s=[[4, 5]],
        preprocessing={"normalization": "whole_input"},
    )
    return music_series(
        stimulus, values=np.arange(6).reshape(2, 3), times_s=np.array([4.25, 4.75]),
        bounds_s=np.array([[4.2, 4.3], [4.7, 4.8]]), names=["a", "b", "c"],
        extractor="test.music", params={"layers": [1]}, context=context, hop_s=0.5,
    )


def test_coordinate_conversion_keeps_source_context_explicit():
    source = _feature()
    converted = in_clock(source, "scan")
    np.testing.assert_allclose(converted.times_s, [5.5, 6.5])
    np.testing.assert_allclose(converted.time_bounds_s, [[5.4, 5.6], [6.4, 6.6]])
    assert converted.metadata["computation_context"] == source.metadata["computation_context"]
    assert converted.metadata["computation_context"]["clock"] == "recording"
    payload = FeatureBundle({"music": converted}).temporal_payload("music")
    assert payload.metadata == converted.metadata


@pytest.mark.parametrize("fmt", ["npz", "zarr"])
def test_music_context_storage_roundtrip(tmp_path, fmt):
    if fmt == "zarr":
        pytest.importorskip("zarr")
    source = _feature()
    restored = read_feature_series(write_feature_series(source, tmp_path, fmt=fmt))
    assert restored.metadata == source.metadata
    assert restored.temporal_context == source.temporal_context
    assert restored.coords == source.coords
    np.testing.assert_array_equal(restored.values, source.values)
    np.testing.assert_array_equal(restored.time_bounds_s, source.time_bounds_s)


def test_local_asset_contents_and_layer_selection_change_identity(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"first")
    (model / "config.json").write_text("{}")
    first = asset_digest(model)
    weights.write_bytes(b"other")
    second = asset_digest(model)
    (model / "model.py").write_text("# new code")
    third = asset_digest(model)
    assert len({first, second, third}) == 3
    def key(revision, layers):
        return cache_fingerprint(
            extractor_name="audio.music.mert", params={"layers": layers},
            code_version="music-v1", model_revision=revision, upstream_ids=["audio"],
        )
    assert key(first, [1]) != key(second, [1])
    assert key(first, [1]) != key(first, [2])


def test_invalid_context_and_pcm_are_rejected():
    stimulus = AudioStimulus.from_array(np.ones(100, dtype=np.float32), 100)
    with pytest.raises(ValueError, match="exceed"):
        computation_context(stimulus, policy="local", bounds_s=[[0, 2]], preprocessing={})
    with pytest.raises(ValueError, match="ordered"):
        computation_context(stimulus, policy="local", bounds_s=[[1, 0]], preprocessing={})
    with pytest.raises(ValueError, match="integer PCM"):
        mono_audio(AudioStimulus.from_array(np.ones(10, dtype=np.int16), 100))
    stereo = AudioStimulus.from_array(np.ones((10, 2), dtype=np.float32), 100)
    with pytest.raises(ValueError, match="one channel"):
        mono_audio(stereo, channel_policy="mono")


def test_resampling_refuses_to_copy_native_context_onto_another_grid():
    from natural_features.fmri.resample import resample_feature_series

    with pytest.raises(ValueError, match="context-aware"):
        resample_feature_series(_feature(), tr_s=1)
