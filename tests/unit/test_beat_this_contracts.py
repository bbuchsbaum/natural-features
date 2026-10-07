from __future__ import annotations

from contextlib import nullcontext
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.backend_errors import BackendInferenceError, BackendLoadError
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import ClockMap, TemporalContext
from natural_features.features.audio import beat


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.fixture
def backend(monkeypatch, tmp_path):
    state = {"eval": False, "loads": 0, "bad_shape": False, "fail_inference": False}

    class Model:
        def __init__(self, dim=128):
            assert dim == 128

        def load_state_dict(self, weights):
            assert weights == {"x": 1}

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            state["eval"] = True
            return self

    def load(stream, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is True
        assert stream.read() == b"weights"
        state["loads"] += 1
        return {"hyper_parameters": {"dim": 128, "ignored_training": 99}, "state_dict": {"x": 1}}

    def predict(*, spect, chunk_size, border_size, overlap_mode, model):
        assert state["eval"]
        assert (chunk_size, border_size, overlap_mode) == (1500, 6, "keep_first")
        if state["fail_inference"]:
            raise RuntimeError("backend broke")
        n = spect.shape[0] - int(state["bad_shape"])
        return {"beat": Tensor(np.full(n, np.log(3))), "downbeat": Tensor(np.full(n, -np.log(3)))}

    torch = SimpleNamespace(
        load=load, tensor=lambda x, **kw: Tensor(x), float32=np.float32,
        inference_mode=nullcontext, __version__="test", hub=SimpleNamespace(get_dir=lambda: str(tmp_path)),
    )
    deps = SimpleNamespace(
        torch=torch, model=Model, predict=predict, version="1.1.0",
        frontend_version="test", rotary_version="test",
        replace_keys=lambda weights, *_: weights,
        frontend=lambda **kw: lambda wav: Tensor(np.ones((1 + len(wav.values) // 441, 128))),
    )
    monkeypatch.setattr(beat, "_dependencies", lambda: deps)
    monkeypatch.setattr(beat, "urlopen", lambda *_args, **_kw: pytest.fail("unexpected network access"))
    checkpoint = tmp_path / "test.ckpt"
    checkpoint.write_bytes(b"weights")
    return state, checkpoint


def _audio(n=22050, sr=22050):
    return AudioStimulus.from_array(
        np.zeros(n, dtype=np.float32), sr, start_offset_s=3, clock="music",
        temporal_context=TemporalContext((ClockMap("music", "scanner", offset_s=5),)),
    )


def test_activation_scores_timing_clock_and_identity(backend):
    state, checkpoint = backend
    out = beat.music_beat_activations(_audio(), checkpoint=str(checkpoint))
    np.testing.assert_allclose(out.values, np.tile([0.75, 0.25], (51, 1)), atol=1e-7)
    np.testing.assert_allclose(out.times_s, 3 + np.arange(51) / 50)
    assert out.clock == "music" and out.temporal_context == _audio().temporal_context
    assert out.coords["feature"] == ["beat", "downbeat"]
    assert out.metadata["calibrated_confidence"] is False
    assert out.metadata["model_revision"].startswith("sha256:")
    np.testing.assert_allclose(out.metadata["computation_context"]["dependency_bounds_s"],
                               np.tile([3, 4], (51, 1)))
    repeat = beat.music_beat_activations(_audio(), checkpoint=str(checkpoint))
    np.testing.assert_array_equal(out.values, repeat.values)
    logits = beat.music_beat_activations(_audio(), checkpoint=str(checkpoint), representation="logit")
    np.testing.assert_allclose(logits.values[0], [np.log(3), -np.log(3)])
    assert logits.metadata["extractor_id"] != out.metadata["extractor_id"]
    assert state["loads"] == 3


@pytest.mark.parametrize("n", [513, 1024, 22050, 30001])
def test_frontend_support_matches_explicit_reflected_sample_indices(n):
    observed = beat._frontend_bounds(n) * 22050
    padded = np.pad(np.arange(n), (512, 512), mode="reflect")
    expected = []
    for start in range(0, n + 1, 441):
        indices = padded[start:start + 1024]
        expected.append([indices.min(), indices.max() + 1])
    np.testing.assert_allclose(observed, expected, atol=1e-9)


def test_chunk_context_accounts_for_shifted_last_chunk_and_keep_first():
    # A 1500-frame piece yields two almost-complete overlapping chunks.
    support = np.column_stack([np.arange(1500), np.arange(1500) + 1]).astype(float)
    observed = beat._chunk_context(support)
    np.testing.assert_array_equal(observed[:1488], np.tile([0, 1494], (1488, 1)))
    np.testing.assert_array_equal(observed[1488:], np.tile([6, 1500], (12, 1)))
    long = np.column_stack([np.arange(3000), np.arange(3000) + 1]).astype(float)
    observed = beat._chunk_context(long)
    np.testing.assert_array_equal(observed[1488], [1482, 2982])
    np.testing.assert_array_equal(observed[-1], [1506, 3000])


def test_offline_cache_miss_and_bad_digest_never_reach_loader(backend, tmp_path):
    state, checkpoint = backend
    with pytest.raises(BackendLoadError, match="not cached"):
        beat.music_beat_activations(_audio(), cache_dir=str(tmp_path))
    with pytest.raises(BackendLoadError, match="SHA256 mismatch"):
        beat.music_beat_activations(_audio(), checkpoint=str(checkpoint), checkpoint_sha256="0" * 64)
    assert state["loads"] == 0
    good = hashlib.sha256(b"weights").hexdigest()
    beat.music_beat_activations(_audio(), checkpoint=str(checkpoint), checkpoint_sha256=good)


def test_rate_short_audio_and_representation_refusals(backend):
    state, checkpoint = backend
    for stimulus, kwargs, message in [
        (_audio(sr=16000), {}, "22050"),
        (_audio(n=512), {}, "512"),
        (_audio(), {"representation": "confidence"}, "representation"),
    ]:
        with pytest.raises(ValueError, match=message):
            beat.music_beat_activations(stimulus, checkpoint=str(checkpoint), **kwargs)
    assert state["loads"] == 0


@pytest.mark.parametrize("failure", ["bad_shape", "fail_inference"])
def test_backend_failures_are_typed(backend, failure):
    state, checkpoint = backend
    state[failure] = True
    with pytest.raises(BackendInferenceError):
        beat.music_beat_activations(_audio(), checkpoint=str(checkpoint))


def test_corrupt_local_checkpoint_is_load_error(backend):
    _, checkpoint = backend
    checkpoint.write_bytes(b"bad")
    with pytest.raises(BackendLoadError):
        beat.music_beat_activations(_audio(), checkpoint=str(checkpoint))
