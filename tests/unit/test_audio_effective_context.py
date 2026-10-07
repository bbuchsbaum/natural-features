from __future__ import annotations
import sys
import types
from types import SimpleNamespace
import numpy as np
import pytest
from natural_features.core.backend_errors import BackendInferenceError
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.neural import (
    audio_ast_embeddings,
    audio_clap_embeddings,
)


class Tensor:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.x


def install(monkeypatch):
    seen = []

    class Processor:
        sampling_rate = 1000
        nb_max_samples = 1000
        max_length = 98

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, *args, **kwargs):
            wave = np.asarray(
                kwargs.get("audio", kwargs.get("audios", args[0] if args else None))
            )
            capacity = 1000 if "truncation" in kwargs else 995
            assert len(wave) <= capacity, "hidden processor truncation must never occur"
            seen.append((wave.copy(), kwargs.copy()))
            return {"x": Tensor(wave)}

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            self.training = False
            return self

        def embedding(self, x):
            assert self.training is False
            v = x.numpy()
            return Tensor([[v[0], v[-1], v.mean(), v.std()]])

        def get_audio_features(self, x):
            return self.embedding(x)

        def __call__(self, x):
            return SimpleNamespace(pooler_output=self.embedding(x))

    torch = types.ModuleType("torch")

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    torch.no_grad = NoGrad
    tf = types.ModuleType("transformers")
    tf.AutoProcessor = Processor
    tf.AutoFeatureExtractor = Processor
    tf.ClapModel = Model
    tf.ASTModel = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    return seen, Processor


def stimulus(n):
    return AudioStimulus.from_array(
        np.arange(n, dtype=np.float32) / 2400, sr_hz=1000, start_offset_s=3.25
    )


@pytest.mark.parametrize("fn", [audio_clap_embeddings, audio_ast_embeddings])
def test_long_audio_is_rejected_before_inference(monkeypatch, fn):
    seen, _ = install(monkeypatch)
    with pytest.raises(ValueError, match="Split the audio explicitly"):
        fn(stimulus(2400))
    assert seen == []


@pytest.mark.parametrize(
    "fn,capacity", [(audio_clap_embeddings, 1000), (audio_ast_embeddings, 995)]
)
@pytest.mark.parametrize("policy", ["start_crop", "center_crop"])
def test_explicit_crop_bytes_clock_and_context_are_exact(
    monkeypatch, fn, capacity, policy
):
    seen, _ = install(monkeypatch)
    original = stimulus(2400)
    start = 0 if policy == "start_crop" else (2400 - capacity) // 2
    expected = original.samples[start : start + capacity]
    first = fn(original, context_policy=policy)
    np.random.seed(8)
    second = fn(original, context_policy=policy)
    np.testing.assert_array_equal(seen[0][0], expected)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_allclose(
        first.values, [[expected[0], expected[-1], expected.mean(), expected.std()]]
    )
    np.testing.assert_array_equal(first.times_s, [3.25 + start / 1000])
    np.testing.assert_array_equal(
        first.time_bounds_s, [[3.25 + start / 1000, 3.25 + (start + capacity) / 1000]]
    )
    ctx = first.metadata["computation_context"]
    assert ctx["clock"] == str(original.clock)
    assert ctx["input_bounds_s"] == [3.25, 5.65]
    assert ctx["dependency_bounds_s"] == first.time_bounds_s.tolist()
    assert ctx["preprocessing"]["crop_start_sample"] == start
    assert ctx["preprocessing"]["used_samples"] == capacity


@pytest.mark.parametrize("fn", [audio_clap_embeddings, audio_ast_embeddings])
def test_short_input_keeps_all_source_samples_and_offset(monkeypatch, fn):
    seen, _ = install(monkeypatch)
    original = stimulus(200)
    result = fn(original)
    np.testing.assert_array_equal(seen[0][0], original.samples)
    np.testing.assert_array_equal(result.time_bounds_s, [[3.25, 3.45]])
    assert result.metadata["computation_context"]["policy"] == "bidirectional"


@pytest.mark.parametrize("padding", ["repeat", "repeatpad", "pad"])
def test_clap_padding_choice_is_explicit_and_recorded(monkeypatch, padding):
    seen, _ = install(monkeypatch)
    result = audio_clap_embeddings(stimulus(200), padding=padding)
    assert seen[0][1]["padding"] == padding
    assert result.metadata["computation_context"]["preprocessing"]["padding"] == padding


@pytest.mark.parametrize(
    "fn,attr",
    [(audio_clap_embeddings, "nb_max_samples"), (audio_ast_embeddings, "max_length")],
)
def test_unknown_capacity_is_not_guessed(monkeypatch, fn, attr):
    _, processor = install(monkeypatch)
    delattr(processor, attr)
    with pytest.raises(BackendInferenceError, match="declare"):
        fn(stimulus(200))


@pytest.mark.parametrize("fn", [audio_clap_embeddings, audio_ast_embeddings])
def test_invalid_crop_policy_fails(monkeypatch, fn):
    install(monkeypatch)
    with pytest.raises(ValueError, match="context_policy"):
        fn(stimulus(200), context_policy="random")


def test_invalid_padding_fails_without_backend_load():
    with pytest.raises(ValueError, match="padding"):
        audio_clap_embeddings(stimulus(200), padding="mystery")
