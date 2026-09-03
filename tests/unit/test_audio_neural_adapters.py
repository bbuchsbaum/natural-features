from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.neural import audio_ast_embeddings, audio_clap_embeddings


class _Tensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def _audio() -> AudioStimulus:
    return AudioStimulus.from_array(
        np.linspace(-0.5, 0.5, 8, dtype=np.float32),
        sr_hz=8,
        start_offset_s=2.0,
    )


def _install_audio_models(monkeypatch) -> dict[str, int]:  # noqa: ANN001
    calls = {"clap": 0, "ast": 0}

    class Processor:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "Processor":
            return cls()

        def __call__(self, *args: object, **kwargs: object) -> dict[str, _Tensor]:
            # transformers renamed this keyword from `audios` to `audio` in v5 and the
            # extractor now tries the new name first, so accept either.
            waveform = kwargs.get("audio", kwargs.get("audios", args[0] if args else None))
            assert np.asarray(waveform).shape == (8,)
            assert kwargs["sampling_rate"] == 8
            return {"input_values": _Tensor([[1.0]])}

    class Clap:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "Clap":
            return cls()

        def get_audio_features(self, **_inputs: object) -> _Tensor:
            calls["clap"] += 1
            return _Tensor([[1.0, 2.0, 3.0]])

    class AST:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "AST":
            return cls()

        def __call__(self, **_inputs: object) -> object:
            calls["ast"] += 1
            return SimpleNamespace(pooler_output=_Tensor([[4.0, 5.0]]))

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: _NoGrad()
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = Processor
    transformers.ClapModel = Clap
    transformers.AutoFeatureExtractor = Processor
    transformers.ASTModel = AST
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def _assert_clip_contract(result, expected: np.ndarray) -> None:  # noqa: ANN001
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(result.times_s, np.asarray([2.0]))
    np.testing.assert_array_equal(result.time_bounds_s, np.asarray([[2.0, 3.0]]))
    assert result.timebase.kind == "audio_summary"
    assert result.timebase.support.kind == "interval"
    assert result.metadata["temporal_scope"] == "clip"


def test_clap_uses_native_audio_projection_and_clip_interval(monkeypatch) -> None:  # noqa: ANN001
    calls = _install_audio_models(monkeypatch)

    result = audio_clap_embeddings(_audio(), model="fake-clap")

    _assert_clip_contract(result, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert calls == {"clap": 1, "ast": 0}
    assert result.metadata["representation"] == "audio_projection"


def test_ast_uses_native_pooler_output_and_clip_interval(monkeypatch) -> None:  # noqa: ANN001
    calls = _install_audio_models(monkeypatch)

    result = audio_ast_embeddings(_audio(), model="fake-ast")

    _assert_clip_contract(result, np.asarray([[4.0, 5.0]], dtype=np.float32))
    assert calls == {"clap": 0, "ast": 1}
    assert result.metadata["representation"] == "pooler_output"
