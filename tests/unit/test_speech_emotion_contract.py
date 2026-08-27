from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import numpy as np

from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.emotion import speech_emotion


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


def test_speech_emotion_is_explicitly_clip_level(monkeypatch) -> None:  # noqa: ANN001
    class FeatureExtractor:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FeatureExtractor":
            return cls()

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, _Tensor]:
            return {"input_values": _Tensor([[0.0]])}

    class Model:
        config = SimpleNamespace(id2label={0: "calm", 1: "excited"})

        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "Model":
            return cls()

        def __call__(self, **_inputs: object) -> object:
            return SimpleNamespace(logits=_Tensor([[0.0, np.log(3.0)]]))

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: _NoGrad()
    transformers = types.ModuleType("transformers")
    transformers.AutoFeatureExtractor = FeatureExtractor
    transformers.AutoModelForAudioClassification = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    stimulus = AudioStimulus.from_array(
        np.zeros(16, dtype=np.float32),
        sr_hz=8,
        start_offset_s=5.0,
    )

    result = speech_emotion(stimulus, model="fake-emotion")

    np.testing.assert_allclose(result.values, np.asarray([[0.25, 0.75]], dtype=np.float32))
    np.testing.assert_array_equal(result.time_bounds_s, np.asarray([[5.0, 7.0]]))
    assert result.timebase.support.kind == "interval"
    assert result.metadata["temporal_scope"] == "clip"
    assert "hop_s" not in inspect.signature(speech_emotion).parameters
    assert "hop_s" not in Registry.with_builtin_specs().get("speech.emotion").params
