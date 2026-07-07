from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.execution import resolve_execution_mode
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.emotion import speech_emotion
from natural_features.features.speech.phonology import ctc_phone_posteriors
from natural_features.features.speech.vad import neural_vad
from natural_features.workflows.multiscale_language import extract_multiscale_language


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    x = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_resolve_execution_mode_defaults_and_conflicts() -> None:
    mode, strict = resolve_execution_mode()
    assert mode == "fallback"
    assert strict is False

    mode, strict = resolve_execution_mode(strict_dependency=True)
    assert mode == "strict"
    assert strict is True

    with pytest.raises(ValueError):
        resolve_execution_mode(execution_mode="strict", strict_dependency=False)


def test_asr_metadata_has_execution_mode() -> None:
    out = whisper_transcribe(_audio(), execution_mode="fallback")
    assert out["words"].metadata.get("execution_mode") == "fallback"
    assert "fallback_used" in out["words"].metadata


def test_ctc_posteriors_mark_fallback_provenance() -> None:
    post = ctc_phone_posteriors(
        _audio(),
        model="__missing__/__missing__",
        local_files_only=True,
        execution_mode="fallback",
    )
    assert post.metadata.get("execution_mode") == "fallback"
    assert post.metadata.get("fallback_used") is True


def test_ctc_posteriors_strict_mode_fails_loudly() -> None:
    with pytest.raises(RuntimeError):
        ctc_phone_posteriors(
            _audio(),
            model="__missing__/__missing__",
            local_files_only=True,
            execution_mode="strict",
        )


def test_speech_emotion_strict_mode_uses_transformers_backend(monkeypatch) -> None:  # noqa: ANN001
    class FakeTensor:
        def __init__(self, array: np.ndarray):
            self.array = np.asarray(array, dtype=np.float32)

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self.array

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> None:
            return None

    class FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-emotion"
            assert local_files_only is True
            return cls()

        def __call__(self, wav: np.ndarray, sampling_rate: int, return_tensors: str) -> dict[str, np.ndarray]:
            assert wav.ndim == 1
            assert sampling_rate == 8000
            assert return_tensors == "pt"
            return {"input_values": wav[None, :]}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "calm", 1: "excited", 2: "sad"})

        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-emotion"
            assert local_files_only is True
            return cls()

        def __call__(self, **_inputs: object) -> object:
            return SimpleNamespace(logits=FakeTensor(np.asarray([[0.0, 2.0, -1.0]], dtype=np.float32)))

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: NoGrad()
    transformers = types.ModuleType("transformers")
    transformers.AutoFeatureExtractor = FakeFeatureExtractor
    transformers.AutoModelForAudioClassification = FakeModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    out = speech_emotion(
        _audio(),
        model="fake-emotion",
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    assert out.metadata["backend"] == "transformers_audio_classification"
    assert out.metadata["fallback_used"] is False
    assert out.coords["feature"] == ["calm", "excited", "sad"]
    assert out.values.shape == (1, 3)
    np.testing.assert_allclose(out.values.sum(axis=1), np.asarray([1.0], dtype=np.float32))


def test_neural_vad_strict_mode_uses_silero_backend(monkeypatch) -> None:  # noqa: ANN001
    class FakeTorchTensor:
        def __init__(self, array: np.ndarray):
            self.array = np.asarray(array, dtype=np.float32)

    class FakeProb:
        def __init__(self, value: float):
            self.value = float(value)

        def item(self) -> float:
            return self.value

    class FakeSileroModel:
        def __init__(self) -> None:
            self.calls = 0
            self.reset_count = 0

        def reset_states(self) -> None:
            self.reset_count += 1

        def __call__(self, chunk: FakeTorchTensor, sr_hz: int) -> FakeProb:
            assert sr_hz == 8000
            assert chunk.array.shape == (256,)
            self.calls += 1
            return FakeProb(min(0.95, 0.05 + 0.01 * self.calls))

    fake_model = FakeSileroModel()
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda array: FakeTorchTensor(array)
    silero_vad = types.ModuleType("silero_vad")
    silero_vad.load_silero_vad = lambda: fake_model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "silero_vad", silero_vad)

    out = neural_vad(
        _audio(),
        model="silero_vad",
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    assert out.metadata["backend"] == "silero_vad_package"
    assert out.metadata["fallback_used"] is False
    assert out.metadata["model_sample_rate_hz"] == 8000
    assert out.values.shape == (100, 1)
    assert fake_model.calls > 0
    assert fake_model.reset_count == 2
    assert np.logical_and(out.values >= 0.0, out.values <= 1.0).all()


def test_multiscale_language_provider_fallback_when_openai_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = extract_multiscale_language(
        "hello world this is a fallback provider test",
        scales_s=[2.0],
        provider_config={"provider": "openai", "model": "text-embedding-3-large"},
        execution_mode="fallback",
    )
    prov = res.qc["provider_resolution"]
    assert prov["requested_provider"] == "openai"
    assert prov["resolved_provider"] == "local_bow"
    assert prov["fallback_used"] is True


def test_multiscale_language_provider_strict_mode_fails(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        extract_multiscale_language(
            "hello world",
            scales_s=[2.0],
            provider_config={"provider": "openai", "model": "text-embedding-3-large"},
            execution_mode="strict",
        )
