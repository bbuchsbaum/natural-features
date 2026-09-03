from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
    OptionalBackendError,
)
from natural_features.core.stimulus import AudioStimulus, ImageStimulus
from natural_features.features.audio.neural import audio_clap_embeddings
from natural_features.features.language.providers import OpenAIEmbeddingProvider
from natural_features.features.language.syntax import syntactic_features
from natural_features.features.preprocess import text_tokenize
from natural_features.features.vision.neural import vision_clip_embeddings


def _audio() -> AudioStimulus:
    return AudioStimulus.from_array(np.zeros(8, dtype=np.float32), sr_hz=8)


def test_missing_backend_dependency_has_a_distinct_error(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(BackendDependencyError) as raised:
        audio_clap_embeddings(_audio())

    assert raised.value.phase == "dependency"
    assert raised.value.backend == "CLAP"
    assert isinstance(raised.value, OptionalBackendError)


def test_model_load_failure_is_not_reported_as_inference(monkeypatch) -> None:  # noqa: ANN001
    class FailsToLoad:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> None:
            raise OSError("missing local weights")

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = FailsToLoad
    transformers.ClapModel = FailsToLoad
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(BackendLoadError) as raised:
        audio_clap_embeddings(_audio(), model="missing")

    assert raised.value.phase == "load"
    assert isinstance(raised.value.__cause__, OSError)


def test_evaluation_failure_is_not_reported_as_loading(monkeypatch) -> None:  # noqa: ANN001
    class Loads:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "Loads":
            return cls()

        def __call__(self, **_kwargs: object) -> dict[str, object]:
            raise ValueError("bad waveform")

        def get_audio_features(self, **_kwargs: object) -> object:
            raise AssertionError("processor should fail first")

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = Loads
    transformers.ClapModel = Loads
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(BackendInferenceError) as raised:
        audio_clap_embeddings(_audio(), model="loaded")

    assert raised.value.phase == "inference"
    assert isinstance(raised.value.__cause__, ValueError)


def test_other_named_adapters_share_the_dependency_contract(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", None)
    image = ImageStimulus.from_array(np.zeros((4, 4, 3), dtype=np.uint8))

    with pytest.raises(BackendDependencyError):
        vision_clip_embeddings(image)

    monkeypatch.setitem(sys.modules, "spacy", None)
    with pytest.raises(BackendDependencyError):
        syntactic_features(text_tokenize("one two"))


def test_provider_configuration_failure_is_a_load_error(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("NF_TEST_MISSING_OPENAI_KEY", raising=False)

    with pytest.raises(BackendLoadError, match="missing API key"):
        OpenAIEmbeddingProvider(api_key_env_var="NF_TEST_MISSING_OPENAI_KEY")
