from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

from natural_features.core.stimulus import ImageStimulus, VideoStimulus
from natural_features.features.vision.face import face_detection
from natural_features.features.vision.neural import vision_clip_embeddings, vision_dino_embeddings
from natural_features.features.vision.semantic import vision_semantic_views


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array, dtype=np.float32)

    def to(self, _device: object) -> "_FakeTensor":
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def _install_fake_torch(monkeypatch) -> None:  # noqa: ANN001
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: _NoGrad()
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_mediapipe_face_backend_with_fake_module(monkeypatch) -> None:  # noqa: ANN001
    class FakeDetector:
        def __init__(self, model_selection: int, min_detection_confidence: float):
            assert model_selection == 0
            assert min_detection_confidence == 0.5

        def process(self, _frame: np.ndarray) -> object:
            bbox = SimpleNamespace(xmin=0.2, ymin=0.3, width=0.4, height=0.2)
            det = SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=bbox))
            return SimpleNamespace(detections=[det])

        def close(self) -> None:
            return None

    mp = types.ModuleType("mediapipe")
    mp.solutions = SimpleNamespace(face_detection=SimpleNamespace(FaceDetection=FakeDetector))
    monkeypatch.setitem(sys.modules, "mediapipe", mp)

    stim = ImageStimulus.from_array(np.ones((8, 8, 3), dtype=np.float32))
    out = face_detection(stim, execution_mode="strict", strict_dependency=True)

    assert out.values.shape == (1, 5)
    np.testing.assert_allclose(out.values[0], np.array([1.0, 1.0, 0.08, 0.4, 0.4], dtype=np.float32))
    assert out.metadata["backend"] == "mediapipe"
    assert out.metadata["fallback_used"] is False


def test_clip_backend_with_fake_transformers(monkeypatch) -> None:  # noqa: ANN001
    _install_fake_torch(monkeypatch)

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-clip"
            assert local_files_only
            return cls()

        def __call__(self, images: list[object], return_tensors: str) -> dict[str, _FakeTensor]:
            assert return_tensors == "pt"
            return {"pixel_values": _FakeTensor(np.zeros((len(images), 3, 2, 2), dtype=np.float32))}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-clip"
            assert local_files_only
            return cls()

        def to(self, _device: object) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

        def get_image_features(self, pixel_values: _FakeTensor) -> _FakeTensor:
            batch = pixel_values.array.shape[0]
            return _FakeTensor(np.tile(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), (batch, 1)))

    transformers = types.ModuleType("transformers")
    transformers.CLIPProcessor = FakeProcessor
    transformers.CLIPModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    stim = ImageStimulus.from_array(np.ones((4, 4, 3), dtype=np.float32))
    out = vision_clip_embeddings(stim, model="fake-clip", dim=3, execution_mode="strict", strict_dependency=True)

    assert out.values.shape == (1, 3)
    np.testing.assert_allclose(out.values[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert out.metadata["backend"] == "transformers_clip"


def test_semantic_views_backend_with_fake_clip(monkeypatch) -> None:  # noqa: ANN001
    _install_fake_torch(monkeypatch)

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-clip"
            assert local_files_only
            return cls()

        def __call__(
            self,
            text: list[str],
            images: list[object],
            return_tensors: str,
            padding: bool,
        ) -> dict[str, _FakeTensor]:
            assert text == ["a video frame showing cat", "a video frame showing dog"]
            assert return_tensors == "pt"
            assert padding is True
            return {
                "pixel_values": _FakeTensor(np.zeros((len(images), 3, 2, 2), dtype=np.float32)),
                "input_ids": _FakeTensor(np.zeros((len(text), 4), dtype=np.float32)),
            }

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-clip"
            assert local_files_only
            return cls()

        def to(self, _device: object) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

        def __call__(self, pixel_values: _FakeTensor, input_ids: _FakeTensor) -> object:
            assert pixel_values.array.shape[0] == 2
            assert input_ids.array.shape[0] == 2
            return SimpleNamespace(logits_per_image=_FakeTensor(np.asarray([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)))

    transformers = types.ModuleType("transformers")
    transformers.CLIPProcessor = FakeProcessor
    transformers.CLIPModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    stim = VideoStimulus.from_array(np.ones((2, 4, 4, 3), dtype=np.float32), fps=2.0)
    out = vision_semantic_views(
        stim,
        model="fake-clip",
        labels=["cat", "dog"],
        execution_mode="strict",
        strict_dependency=True,
    )

    assert out.metadata["backend"] == "transformers_clip_zero_shot"
    assert out.metadata["fallback_used"] is False
    assert out.label.tolist() == ["dog", "cat"]
    assert out.extra["label_index"].tolist() == [1, 0]
    assert out.confidence is not None
    assert np.all(out.confidence > 0.8)


def test_dino_backend_with_fake_transformers(monkeypatch) -> None:  # noqa: ANN001
    _install_fake_torch(monkeypatch)

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-dino"
            assert local_files_only
            return cls()

        def __call__(self, images: list[object], return_tensors: str) -> dict[str, _FakeTensor]:
            assert return_tensors == "pt"
            return {"pixel_values": _FakeTensor(np.zeros((len(images), 3, 2, 2), dtype=np.float32))}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model: str, local_files_only: bool):  # noqa: ANN102, ANN202
            assert model == "fake-dino"
            assert local_files_only
            return cls()

        def to(self, _device: object) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

        def __call__(self, pixel_values: _FakeTensor, output_hidden_states: bool) -> object:
            assert output_hidden_states
            batch = pixel_values.array.shape[0]
            h0 = np.zeros((batch, 2, 4), dtype=np.float32)
            h1 = np.ones((batch, 2, 4), dtype=np.float32)
            h2 = np.full((batch, 2, 4), 2.0, dtype=np.float32)
            return SimpleNamespace(hidden_states=[_FakeTensor(h0), _FakeTensor(h1), _FakeTensor(h2)])

    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = FakeProcessor
    transformers.AutoModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    stim = VideoStimulus.from_array(np.ones((2, 4, 4, 3), dtype=np.float32), fps=2.0)
    out = vision_dino_embeddings(
        stim,
        model="fake-dino",
        layers=[1, 2],
        dim=2,
        execution_mode="strict",
        strict_dependency=True,
    )

    assert out.values.shape == (2, 4)
    np.testing.assert_allclose(out.values[0], np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
    assert out.metadata["backend"] == "transformers_dino"
