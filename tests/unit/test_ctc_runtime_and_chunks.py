from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.phonology import (
    CTCModelRuntime,
    clear_ctc_runtime,
    ctc_phone_posteriors,
    load_ctc_runtime,
)


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)

    def to(self, _device: object) -> FakeTensor:
        return self

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self.array)

    def __getitem__(self, idx: object) -> FakeTensor:
        return FakeTensor(self.array[idx])


class _Context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


class FakeCTCProcessor:
    n_pretrained = 0
    sampling_rate = 8000

    def __init__(self) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=self.sampling_rate)
        self.tokenizer = SimpleNamespace(
            convert_ids_to_tokens=lambda index: ["aa", "bb", "cc"][int(index)]
        )

    @classmethod
    def from_pretrained(cls, model: str, local_files_only: bool = True) -> FakeCTCProcessor:
        cls.n_pretrained += 1
        inst = cls()
        inst.model_id = model
        inst.local_files_only = local_files_only
        return inst

    def __call__(
        self,
        wav: np.ndarray,
        sampling_rate: int,
        return_tensors: str = "pt",
    ) -> dict[str, FakeTensor]:
        assert return_tensors == "pt"
        assert sampling_rate == self.sampling_rate
        return {"input_values": FakeTensor(np.asarray(wav, dtype=np.float32))}


class FakeCTCModel:
    n_pretrained = 0
    n_forward = 0
    input_lengths: ClassVar[list[int]] = []
    fail_load = False
    fail_forward: Exception | None = None
    peak_class_by_call: ClassVar[list[int]] = []

    def __init__(self) -> None:
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, model: str, local_files_only: bool = True) -> FakeCTCModel:
        if cls.fail_load:
            raise OSError("missing local weights")
        cls.n_pretrained += 1
        inst = cls()
        inst.model_id = model
        inst.local_files_only = local_files_only
        return inst

    def to(self, device: str) -> FakeCTCModel:
        self.device = device
        return self

    def eval(self) -> FakeCTCModel:
        return self

    def __call__(self, **inputs: object) -> SimpleNamespace:
        if self.fail_forward is not None:
            raise self.fail_forward
        type(self).n_forward += 1
        values = inputs["input_values"]
        arr = values.array if isinstance(values, FakeTensor) else np.asarray(values)
        n_samples = int(arr.shape[-1])
        type(self).input_lengths.append(n_samples)
        hop = max(1, round(0.02 * FakeCTCProcessor.sampling_rate))
        n_frames = max(1, n_samples // hop)
        peak = len(type(self).peak_class_by_call)
        type(self).peak_class_by_call.append(peak % 3)
        logits = np.full((1, n_frames, 3), -5.0, dtype=np.float32)
        logits[0, :, peak % 3] = 5.0
        return SimpleNamespace(logits=FakeTensor(logits))


def _audio(duration_s: float = 0.5, sr: int = 8000) -> AudioStimulus:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def _softmax(tensor: FakeTensor | np.ndarray, dim: int = -1) -> FakeTensor:
    arr = np.asarray(tensor.array if isinstance(tensor, FakeTensor) else tensor, dtype=np.float64)
    shifted = arr - np.max(arr, axis=dim, keepdims=True)
    exp = np.exp(shifted)
    return FakeTensor((exp / np.sum(exp, axis=dim, keepdims=True)).astype(np.float32))


def _install_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool = False,
    mps: bool = False,
) -> None:
    FakeCTCProcessor.n_pretrained = 0
    FakeCTCModel.n_pretrained = 0
    FakeCTCModel.n_forward = 0
    FakeCTCModel.input_lengths = []
    FakeCTCModel.fail_load = False
    FakeCTCModel.fail_forward = None
    FakeCTCModel.peak_class_by_call = []

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    torch.no_grad = lambda: _Context()
    torch.inference_mode = lambda: _Context()
    torch.softmax = _softmax
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = FakeCTCProcessor
    transformers.AutoModelForCTC = FakeCTCModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


@pytest.fixture(autouse=True)
def _reset_ctc_runtime() -> None:
    clear_ctc_runtime()
    yield
    clear_ctc_runtime()


def test_ctc_runtime_is_reused_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch, cuda=False, mps=False)
    audio = _audio(duration_s=0.4)
    first = ctc_phone_posteriors(audio, model="fake-ctc", device="cpu", local_files_only=True)
    second = ctc_phone_posteriors(audio, model="fake-ctc", device="cpu", local_files_only=True)
    assert FakeCTCModel.n_pretrained == 1
    assert FakeCTCProcessor.n_pretrained == 1
    assert first.metadata["device"] == "cpu"
    assert second.metadata["backend"] == "transformers_ctc"
    other = ctc_phone_posteriors(audio, model="fake-ctc-b", device="cpu", local_files_only=True)
    assert FakeCTCModel.n_pretrained == 2
    assert other.metadata["backend"] == "transformers_ctc"


def test_device_auto_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _audio(duration_s=0.3)
    _install_backend(monkeypatch, cuda=True, mps=True)
    cuda_out = ctc_phone_posteriors(audio, model="fake-ctc", device="auto", local_files_only=True)
    assert cuda_out.metadata["device"] == "cuda"
    clear_ctc_runtime()

    _install_backend(monkeypatch, cuda=False, mps=True)
    mps_out = ctc_phone_posteriors(audio, model="fake-ctc", device="auto", local_files_only=True)
    assert mps_out.metadata["device"] == "mps"
    clear_ctc_runtime()

    _install_backend(monkeypatch, cuda=False, mps=False)
    cpu_out = ctc_phone_posteriors(audio, model="fake-ctc", device="auto", local_files_only=True)
    assert cpu_out.metadata["device"] == "cpu"


def test_explicit_cuda_unavailable_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        ctc_phone_posteriors(
            _audio(),
            model="fake-ctc",
            device="cuda",
            local_files_only=True,
            execution_mode="strict",
        )


def test_short_audio_uses_one_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch)
    out = ctc_phone_posteriors(
        _audio(duration_s=1.0),
        model="fake-ctc",
        device="cpu",
        local_files_only=True,
        chunk_window_s=2.0,
        chunk_overlap_s=0.5,
    )
    assert FakeCTCModel.n_forward == 1
    assert out.metadata["chunk_count"] == 1
    assert out.values.shape[0] == len(out.times_s)
    assert np.all(np.diff(out.times_s) >= 0)


def test_long_audio_stitches_interior_overlap_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch)
    out = ctc_phone_posteriors(
        _audio(duration_s=3.0),
        model="fake-ctc",
        device="cpu",
        local_files_only=True,
        chunk_window_s=2.0,
        chunk_overlap_s=1.0,
    )
    assert FakeCTCModel.n_forward == 2
    assert out.metadata["chunk_count"] == 2
    assert np.all(np.diff(out.times_s) >= 0)
    assert out.times_s[0] == pytest.approx(0.0)
    assert out.times_s[-1] < 3.0
    labels = list(out.coords["feature"])
    aa = labels.index("aa")
    bb = labels.index("bb")
    first_region = out.times_s < 1.5
    second_region = out.times_s >= 1.5
    assert np.any(first_region)
    assert np.any(second_region)
    assert np.all(out.values[first_region, aa] > 0.9)
    assert np.all(out.values[second_region, bb] > 0.9)


def test_load_failure_is_not_reported_as_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch)
    FakeCTCModel.fail_load = True
    with pytest.raises(RuntimeError, match="unavailable") as raised:
        ctc_phone_posteriors(
            _audio(),
            model="missing-ctc",
            device="cpu",
            local_files_only=True,
            execution_mode="strict",
        )
    assert "inference failed" not in str(raised.value)
    assert isinstance(raised.value.__cause__, OSError)


def test_evaluation_failure_is_not_reported_as_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch)
    FakeCTCModel.fail_forward = ValueError("bad waveform")
    with pytest.raises(RuntimeError, match="CTC inference failed") as raised:
        ctc_phone_posteriors(
            _audio(),
            model="fake-ctc",
            device="cpu",
            local_files_only=True,
            execution_mode="strict",
        )
    assert "unavailable" not in str(raised.value)
    assert isinstance(raised.value.__cause__, ValueError)


def test_oom_hint_mentions_chunk_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch)
    FakeCTCModel.fail_forward = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="chunk_window_s"):
        ctc_phone_posteriors(
            _audio(),
            model="fake-ctc",
            device="cpu",
            local_files_only=True,
            execution_mode="strict",
        )


def test_explicit_runtime_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_backend(monkeypatch)
    runtime = load_ctc_runtime(model="fake-ctc", local_files_only=True, device="cpu")
    assert isinstance(runtime, CTCModelRuntime)
    loaded = FakeCTCModel.n_pretrained
    ctc_phone_posteriors(
        _audio(),
        model="fake-ctc",
        runtime=runtime,
        device="cpu",
        local_files_only=True,
    )
    assert FakeCTCModel.n_pretrained == loaded
    ctc_phone_posteriors(
        _audio(),
        model="fake-ctc",
        device="cpu",
        local_files_only=True,
    )
    assert FakeCTCModel.n_pretrained == loaded + 1
