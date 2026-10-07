"""Deterministic contracts for the strict MERT adapter; no model download occurs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.backend_errors import BackendInferenceError, BackendLoadError
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.mert import audio_music_mert


class _Tensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value, dtype=np.float32)
        self.devices: list[str] = []

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value

    def to(self, device: str) -> "_Tensor":
        self.devices.append(device)
        return self


class _InferenceMode:
    def __init__(self, calls: dict[str, int]) -> None:
        self.calls = calls

    def __enter__(self) -> None:
        self.calls["inference"] += 1

    def __exit__(self, *_exc: object) -> None:
        return None


def _snapshot(tmp_path, *, auto_map: bool = False):  # noqa: ANN001
    root = tmp_path / "mert-snapshot"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "x.Model"}} if auto_map else {}))
    return root


def _audio(n_samples: int = 13) -> AudioStimulus:
    return AudioStimulus.from_array(
        np.linspace(-0.5, 0.5, n_samples, dtype=np.float32), sr_hz=8, start_offset_s=1.0
    )


def _install_fake_backend(  # noqa: ANN001
    monkeypatch, *, bad_shape: bool = False, config_overrides=None, processor_rate: int | None = None,
    expect_trust: bool = False, load_error: Exception | None = None, fail_eval: bool = False,
    fail_inference: bool = False, output_frames: int = 3,
):
    calls = {"inference": 0, "eval": 0, "processor": 0, "model": 0}
    config_values = {
        "sample_rate": 8,
        "conv_kernel": [3, 2],
        "conv_stride": [2, 2],
        "model_type": "mert_model",
        "num_hidden_layers": 2,
        "hidden_size": 2,
        "feature_extractor_cqt": False,
        "add_adapter": False,
    }
    config_values.update(config_overrides or {})

    class Processor:
        do_normalize = True

        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "Processor":
            if load_error is not None:
                raise load_error
            assert kwargs == {"local_files_only": True, "trust_remote_code": expect_trust}
            return cls()

        feature_extractor = SimpleNamespace(sampling_rate=processor_rate or config_values["sample_rate"])

        def __call__(self, waveform: np.ndarray, **kwargs: object) -> dict[str, _Tensor]:
            calls["processor"] += 1
            assert kwargs == {"sampling_rate": config_values["sample_rate"], "return_tensors": "pt"}
            return {"input_values": _Tensor(np.asarray(waveform)[None, :])}

    class Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(**config_values)

        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "Model":
            assert kwargs == {"local_files_only": True, "trust_remote_code": expect_trust}
            return cls()

        def to(self, device: str) -> "Model":
            assert device == "cpu"
            return self

        def eval(self) -> "Model":
            if fail_eval:
                raise ValueError("checkpoint evaluation setup failed")
            calls["eval"] += 1
            return self

        def __call__(self, **kwargs: object) -> object:
            calls["model"] += 1
            assert kwargs["output_hidden_states"] is True
            if fail_inference:
                raise RuntimeError("bad compute")
            n_time = 2 if bad_shape else output_frames
            # Every layer carries a distinct marker, so layer selection order is observable.
            hidden = tuple(_Tensor(np.full((1, n_time, 2), marker, dtype=np.float32)) for marker in (10, 20, 30))
            return SimpleNamespace(hidden_states=hidden)

    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: _InferenceMode(calls)
    transformers = types.ModuleType("transformers")
    transformers.AutoFeatureExtractor = Processor
    transformers.AutoModel = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_native_cells_use_hand_enumerated_valid_convolution_geometry(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Kernels [3,2], strides [2,2] yield R=5, S=4 and starts [0,4,8] for N=13."""

    calls = _install_fake_backend(monkeypatch)
    result = audio_music_mert(_audio(), model=_snapshot(tmp_path), layers=[2, 0])

    np.testing.assert_array_equal(result.values[:, 0, :], np.full((3, 2), 30, dtype=np.float32))
    np.testing.assert_array_equal(result.values[:, 1, :], np.full((3, 2), 10, dtype=np.float32))
    np.testing.assert_allclose(result.times_s, [1.25, 1.75, 2.25])
    np.testing.assert_allclose(result.time_bounds_s, [[1.0, 1.625], [1.5, 2.125], [2.0, 2.625]])
    assert result.coords["layer"] == [2, 0]
    assert result.metadata["receptive_field_samples"] == 5
    assert result.metadata["stride_samples"] == 4
    assert result.metadata["computation_context"]["policy"] == "bidirectional"
    assert result.metadata["computation_context"]["dependency_bounds_s"] == [[1.0, 2.625]]
    assert result.metadata["computation_context"]["preprocessing"]["processor_normalization"] is True
    assert calls == {"inference": 1, "eval": 1, "processor": 1, "model": 1}


@pytest.mark.parametrize("layers", [[], [0, 0], [True], [3], [-1], [0.0]])
def test_layer_selection_is_strict(monkeypatch, tmp_path, layers) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch)
    with pytest.raises(ValueError):
        audio_music_mert(_audio(), model=_snapshot(tmp_path), layers=layers)


def test_short_input_refuses_instead_of_padding(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch)
    with pytest.raises(ValueError, match="requires at least 5"):
        audio_music_mert(_audio(4), model=_snapshot(tmp_path))


@pytest.mark.parametrize("samples,frames", [(5, 1), (8, 1), (9, 2), (12, 2), (13, 3), (16, 3)])
def test_positive_convolution_length_boundaries_keep_only_complete_cells(
    monkeypatch, tmp_path, samples, frames,
) -> None:  # noqa: ANN001
    # Enumerated physical cells for R=5, S=4: [0,5), [4,9), [8,13).
    _install_fake_backend(monkeypatch, output_frames=frames)
    result = audio_music_mert(_audio(samples), model=_snapshot(tmp_path))
    assert result.values.shape == (frames, 3, 2)
    np.testing.assert_allclose(result.time_bounds_s[-1],
                               1 + np.array([(frames - 1) * 4, (frames - 1) * 4 + 5]) / 8)
    assert result.time_bounds_s[-1, 1] <= 1 + samples / 8


def test_inconsistent_output_length_refuses_instead_of_truncating(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, bad_shape=True)
    with pytest.raises(BackendInferenceError, match=r"expected \(1, 3, 2\)"):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))


def test_remote_identifier_requires_an_immutable_revision() -> None:
    with pytest.raises(ValueError, match="40-hex"):
        audio_music_mert(_audio(), model="m-a-p/MERT-v1-95M")
    with pytest.raises(ValueError, match="40-hex"):
        audio_music_mert(_audio(), model="m-a-p/MERT-v1-95M", revision="main")


def test_custom_code_is_refused_before_importing_optional_backends(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(BackendLoadError, match="trust_remote_code=True") as raised:
        audio_music_mert(_audio(), model=_snapshot(tmp_path, auto_map=True))
    assert raised.value.phase == "load"


def test_custom_code_can_run_only_after_explicit_opt_in(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, expect_trust=True)
    result = audio_music_mert(_audio(), model=_snapshot(tmp_path, auto_map=True), trust_remote_code=True)
    assert result.coords["layer"] == [0, 1, 2]


@pytest.mark.parametrize(
    ("filename", "auto_map"),
    [
        ("config.json", {"AutoConfig": "other/source--configuration.MERTConfig"}),
        ("config.json", {"AutoModel": "other/source--modeling.MERTModel"}),
        ("preprocessor_config.json", {"AutoFeatureExtractor": "other/source--processing.MERTProcessor"}),
        ("preprocessor_config.json", {"AutoTokenizer": ["other/source--tokenization.Tokenizer", None]}),
    ],
)
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_custom_code_cannot_resolve_outside_the_hashed_snapshot(
    monkeypatch, tmp_path, filename, auto_map, trust_remote_code,
) -> None:  # noqa: ANN001
    root = _snapshot(tmp_path)
    (root / filename).write_text(json.dumps({"auto_map": auto_map}))
    _install_fake_backend(monkeypatch, load_error=AssertionError("dynamic loader must not run"))
    with pytest.raises(BackendLoadError, match="cross-repository custom code"):
        audio_music_mert(_audio(), model=root, trust_remote_code=trust_remote_code)


def test_processor_custom_code_also_requires_explicit_opt_in(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    root = _snapshot(tmp_path)
    (root / "preprocessor_config.json").write_text(
        json.dumps({"auto_map": {"AutoFeatureExtractor": "processing.MERTProcessor"}})
    )
    with pytest.raises(BackendLoadError, match="trust_remote_code=True"):
        audio_music_mert(_audio(), model=root)
    _install_fake_backend(monkeypatch, expect_trust=True)
    assert audio_music_mert(_audio(), model=root, trust_remote_code=True).coords["layer"] == [0, 1, 2]


def test_versioned_configuration_cannot_bypass_custom_code_preflight(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    root = _snapshot(tmp_path)
    (root / "config.json").write_text(json.dumps({"configuration_files": ["config.4.0.0.json"]}))
    (root / "config.4.0.0.json").write_text(json.dumps({
        "model_type": "mert_model", "auto_map": {"AutoConfig": "other/source--configuration.MERTConfig"},
    }))
    _install_fake_backend(monkeypatch, load_error=AssertionError("dynamic loader must not run"))
    with pytest.raises(BackendLoadError, match="unsupported versioned configuration_files"):
        audio_music_mert(_audio(), model=root, trust_remote_code=True)


@pytest.mark.parametrize("filename,contents,message", [
    (
        "preprocessor_config.json",
        {"feature_extractor": {"auto_map": {"AutoFeatureExtractor": "other/source--processing.MERTProcessor"}}},
        "unsupported nested feature_extractor",
    ),
    (
        "processor_config.json",
        {"feature_extractor": {"auto_map": {"AutoFeatureExtractor": "other/source--processing.MERTProcessor"}}},
        "unsupported processor_config.json",
    ),
    ("adapter_config.json", {"base_model_name_or_path": "other/base-model"}, "unsupported adapter_config.json"),
])
def test_unsupported_loader_indirection_is_refused_before_backend_loading(
    monkeypatch, tmp_path, filename, contents, message,
) -> None:  # noqa: ANN001
    root = _snapshot(tmp_path)
    (root / filename).write_text(json.dumps(contents))
    _install_fake_backend(monkeypatch, load_error=AssertionError("redirecting loader must not run"))
    with pytest.raises(BackendLoadError, match=message):
        audio_music_mert(_audio(), model=root, trust_remote_code=True)


@pytest.mark.parametrize("kind", ["empty", "absolute", "relative", "parent_relative", "path_object"])
def test_unavailable_local_snapshot_is_a_load_error_without_hub_resolution(
    monkeypatch, tmp_path, kind,
) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **_kwargs: pytest.fail("local paths must not reach Hub resolution")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    root = tmp_path / "absent-model"
    models = {
        "empty": root, "absolute": str(root), "relative": "./absent-model",
        "parent_relative": "../absent-parent/model", "path_object": Path("absent-model"),
    }
    if kind == "empty":
        root.mkdir()
    with pytest.raises(BackendLoadError) as error:
        audio_music_mert(_audio(), model=models[kind], revision="0" * 40)
    assert error.value.phase == "load"
    assert isinstance(error.value.__cause__, ValueError if kind == "empty" else FileNotFoundError)


def test_unreadable_local_asset_keeps_load_phase_and_cause(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    root = _snapshot(tmp_path)
    original_open = Path.open
    denied = PermissionError("synthetic unreadable model asset")

    def open_asset(path, *args, **kwargs):
        if path == root / "config.json":
            raise denied
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_asset)
    with pytest.raises(BackendLoadError) as error:
        audio_music_mert(_audio(), model=root)
    assert error.value.__cause__ is denied


def test_rate_mismatch_is_an_inference_error(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, config_overrides={"sample_rate": 16})
    with pytest.raises(BackendInferenceError, match="expects 16 Hz"):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))


def test_processor_and_config_rate_disagreement_is_a_load_error(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, processor_rate=16)
    with pytest.raises(BackendLoadError, match="disagrees"):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))


def test_adapter_and_padded_geometries_are_refused(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, config_overrides={"add_adapter": True})
    with pytest.raises(BackendLoadError, match="adapter-equipped"):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))

    _install_fake_backend(monkeypatch, config_overrides={"conv_padding": [1, 0]})
    with pytest.raises(BackendLoadError, match="padded convolution"):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"feature_extractor_cqt": True}, "CQT"),
        ({"conv_dilation": [1, 2]}, "non-unit convolution dilation"),
    ],
)
def test_unsupported_cqt_and_dilation_are_load_errors(monkeypatch, tmp_path, overrides, message) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, config_overrides=overrides)
    with pytest.raises(BackendLoadError, match=message):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))


def test_malformed_geometry_and_non_mert_model_are_load_errors(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, config_overrides={"conv_kernel": [0, 2]})
    with pytest.raises(BackendLoadError, match="geometry is invalid"):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))

    _install_fake_backend(monkeypatch, config_overrides={"model_type": "wav2vec2"})
    with pytest.raises(BackendLoadError, match="not a MERT"):
        audio_music_mert(_audio(), model=_snapshot(tmp_path))


@pytest.mark.parametrize("error", [OSError("missing local artifact"), ValueError("invalid checkpoint config")])
def test_loader_failures_keep_their_phase(monkeypatch, tmp_path, error) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, load_error=error)
    with pytest.raises(BackendLoadError) as load_error:
        audio_music_mert(_audio(), model=_snapshot(tmp_path))
    assert load_error.value.__cause__ is error


def test_model_eval_value_error_is_a_load_failure(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch, fail_eval=True)
    with pytest.raises(BackendLoadError) as load_error:
        audio_music_mert(_audio(), model=_snapshot(tmp_path))
    assert isinstance(load_error.value.__cause__, ValueError)

    _install_fake_backend(monkeypatch, fail_inference=True)
    with pytest.raises(BackendInferenceError) as inference_error:
        audio_music_mert(_audio(), model=_snapshot(tmp_path))
    assert isinstance(inference_error.value.__cause__, RuntimeError)


def test_remote_offline_cache_miss_is_a_load_error(monkeypatch) -> None:  # noqa: ANN001
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **_kwargs: (_ for _ in ()).throw(OSError("offline cache miss"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(BackendLoadError, match="locally cached"):
        audio_music_mert(_audio(), model="m-a-p/MERT-v1-95M", revision="0" * 40)


def test_changed_local_asset_changes_model_identity(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch)
    root = _snapshot(tmp_path)
    first = audio_music_mert(_audio(), model=root, layers=[0])
    (root / "extra.py").write_text("# changed reviewed local source\n")
    second = audio_music_mert(_audio(), model=root, layers=[0])
    assert first.metadata["model_revision"] != second.metadata["model_revision"]


def test_repeat_evaluation_is_deterministic(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    _install_fake_backend(monkeypatch)
    root = _snapshot(tmp_path)
    first = audio_music_mert(_audio(), model=root, layers=[2, 0])
    second = audio_music_mert(_audio(), model=root, layers=[2, 0])
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.times_s, second.times_s)
    assert first.metadata == second.metadata
