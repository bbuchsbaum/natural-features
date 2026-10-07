"""Strict, native-temporal MERT hidden-state extraction.

This module deliberately has no module-level optional imports.  MERT assets are
resolved to an immutable local snapshot before Transformers sees them, so a
cache pathname alone is never used as a model identity.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio._music_contract import (
    asset_digest,
    computation_context,
    mono_audio,
    music_series,
)

_BACKEND = "MERT"
_DEFAULT_MODEL = "m-a-p/MERT-v1-95M"
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must contain positive integers")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return value


def _conv_geometry(config: Any) -> tuple[int, int]:
    """Return valid-convolution ``(receptive_field, stride)`` for MERT."""
    if getattr(config, "model_type", None) != "mert_model":
        raise BackendLoadError(_BACKEND, "checkpoint config is not a MERT model")
    kernels = getattr(config, "conv_kernel", None)
    strides = getattr(config, "conv_stride", None)
    if not isinstance(kernels, (list, tuple)) or not isinstance(strides, (list, tuple)):
        raise BackendLoadError(_BACKEND, "checkpoint config lacks conv_kernel/conv_stride geometry")
    if not kernels or len(kernels) != len(strides):
        raise BackendLoadError(_BACKEND, "checkpoint convolution geometry is empty or inconsistent")
    dilation = getattr(config, "conv_dilation", None)
    if dilation is not None:
        if not isinstance(dilation, (list, tuple)) or len(dilation) != len(kernels) or any(
            isinstance(x, bool) or not isinstance(x, (int, np.integer)) or int(x) != 1
            for x in dilation
        ):
            raise BackendLoadError(_BACKEND, "non-unit convolution dilation is unsupported")
    padding = getattr(config, "conv_padding", None)
    if padding is not None:
        if not isinstance(padding, (list, tuple)) or any(
            isinstance(x, bool) or not isinstance(x, (int, np.integer)) or int(x) != 0
            for x in padding
        ):
            raise BackendLoadError(_BACKEND, "padded convolution geometry is unsupported")
    if bool(getattr(config, "feature_extractor_cqt", False)):
        raise BackendLoadError(_BACKEND, "CQT feature extractors are unsupported")
    if bool(getattr(config, "add_adapter", False)):
        raise BackendLoadError(_BACKEND, "adapter-equipped MERT checkpoints are unsupported")

    receptive_field, total_stride = 1, 1
    try:
        for kernel, stride in zip(kernels, strides, strict=True):
            k = _integer(kernel, name="conv_kernel")
            s = _integer(stride, name="conv_stride")
            receptive_field += (k - 1) * total_stride
            total_stride *= s
    except ValueError as exc:
        raise BackendLoadError(_BACKEND, "checkpoint convolution geometry is invalid") from exc
    return receptive_field, total_stride


def _valid_output_length(n_samples: int, receptive_field: int, stride: int) -> int:
    if n_samples < receptive_field:
        raise ValueError(
            f"audio has {n_samples} samples, but this MERT convolution requires at least "
            f"{receptive_field} samples"
        )
    return 1 + (n_samples - receptive_field) // stride


def _sample_rate(value: Any, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise BackendLoadError(_BACKEND, f"{source} does not declare a valid sampling rate")
    value = float(value)
    if not np.isfinite(value) or value <= 0 or int(value) != value:
        raise BackendLoadError(_BACKEND, f"{source} does not declare a valid sampling rate")
    return int(value)


def _config_count(value: Any, *, source: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise BackendLoadError(_BACKEND, f"{source} must be an integer")
    count = int(value)
    if count < 0 or (count == 0 and not allow_zero):
        raise BackendLoadError(_BACKEND, f"{source} must be positive")
    return count


def _processor_rate(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", processor)
    return _sample_rate(getattr(feature_extractor, "sampling_rate", None), source="processor")


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    raise BackendInferenceError(_BACKEND, "processor did not return a mapping of model inputs")


def _snapshot(model: str | Path, revision: str | None, *, local_files_only: bool) -> tuple[Path, str]:
    try:
        path = Path(model).expanduser()
        if path.exists():
            if not path.is_dir():
                raise NotADirectoryError(f"local model asset '{path}' must be a directory")
            return path.resolve(), asset_digest(path)
        if isinstance(model, Path) or path.is_absolute() or str(model).startswith(("./", "../", "~")):
            raise FileNotFoundError(f"local model snapshot '{path}' does not exist")
    except (OSError, ValueError, RuntimeError) as exc:
        raise BackendLoadError(_BACKEND, f"local model snapshot '{model}' could not be loaded") from exc
    if revision is None or not _PINNED_REVISION.fullmatch(revision):
        raise ValueError("remote MERT model IDs require an explicit 40-hex pinned revision")
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except (ImportError, OSError) as exc:
        raise BackendDependencyError(
            _BACKEND, "huggingface_hub is required to resolve a pinned local MERT snapshot"
        ) from exc
    try:
        root = Path(
            snapshot_download(
                repo_id=str(model),
                revision=revision,
                local_files_only=local_files_only,
                allow_patterns=[
                    "config.json", "preprocessor_config.json", "*.py",
                    "pytorch_model.bin", "model.safetensors",
                ],
            )
        )
        digest = asset_digest(root)
    except Exception as exc:
        scope = "locally cached" if local_files_only else "available"
        raise BackendLoadError(
            _BACKEND, f"pinned model '{model}' at revision '{revision}' is not {scope}"
        ) from exc
    return root, f"{revision}+{digest}"


def _custom_code_requested(root: Path) -> bool:
    """Admit flat MERT configurations whose code maps stay in the hashed snapshot."""
    for filename in ("processor_config.json", "adapter_config.json"):
        if (root / filename).exists():
            raise BackendLoadError(
                _BACKEND,
                f"snapshot contains unsupported {filename} indirection; use a standalone MERT "
                "checkpoint with flat config.json and preprocessor_config.json files",
            )
    requested = False
    for filename in ("config.json", "preprocessor_config.json"):
        config_path = root / filename
        if filename != "config.json" and not config_path.exists():
            continue
        try:
            with config_path.open(encoding="utf-8") as stream:
                config = json.load(stream)
        except Exception as exc:
            raise BackendLoadError(_BACKEND, f"snapshot '{root}' has no readable {filename}") from exc
        if not isinstance(config, dict):
            raise BackendLoadError(_BACKEND, f"{filename} must contain a configuration object")
        if "feature_extractor" in config:
            raise BackendLoadError(
                _BACKEND, f"{filename} contains an unsupported nested feature_extractor configuration",
            )
        if "configuration_files" in config:
            raise BackendLoadError(
                _BACKEND,
                f"{filename} declares unsupported versioned configuration_files; use a single "
                "reviewed configuration inside the content-identified snapshot",
            )
        auto_map = config.get("auto_map") or {}
        if not isinstance(auto_map, dict):
            raise BackendLoadError(_BACKEND, f"{filename} auto_map must be an object")
        for references in auto_map.values():
            references = references if isinstance(references, (list, tuple)) else [references]
            for reference in references:
                if reference is None:
                    continue
                # Transformers treats repo--module.Class as a separate code repository,
                # whose cached revision is not covered by this snapshot's identity.
                if isinstance(reference, str) and "--" in reference:
                    raise BackendLoadError(
                        _BACKEND,
                        f"{filename} declares cross-repository custom code; source must reside "
                        "inside the content-identified snapshot",
                    )
                if not isinstance(reference, str) or not re.fullmatch(
                    r"[A-Za-z_]\w*\.[A-Za-z_]\w*", reference, flags=re.ASCII
                ):
                    raise BackendLoadError(_BACKEND, f"{filename} has an invalid local auto_map reference")
        requested = requested or bool(auto_map)
    return requested


def _selected_layers(layers: Sequence[int] | None, n_hidden: int) -> list[int]:
    chosen = list(range(n_hidden)) if layers is None else list(layers)
    if not chosen:
        raise ValueError("layers must contain at least one integer layer index")
    if any(isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) for layer in chosen):
        raise ValueError("layers must contain integer layer indices")
    selected = [int(layer) for layer in chosen]
    if len(set(selected)) != len(selected):
        raise ValueError("layers must not contain duplicates")
    if min(selected) < 0 or max(selected) >= n_hidden:
        raise ValueError(f"requested MERT layers must be within [0, {n_hidden - 1}]")
    return selected


def audio_music_mert(
    stimulus: AudioStimulus,
    *,
    model: str | Path = _DEFAULT_MODEL,
    revision: str | None = None,
    layers: Sequence[int] | None = None,
    channel_policy: str = "mean",
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    device: str = "cpu",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return native MERT states on their valid convolution sample cells.

    A remote ``model`` always needs a 40-character commit ``revision``.  Local
    directories are identified by their content digest.  Processing runs once on
    the complete supplied waveform; MERT's bidirectional transformer context is
    therefore recorded for every layer, including layer zero.
    """
    mode, _ = resolve_execution_mode(
        execution_mode=execution_mode, strict_dependency=strict_dependency
    )
    if not isinstance(local_files_only, bool) or not isinstance(trust_remote_code, bool):
        raise ValueError("local_files_only and trust_remote_code must be booleans")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    waveform = mono_audio(stimulus, channel_policy=channel_policy)
    root, model_revision = _snapshot(model, revision, local_files_only=local_files_only)
    if _custom_code_requested(root) and not trust_remote_code:
        raise BackendLoadError(
            _BACKEND,
            "snapshot declares custom Transformers code; set trust_remote_code=True only for a "
            "reviewed, source-pinned snapshot",
        )
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
        from transformers import AutoFeatureExtractor, AutoModel  # type: ignore
    except (ImportError, OSError) as exc:
        raise BackendDependencyError(_BACKEND, "torch and transformers with MERT support are required") from exc
    try:
        processor = AutoFeatureExtractor.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=trust_remote_code
        )
        net = AutoModel.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        raise BackendLoadError(_BACKEND, f"pinned snapshot '{root}' could not be loaded") from exc
    try:
        net = net.to(device) if hasattr(net, "to") else net
        net.eval()
    except Exception as exc:
        raise BackendLoadError(_BACKEND, f"pinned snapshot '{root}' could not enter evaluation mode") from exc
    try:
        processor_sr = _processor_rate(processor)
        config_sr = _sample_rate(getattr(net.config, "sample_rate", None), source="model config")
        if processor_sr != config_sr:
            raise BackendLoadError(
                _BACKEND, f"processor rate {processor_sr} Hz disagrees with config rate {config_sr} Hz"
            )
        receptive_field, stride = _conv_geometry(net.config)
        expected_layers = _config_count(
            getattr(net.config, "num_hidden_layers", None),
            source="model config num_hidden_layers",
            allow_zero=True,
        ) + 1
        expected_width = _config_count(
            getattr(net.config, "hidden_size", None), source="model config hidden_size"
        )
    except BackendLoadError:
        raise
    except Exception as exc:
        raise BackendLoadError(_BACKEND, f"pinned snapshot '{root}' has an invalid configuration") from exc
    if int(stimulus.sr_hz) != config_sr:
        raise BackendInferenceError(
            _BACKEND,
            f"model expects {config_sr} Hz audio but stimulus is {stimulus.sr_hz} Hz; resample explicitly first",
        )
    expected_time = _valid_output_length(len(waveform), receptive_field, stride)
    selected = _selected_layers(layers, expected_layers)

    try:
        inputs = processor(waveform, sampling_rate=config_sr, return_tensors="pt")
        raw_input = inputs.get("input_values") if isinstance(inputs, dict) else getattr(inputs, "input_values", None)
        if raw_input is None:
            raise BackendInferenceError(_BACKEND, "processor did not return input_values")
        input_values = _tensor_numpy(raw_input)
        if input_values.ndim != 2 or input_values.shape != (1, len(waveform)):
            raise BackendInferenceError(
                _BACKEND,
                "processor changed waveform length or batch shape; implicit padding, truncation, and resampling are unsupported",
            )
        if not np.all(np.isfinite(input_values)):
            raise BackendInferenceError(_BACKEND, "processor returned non-finite input_values")
        inputs = _move_inputs(inputs, device)
        inference_mode = getattr(torch, "inference_mode", None)
        if inference_mode is None:
            raise BackendDependencyError(_BACKEND, "torch.inference_mode is required for strict MERT evaluation")
        with inference_mode():
            output = net(**inputs, output_hidden_states=True)
    except (BackendDependencyError, BackendInferenceError):
        raise
    except Exception as exc:
        raise BackendInferenceError(_BACKEND, "processor or MERT evaluation failed") from exc

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise BackendInferenceError(_BACKEND, "MERT did not return hidden_states")
    if len(hidden_states) != expected_layers:
        raise BackendInferenceError(
            _BACKEND,
            f"MERT returned {len(hidden_states)} hidden states; config requires {expected_layers}",
        )
    arrays: list[np.ndarray] = []
    try:
        for layer in selected:
            values = _tensor_numpy(hidden_states[layer])
            if (
                values.ndim != 3
                or values.shape[0] != 1
                or values.shape[1] != expected_time
                or values.shape[2] != expected_width
            ):
                raise BackendInferenceError(
                    _BACKEND,
                    f"layer {layer} has shape {values.shape}; expected "
                    f"(1, {expected_time}, {expected_width})",
                )
            if not np.all(np.isfinite(values)):
                raise BackendInferenceError(_BACKEND, f"layer {layer} contains non-finite values")
            arrays.append(values[0].astype(np.float32, copy=False))
    except BackendInferenceError:
        raise
    except Exception as exc:
        raise BackendInferenceError(_BACKEND, "MERT hidden-state conversion failed") from exc
    values = np.stack(arrays, axis=1)
    starts = np.arange(expected_time, dtype=np.float64) * stride / float(config_sr)
    bounds = np.column_stack([starts, starts + receptive_field / float(config_sr)])
    bounds += float(stimulus.start_offset_s)
    times = float(stimulus.start_offset_s) + (
        np.arange(expected_time, dtype=np.float64) * stride + (receptive_field - 1) / 2.0
    ) / float(config_sr)
    clip_bounds = np.asarray(
        [[float(stimulus.start_offset_s), float(stimulus.start_offset_s) + len(waveform) / config_sr]],
        dtype=np.float64,
    )
    context = computation_context(
        stimulus,
        policy="bidirectional",
        bounds_s=clip_bounds,
        preprocessing={
            "channel_policy": channel_policy,
            "processor_sampling_rate_hz": config_sr,
            "processor_normalization": bool(getattr(processor, "do_normalize", False)),
            "whole_input": True,
        },
    )
    model_revision = (
        f"{model_revision};torch={getattr(torch, '__version__', 'unknown')};"
        f"transformers={getattr(transformers, '__version__', 'unknown')}"
    )
    result = music_series(
        stimulus,
        values=values,
        times_s=times,
        bounds_s=bounds,
        extractor="audio.music.mert",
        params={
            "model": str(model), "revision": revision, "layers": selected,
            "channel_policy": channel_policy, "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code, "device": device,
        },
        context=context,
        hop_s=stride / float(config_sr),
        dims=("time", "layer", "unit"),
        coords={"layer": selected, "unit": [f"u{i}" for i in range(values.shape[2])]},
        model_revision=model_revision,
        extra={
            "backend": "transformers_mert", "receptive_field_samples": receptive_field,
            "stride_samples": stride, "device": device,
        },
    )
    return replace(
        result,
        metadata=add_execution_provenance(
            result.metadata, execution_mode=mode, fallback_used=False, backend="transformers_mert"
        ),
    )


__all__ = ["audio_music_mert"]
