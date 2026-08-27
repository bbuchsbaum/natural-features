"""Model-specific, clip-level audio neural embedding extractors."""

from __future__ import annotations

from typing import Any

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import SupportSpec, TimebaseSpec
from natural_features.features.common import extractor_metadata


def _mono_waveform(stimulus: AudioStimulus) -> np.ndarray:
    waveform = stimulus.samples.astype(np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _single_clip_embedding(value: Any, *, backend: str, dim: int | None) -> np.ndarray:
    values = _numpy(value)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] == 0:
        raise BackendInferenceError(
            backend,
            f"expected one clip embedding with shape (1, feature), received {values.shape}",
        )
    if not np.all(np.isfinite(values)):
        raise BackendInferenceError(backend, "model returned non-finite embedding values")
    if dim is not None:
        expected = int(dim)
        if expected <= 0:
            raise ValueError("dim must be > 0")
        if values.shape[1] != expected:
            raise ValueError(
                f"Requested dim={expected}, but the model returned its native "
                f"dimension {values.shape[1]}; dimensionality reduction must be "
                "an explicit downstream transform"
            )
    return values.astype(np.float32, copy=False)


def _clip_result(
    stimulus: AudioStimulus,
    *,
    values: np.ndarray,
    extractor_name: str,
    params: dict[str, object],
    backend: str,
    representation: str,
    execution_mode: str,
) -> FeatureSeries:
    onset = float(stimulus.start_offset_s)
    offset = onset + (stimulus.samples.shape[0] / float(stimulus.sr_hz))
    metadata = add_execution_provenance(
        extractor_metadata(
            extractor_name,
            params=params,
            model_revision=str(params["model"]),
            extra={
                "backend": backend,
                "representation": representation,
                "temporal_scope": "clip",
            },
        ),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values,
        times_s=np.asarray([onset], dtype=np.float64),
        dims=("time", "feature"),
        coords={"feature": [f"dim_{i}" for i in range(values.shape[1])]},
        metadata=metadata,
        timebase=TimebaseSpec(
            kind="audio_summary",
            reference=stimulus.clock,
            alignment="onset",
            support=SupportSpec(kind="interval", anchor="onset"),
        ),
        time_bounds_s=np.asarray([[onset, offset]], dtype=np.float64),
        temporal_context=stimulus.temporal_context,
    )


def audio_clap_embeddings(
    stimulus: AudioStimulus,
    *,
    model: str = "laion/clap-htsat-unfused",
    dim: int | None = None,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return the CLAP model's native whole-clip audio projection."""

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    backend = "CLAP"
    params: dict[str, object] = {
        "model": model,
        "dim": dim,
        "local_files_only": local_files_only,
    }
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor, ClapModel  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            backend,
            "transformers and torch with CLAP support are required",
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
        net = ClapModel.from_pretrained(model, local_files_only=local_files_only)
    except Exception as exc:
        raise BackendLoadError(backend, f"model '{model}' is unavailable") from exc

    try:
        inputs = processor(
            audios=_mono_waveform(stimulus),
            sampling_rate=stimulus.sr_hz,
            return_tensors="pt",
        )
        with torch.no_grad():
            embedding = net.get_audio_features(**inputs)
    except Exception as exc:
        raise BackendInferenceError(backend, "audio projection failed") from exc
    values = _single_clip_embedding(embedding, backend=backend, dim=dim)
    return _clip_result(
        stimulus,
        values=values,
        extractor_name="audio.clap",
        params=params,
        backend="transformers_clap",
        representation="audio_projection",
        execution_mode=mode,
    )


def audio_ast_embeddings(
    stimulus: AudioStimulus,
    *,
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    dim: int | None = None,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return the AST model's native whole-clip pooled representation."""

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    backend = "AST"
    params: dict[str, object] = {
        "model": model,
        "dim": dim,
        "local_files_only": local_files_only,
    }
    try:
        import torch  # type: ignore
        from transformers import ASTModel, AutoFeatureExtractor  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            backend,
            "transformers and torch with AST support are required",
        ) from exc

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
        net = ASTModel.from_pretrained(model, local_files_only=local_files_only)
    except Exception as exc:
        raise BackendLoadError(backend, f"model '{model}' is unavailable") from exc

    try:
        inputs = feature_extractor(
            _mono_waveform(stimulus),
            sampling_rate=stimulus.sr_hz,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = net(**inputs)
    except Exception as exc:
        raise BackendInferenceError(backend, "pooled audio representation failed") from exc
    pooled = getattr(model_output, "pooler_output", None)
    if pooled is None:
        raise BackendInferenceError(backend, "ASTModel did not return pooler_output")
    values = _single_clip_embedding(pooled, backend=backend, dim=dim)
    return _clip_result(
        stimulus,
        values=values,
        extractor_name="audio.ast",
        params=params,
        backend="transformers_ast",
        representation="pooler_output",
        execution_mode=mode,
    )
