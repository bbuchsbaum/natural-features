"""Whole-clip speech-emotion classification."""

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


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("emotion logits must be a non-empty 1-D vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("emotion logits must be finite")
    centered = values - np.max(values)
    exp = np.exp(centered).astype(np.float32)
    return exp / exp.sum(dtype=np.float32)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _transformers_emotion(
    stimulus: AudioStimulus,
    *,
    model: str,
    local_files_only: bool,
    execution_mode: str,
    params: dict[str, object],
) -> FeatureSeries:
    backend = "speech emotion"
    try:
        import torch  # type: ignore
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            backend,
            "transformers and torch with audio-classification support are required",
        ) from exc

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
        net = AutoModelForAudioClassification.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise BackendLoadError(backend, f"model '{model}' is unavailable") from exc

    waveform = stimulus.samples.astype(np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    try:
        inputs = feature_extractor(
            waveform,
            sampling_rate=stimulus.sr_hz,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = net(**inputs)
    except Exception as exc:
        raise BackendInferenceError(backend, "whole-clip classification failed") from exc

    logits = getattr(model_output, "logits", None)
    if logits is None:
        raise BackendInferenceError(backend, "model did not return logits")
    logits_array = _numpy(logits)
    if logits_array.ndim != 2 or logits_array.shape[0] != 1:
        raise BackendInferenceError(
            backend,
            f"expected one clip logits row, received {logits_array.shape}",
        )
    probabilities = _softmax(logits_array[0])
    id2label = getattr(getattr(net, "config", None), "id2label", None) or {}
    labels = [
        str(id2label.get(index, id2label.get(str(index), f"emotion_{index}")))
        for index in range(probabilities.shape[0])
    ]
    onset = float(stimulus.start_offset_s)
    offset = onset + (stimulus.samples.shape[0] / float(stimulus.sr_hz))
    metadata = add_execution_provenance(
        extractor_metadata(
            "speech.emotion",
            params=params,
            model_revision=model,
            extra={
                "backend": "transformers_audio_classification",
                "labels": labels,
                "temporal_scope": "clip",
                "quantity": "class_probability",
            },
        ),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=probabilities.reshape(1, -1).astype(np.float32),
        times_s=np.asarray([onset], dtype=np.float64),
        dims=("time", "feature"),
        coords={"feature": labels},
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


def speech_emotion(
    stimulus: AudioStimulus,
    *,
    model: str = "superb/wav2vec2-base-superb-er",
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return one class-probability row covering the complete audio clip.

    This extractor is intentionally clip-level. It does not imply framewise
    estimates or manufacture a sampling rate from a requested hop size.
    """

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    params: dict[str, object] = {
        "model": model,
        "local_files_only": local_files_only,
    }
    return _transformers_emotion(
        stimulus,
        model=model,
        local_files_only=local_files_only,
        execution_mode=mode,
        params=params,
    )
