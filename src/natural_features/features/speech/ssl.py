"""Speech SSL representation extractors."""

from __future__ import annotations

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.common import extractor_metadata


def wavlm_hidden_states(
    stimulus: AudioStimulus,
    *,
    model: str = "microsoft/wavlm-base-plus",
    layers: list[int] | None = None,
    stride_s: float = 0.02,
    pooling: str = "none",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    layers = layers or [2, 6, 12]
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            "WavLM",
            "transformers and torch are required",
        ) from exc

    try:
        fe = AutoFeatureExtractor.from_pretrained(model, local_files_only=True)
        net = AutoModel.from_pretrained(model, local_files_only=True)
    except Exception as exc:
        raise BackendLoadError("WavLM", f"model '{model}' is unavailable locally") from exc

    wav = stimulus.samples.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    try:
        inputs = fe(
            wav,
            sampling_rate=stimulus.sr_hz,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = net(**inputs, output_hidden_states=True)
    except Exception as exc:
        raise BackendInferenceError("WavLM", "feature extraction or evaluation failed") from exc
    hstates = out.hidden_states
    selected = []
    for layer in layers:
        l_idx = int(layer)
        if l_idx < 0 or l_idx >= len(hstates):
            raise ValueError(
                f"Requested WavLM layer {l_idx} is outside "
                f"[0, {len(hstates) - 1}]"
            )
        arr = hstates[l_idx][0].detach().cpu().numpy().astype(np.float32)
        selected.append(arr)
    min_t = min(x.shape[0] for x in selected)
    selected = [x[:min_t] for x in selected]
    stack = np.stack(selected, axis=1)  # T x L x D
    if pooling == "mean":
        stack = stack.mean(axis=2, keepdims=True)
    elif pooling == "max":
        stack = stack.max(axis=2, keepdims=True)
    elif pooling != "none":
        raise ValueError(f"Unsupported pooling: {pooling}")
    times = times_from_hop(min_t, stride_s, start_offset_s=stimulus.start_offset_s)
    md = add_execution_provenance(
        extractor_metadata(
            "speech.ssl.wavlm",
            params={"model": model, "layers": layers, "stride_s": stride_s, "pooling": pooling},
            extra={"backend": "transformers_local"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=stack,
        times_s=times,
        dims=("time", "layer", "unit"),
        coords={"layer": layers, "unit": [f"u{i}" for i in range(stack.shape[2])]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=stride_s, sampling_rate_hz=1.0 / stride_s),
    )


def hubert_hidden_states(
    stimulus: AudioStimulus,
    *,
    model: str = "facebook/hubert-base-ls960",
    layers: list[int] | None = None,
    stride_s: float = 0.02,
    pooling: str = "none",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return HuBERT-style hidden states with the same contract as WavLM."""

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    layers = layers or [2, 6, 12]
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            "HuBERT",
            "transformers and torch are required",
        ) from exc
    try:
        fe = AutoFeatureExtractor.from_pretrained(model, local_files_only=True)
        net = AutoModel.from_pretrained(model, local_files_only=True)
    except Exception as exc:
        raise BackendLoadError("HuBERT", f"model '{model}' is unavailable locally") from exc
    wav = stimulus.samples.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    try:
        inputs = fe(wav, sampling_rate=stimulus.sr_hz, return_tensors="pt")
        with torch.no_grad():
            out = net(**inputs, output_hidden_states=True)
    except Exception as exc:
        raise BackendInferenceError("HuBERT", "feature extraction or evaluation failed") from exc
    hstates = out.hidden_states
    selected = []
    for layer in layers:
        l_idx = int(layer)
        if l_idx < 0 or l_idx >= len(hstates):
            raise ValueError(
                f"Requested HuBERT layer {l_idx} is outside "
                f"[0, {len(hstates) - 1}]"
            )
        selected.append(hstates[l_idx][0].detach().cpu().numpy().astype(np.float32))
    min_t = min(x.shape[0] for x in selected)
    stack = np.stack([x[:min_t] for x in selected], axis=1)
    if pooling == "mean":
        stack = stack.mean(axis=2, keepdims=True)
    elif pooling == "max":
        stack = stack.max(axis=2, keepdims=True)
    elif pooling != "none":
        raise ValueError(f"Unsupported pooling: {pooling}")
    times = times_from_hop(min_t, stride_s, start_offset_s=stimulus.start_offset_s)
    md = add_execution_provenance(
        extractor_metadata(
            "speech.hubert",
            params={"model": model, "layers": layers, "stride_s": stride_s, "pooling": pooling},
            extra={"backend": "transformers_local"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=stack,
        times_s=times,
        dims=("time", "layer", "unit"),
        coords={"layer": layers, "unit": [f"u{i}" for i in range(stack.shape[2])]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=stride_s, sampling_rate_hz=1.0 / stride_s),
    )
