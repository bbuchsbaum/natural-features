"""Speech SSL representation extractors."""

from __future__ import annotations

from typing import Any

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import mel
from natural_features.features.common import extractor_metadata
from natural_features.util.hashing import stable_hash


def _fallback_ssl(
    stimulus: AudioStimulus,
    *,
    layers: list[int],
    stride_s: float,
    execution_mode: str,
    fallback_reason: str,
    dim: int = 64,
) -> FeatureSeries:
    base = mel(stimulus, hop_s=stride_s, win_s=max(0.025, stride_s * 2), n_mels=min(64, dim))
    x = base.values.astype(np.float32)
    if x.shape[1] != dim:
        if x.shape[1] > dim:
            x = x[:, :dim]
        else:
            pad = np.zeros((x.shape[0], dim - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
    out = np.zeros((x.shape[0], len(layers), dim), dtype=np.float32)
    for i, layer in enumerate(layers):
        seed = int(stable_hash({"layer": int(layer), "dim": dim}, length=8), 16) % (2**32)
        rng = np.random.default_rng(seed)
        proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, dim)).astype(np.float32)
        out[:, i, :] = np.tanh(x @ proj)
    md = add_execution_provenance(
        extractor_metadata(
            "speech.ssl.wavlm",
            params={"layers": layers, "stride_s": stride_s},
            extra={"backend": "fallback"},
        ),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=fallback_reason,
    )
    return FeatureSeries(
        values=out,
        times_s=base.times_s,
        dims=("time", "layer", "unit"),
        coords={"layer": layers, "unit": [f"u{i}" for i in range(dim)]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=stride_s, sampling_rate_hz=1.0 / stride_s),
    )


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
    mode, strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    layers = layers or [2, 6, 12]
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel  # type: ignore
    except Exception:
        if strict_dependency:
            raise RuntimeError("transformers+torch are required for strict speech SSL extraction.")
        return _fallback_ssl(
            stimulus,
            layers=layers,
            stride_s=stride_s,
            execution_mode=mode,
            fallback_reason="transformers/torch unavailable",
        )

    # Avoid forced downloads in normal paths; fallback if local model is unavailable.
    try:
        fe = AutoFeatureExtractor.from_pretrained(model, local_files_only=True)
        net = AutoModel.from_pretrained(model, local_files_only=True)
    except Exception:
        if strict_dependency:
            raise RuntimeError(f"Model '{model}' unavailable locally for strict mode.")
        return _fallback_ssl(
            stimulus,
            layers=layers,
            stride_s=stride_s,
            execution_mode=mode,
            fallback_reason="local model unavailable",
        )

    wav = stimulus.samples.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    inputs = fe(
        wav,
        sampling_rate=stimulus.sr_hz,
        return_tensors="pt",
    )
    try:
        with torch.no_grad():
            out = net(**inputs, output_hidden_states=True)
    except Exception as exc:
        if strict_dependency:
            raise RuntimeError("Speech SSL inference failed in strict mode.") from exc
        return _fallback_ssl(
            stimulus,
            layers=layers,
            stride_s=stride_s,
            execution_mode=mode,
            fallback_reason=f"model inference failed: {type(exc).__name__}",
        )
    hstates = out.hidden_states
    selected = []
    for l in layers:
        l_idx = max(0, min(int(l), len(hstates) - 1))
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
