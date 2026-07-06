"""Audio neural embedding extractors with deterministic fallbacks."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio.lowlevel import mel
from natural_features.features.common import extractor_metadata
from natural_features.util.hashing import stable_hash


def _fit_dim(values: np.ndarray, dim: int) -> np.ndarray:
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be > 0")
    if values.shape[1] == dim:
        return values.astype(np.float32)
    if values.shape[1] > dim:
        return values[:, :dim].astype(np.float32)
    pad = np.zeros((values.shape[0], dim - values.shape[1]), dtype=np.float32)
    return np.concatenate([values.astype(np.float32), pad], axis=1)


def _fallback_audio_embedding(
    stimulus: AudioStimulus,
    *,
    extractor_name: str,
    stride_s: float,
    dim: int,
    execution_mode: str,
    reason: str,
    params: dict[str, object],
) -> FeatureSeries:
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be > 0")
    base = mel(stimulus, hop_s=stride_s, win_s=max(0.05, stride_s * 2.0), n_mels=min(64, dim), log=True)
    desc = base.values.astype(np.float32)
    seed = int(stable_hash({"extractor": extractor_name, "dim": int(dim)}, length=8), 16) % (2**32)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(desc.shape[1]), size=(desc.shape[1], int(dim))).astype(np.float32)
    values = np.tanh(desc @ proj).astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params=params, extra={"backend": "fallback_projection"}),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason=reason,
    )
    return FeatureSeries(
        values=values,
        times_s=base.times_s,
        dims=("time", "feature"),
        coords={"feature": [f"dim_{i}" for i in range(values.shape[1])]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=stride_s, sampling_rate_hz=1.0 / stride_s),
    )


def _transformers_audio_embedding(
    stimulus: AudioStimulus,
    *,
    extractor_name: str,
    model: str,
    stride_s: float,
    dim: int,
    local_files_only: bool,
    params: dict[str, object],
    execution_mode: str,
) -> FeatureSeries:
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be > 0")
    import torch  # type: ignore
    from transformers import AutoFeatureExtractor, AutoModel  # type: ignore

    fe = AutoFeatureExtractor.from_pretrained(model, local_files_only=local_files_only)
    net = AutoModel.from_pretrained(model, local_files_only=local_files_only)
    wav = stimulus.samples.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    inputs = fe(wav, sampling_rate=stimulus.sr_hz, return_tensors="pt")
    with torch.no_grad():
        out = net(**inputs)
    arr = getattr(out, "last_hidden_state", None)
    if arr is None:
        arr = out[0]
    pooled = arr.detach().cpu().numpy().astype(np.float32)
    if pooled.ndim == 3:
        pooled = pooled.mean(axis=1)
    values = _fit_dim(pooled.reshape(pooled.shape[0], -1), dim)
    md = add_execution_provenance(
        extractor_metadata(extractor_name, params=params, extra={"backend": "transformers_local"}),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values,
        times_s=stimulus.start_offset_s + np.arange(values.shape[0], dtype=np.float64) * float(stride_s),
        dims=("time", "feature"),
        coords={"feature": [f"dim_{i}" for i in range(values.shape[1])]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_embedding"),
    )


def audio_clap_embeddings(
    stimulus: AudioStimulus,
    *,
    model: str = "laion/clap-htsat-unfused",
    stride_s: float = 1.0,
    dim: int = 64,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    params = {"model": model, "stride_s": stride_s, "dim": dim, "local_files_only": local_files_only}
    try:
        return _transformers_audio_embedding(
            stimulus,
            extractor_name="audio.clap",
            model=model,
            stride_s=stride_s,
            dim=dim,
            local_files_only=local_files_only,
            params=params,
            execution_mode=mode,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError("CLAP extraction failed in strict mode.") from exc
        return _fallback_audio_embedding(
            stimulus,
            extractor_name="audio.clap",
            stride_s=stride_s,
            dim=dim,
            execution_mode=mode,
            reason=f"CLAP backend unavailable: {type(exc).__name__}",
            params=params,
        )


def audio_ast_embeddings(
    stimulus: AudioStimulus,
    *,
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    stride_s: float = 1.0,
    dim: int = 64,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    params = {"model": model, "stride_s": stride_s, "dim": dim, "local_files_only": local_files_only}
    try:
        return _transformers_audio_embedding(
            stimulus,
            extractor_name="audio.ast",
            model=model,
            stride_s=stride_s,
            dim=dim,
            local_files_only=local_files_only,
            params=params,
            execution_mode=mode,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError("AST extraction failed in strict mode.") from exc
        return _fallback_audio_embedding(
            stimulus,
            extractor_name="audio.ast",
            stride_s=stride_s,
            dim=dim,
            execution_mode=mode,
            reason=f"AST backend unavailable: {type(exc).__name__}",
            params=params,
        )
