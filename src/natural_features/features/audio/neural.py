"""Model-specific, clip-level audio neural embedding extractors."""

from __future__ import annotations

from typing import Any

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendInferenceError,
    BackendLoadError,
)
from natural_features.core.execution import (
    add_execution_provenance,
    resolve_execution_mode,
)
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import SupportSpec, TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.audio._music_contract import computation_context


def _mono_waveform(stimulus: AudioStimulus) -> np.ndarray:
    waveform = stimulus.samples.astype(np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform


def _numpy(value: Any) -> np.ndarray:
    # transformers >= 5 returns a ModelOutput (e.g. BaseModelOutputWithPooling) where
    # earlier versions returned the projection tensor directly. Unwrap it before
    # casting, or np.asarray sees an object with no numeric interpretation.
    if not hasattr(value, "detach") and not isinstance(value, np.ndarray):
        for attr in ("audio_embeds", "pooler_output", "last_hidden_state"):
            inner = getattr(value, attr, None)
            if inner is not None:
                value = inner
                break
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
        raise BackendInferenceError(
            backend, "model returned non-finite embedding values"
        )
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


def _require_processor_sample_rate(
    processor: Any, stimulus: AudioStimulus, backend: str
) -> None:
    """Fail loudly on a sample-rate mismatch instead of resampling silently.

    CLAP expects 48 kHz and AST expects 16 kHz. Resampling here would be a hidden
    signal-processing decision, which the strict execution policy rules out, so point
    the caller at ``audio.resample`` instead.
    """

    fe = getattr(processor, "feature_extractor", processor)
    expected = getattr(fe, "sampling_rate", None)
    if expected is None or int(expected) == int(stimulus.sr_hz):
        return
    raise BackendInferenceError(
        backend,
        f"model expects {int(expected)} Hz audio but the stimulus is "
        f"{int(stimulus.sr_hz)} Hz; resample it first with audio.resample "
        f"(target_sr_hz={int(expected)}) rather than relying on an implicit conversion",
    )


def _effective_audio_input(
    stimulus: AudioStimulus,
    processor: Any,
    *,
    backend: str,
    context_policy: str,
    padding: str,
) -> tuple[np.ndarray, AudioStimulus, dict[str, Any]]:
    """Apply an explicit source crop before any native processor truncation."""
    if context_policy not in {"error", "start_crop", "center_crop"}:
        raise ValueError(
            "context_policy must be 'error', 'start_crop' or 'center_crop'"
        )
    wave = _mono_waveform(stimulus)
    if wave.size == 0 or not np.all(np.isfinite(wave)):
        raise ValueError("audio must contain finite samples and be nonempty")
    fe = getattr(processor, "feature_extractor", processor)
    sr = float(stimulus.sr_hz)
    if backend == "CLAP":
        limit_value = getattr(fe, "nb_max_samples", None)
        if limit_value is None:
            seconds = getattr(fe, "max_length_s", None)
            limit_value = None if seconds is None else float(seconds) * sr
        if limit_value is None:
            raise BackendInferenceError(
                backend, "processor does not declare its input capacity"
            )
        if (
            isinstance(limit_value, bool)
            or not np.isfinite(limit_value)
            or float(limit_value) != int(limit_value)
        ):
            raise BackendInferenceError(
                backend, "processor input capacity must be a positive integer"
            )
        limit = int(limit_value)
        capacity = {"model_max_samples": limit}
    else:
        frames = getattr(fe, "max_length", None)
        if frames is None:
            raise BackendInferenceError(
                backend, "processor does not declare max_length"
            )
        if (
            isinstance(frames, bool)
            or not np.isfinite(frames)
            or float(frames) != int(frames)
            or int(frames) < 1
        ):
            raise BackendInferenceError(
                backend, "processor max_length must be a positive integer"
            )
        # The native Transformers AST frontend uses 25 ms frames / 10 ms hops.
        # This is the source interval covered by max_length complete frames.
        limit = int(np.floor((0.025 + (int(frames) - 1) * 0.01) * sr + 1e-8))
        capacity = {
            "model_max_samples": limit,
            "model_max_frames": int(frames),
            "frontend_frame_length_s": 0.025,
            "frontend_frame_shift_s": 0.01,
        }
    if limit <= 0:
        raise BackendInferenceError(
            backend, "processor input capacity must be positive"
        )
    n = len(wave)
    if n > limit and context_policy == "error":
        raise ValueError(
            f"{backend} accepts at most {limit / sr:.6g} seconds per embedding; "
            f"received {n / sr:.6g}. Split the audio explicitly or choose "
            "context_policy='start_crop' or 'center_crop'."
        )
    start = (n - limit) // 2 if n > limit and context_policy == "center_crop" else 0
    used = wave[start : start + min(n, limit)]
    onset = float(stimulus.start_offset_s) + start / sr
    selected = AudioStimulus(
        samples=used,
        sr_hz=stimulus.sr_hz,
        start_offset_s=onset,
        source=stimulus.source,
        clock=stimulus.clock,
        temporal_context=stimulus.temporal_context,
    )
    end = onset + len(used) / sr
    context = computation_context(
        stimulus,
        policy="bidirectional",
        bounds_s=[[onset, end]],
        preprocessing={
            "context_policy": context_policy,
            "supplied_samples": n,
            "used_samples": len(used),
            "crop_start_sample": start,
            "crop_end_sample": start + len(used),
            "padding": padding,
            **capacity,
        },
    )
    return used, selected, context


def _clip_result(
    stimulus: AudioStimulus,
    *,
    values: np.ndarray,
    extractor_name: str,
    params: dict[str, object],
    backend: str,
    representation: str,
    execution_mode: str,
    computation: dict[str, Any] | None = None,
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
                **(
                    {"computation_context": computation}
                    if computation is not None
                    else {}
                ),
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
    context_policy: str = "error",
    padding: str = "repeatpad",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return one native audio projection with explicit effective source context.

    Long inputs raise by default. start_crop/center_crop select a deterministic
    native-size window; output support and computation context name that window."""

    if padding not in {"repeatpad", "repeat", "pad"}:
        raise ValueError("padding must be repeatpad, repeat or pad")
    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    backend = "CLAP"
    params: dict[str, object] = {
        "model": model,
        "dim": dim,
        "local_files_only": local_files_only,
        "context_policy": context_policy,
        "padding": padding,
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

    _require_processor_sample_rate(processor, stimulus, backend)
    wave, selected, context = _effective_audio_input(
        stimulus,
        processor,
        backend=backend,
        context_policy=context_policy,
        padding=padding,
    )
    net.eval()
    try:
        # transformers renamed this keyword from `audios` to `audio` in v5; call the
        # current name first and fall back so both generations keep working.
        try:
            inputs = processor(
                audio=wave,
                sampling_rate=stimulus.sr_hz,
                return_tensors="pt",
                truncation="rand_trunc",
                padding=padding,
            )
        except (TypeError, ValueError):
            inputs = processor(
                audios=wave,
                sampling_rate=stimulus.sr_hz,
                return_tensors="pt",
                truncation="rand_trunc",
                padding=padding,
            )
        with torch.no_grad():
            embedding = net.get_audio_features(**inputs)
    except Exception as exc:
        raise BackendInferenceError(backend, "audio projection failed") from exc
    values = _single_clip_embedding(embedding, backend=backend, dim=dim)
    return _clip_result(
        selected,
        values=values,
        extractor_name="audio.clap",
        params=params,
        backend="transformers_clap",
        representation="audio_projection",
        execution_mode=mode,
        computation=context,
    )


def audio_ast_embeddings(
    stimulus: AudioStimulus,
    *,
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    dim: int | None = None,
    local_files_only: bool = True,
    context_policy: str = "error",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return one native pooled embedding with explicit effective source context.

    Long inputs raise by default. start_crop/center_crop select a deterministic
    window covered by the native spectrogram capacity."""

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    backend = "AST"
    params: dict[str, object] = {
        "model": model,
        "dim": dim,
        "local_files_only": local_files_only,
        "context_policy": context_policy,
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

    _require_processor_sample_rate(feature_extractor, stimulus, backend)
    wave, selected, context = _effective_audio_input(
        stimulus,
        feature_extractor,
        backend=backend,
        context_policy=context_policy,
        padding="zero_fbank",
    )
    net.eval()
    try:
        inputs = feature_extractor(
            wave,
            sampling_rate=stimulus.sr_hz,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = net(**inputs)
    except Exception as exc:
        raise BackendInferenceError(
            backend, "pooled audio representation failed"
        ) from exc
    pooled = getattr(model_output, "pooler_output", None)
    if pooled is None:
        raise BackendInferenceError(backend, "ASTModel did not return pooler_output")
    values = _single_clip_embedding(pooled, backend=backend, dim=dim)
    return _clip_result(
        selected,
        values=values,
        extractor_name="audio.ast",
        params=params,
        backend="transformers_ast",
        representation="pooler_output",
        execution_mode=mode,
        computation=context,
    )
