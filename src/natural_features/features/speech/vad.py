"""Speech presence baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.common import extractor_metadata
from natural_features.features.audio.lowlevel import _frames, _mono


_SILERO_SAMPLE_RATES = (8000, 16000)


def energy_vad(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.02,
    win_s: float = 0.03,
    threshold: float = 0.5,
) -> FeatureSeries:
    x = _mono(stimulus.samples)
    framed, _ = _frames(x, stimulus.sr_hz, hop_s, win_s)
    energy = np.sqrt(np.mean(framed * framed, axis=1))
    if len(energy) == 0:
        prob = energy
    else:
        norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        prob = np.clip(norm / max(threshold, 1e-6), 0.0, 1.0)
    vals = prob[:, None].astype(np.float32)
    times = times_from_hop(vals.shape[0], hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    metadata = extractor_metadata(
        "speech.vad.energy_vad",
        params={"hop_s": hop_s, "win_s": win_s, "threshold": threshold},
    )
    return FeatureSeries(
        values=vals,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": ["speech_probability"]},
        metadata=metadata,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def speech_vad(
    stimulus: AudioStimulus,
    *,
    frame_s: float = 0.02,
    win_s: float = 0.03,
    threshold: float = 0.5,
) -> EventSeries:
    """Return contiguous speech events from the energy VAD probability series."""

    if frame_s <= 0:
        raise ValueError("frame_s must be > 0")
    if win_s <= 0:
        raise ValueError("win_s must be > 0")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    base = energy_vad(stimulus, hop_s=frame_s, win_s=win_s, threshold=threshold)
    prob = base.values[:, 0]
    active = prob >= float(threshold)
    starts: list[int] = []
    stops: list[int] = []
    in_run = False
    for i, flag in enumerate(active):
        if flag and not in_run:
            starts.append(i)
            in_run = True
        elif not flag and in_run:
            stops.append(i)
            in_run = False
    if in_run:
        stops.append(len(active))

    onset = []
    offset = []
    confidence = []
    half_win = float(win_s) / 2.0
    for start, stop in zip(starts, stops, strict=True):
        onset.append(max(float(stimulus.start_offset_s), float(base.times_s[start]) - half_win))
        offset.append(float(base.times_s[stop - 1]) + half_win)
        confidence.append(float(prob[start:stop].mean()))

    md = extractor_metadata(
        "speech.vad",
        params={"frame_s": frame_s, "win_s": win_s, "threshold": threshold},
        extra={"backend": "energy_threshold", "source_extractor": base.metadata.get("extractor_name", "unknown")},
    )
    return EventSeries(
        onset_s=np.asarray(onset, dtype=np.float64),
        offset_s=np.asarray(offset, dtype=np.float64),
        label=np.asarray(["speech"] * len(onset), dtype=object),
        confidence=np.asarray(confidence, dtype=np.float32),
        extra={
            "frame_start": np.asarray(starts, dtype=np.int64),
            "frame_stop": np.asarray(stops, dtype=np.int64),
        },
        metadata=md,
        timebase=TimebaseSpec(kind="events"),
    )


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32)
    duration_s = len(x) / float(src_sr)
    n_out = max(1, int(round(duration_s * dst_sr)))
    src_t = np.arange(len(x), dtype=np.float64) / float(src_sr)
    dst_t = np.arange(n_out, dtype=np.float64) / float(dst_sr)
    return np.interp(dst_t, src_t, x.astype(np.float32)).astype(np.float32)


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return float(value.item())
    arr = np.asarray(value, dtype=np.float32)
    return float(arr.reshape(-1)[0])


def _load_silero_backend(model: str, *, local_files_only: bool) -> tuple[Any, str]:
    model_path = Path(model).expanduser()
    if model_path.exists():
        import torch  # type: ignore

        if model_path.is_file():
            return torch.jit.load(str(model_path)), "silero_torchscript"
        loaded, _utils = torch.hub.load(str(model_path), "silero_vad", source="local")
        return loaded, "silero_torchhub_local"
    if model in {"silero", "silero_vad", "package"}:
        try:
            from silero_vad import load_silero_vad  # type: ignore
        except Exception as exc:
            if local_files_only:
                raise RuntimeError("silero_vad package is not installed.") from exc
        else:
            return load_silero_vad(), "silero_vad_package"
    if local_files_only:
        raise RuntimeError(f"Silero VAD model '{model}' is not available locally.")

    import torch  # type: ignore

    repo = "snakers4/silero-vad" if model in {"silero", "silero_vad", "package"} else model
    loaded, _utils = torch.hub.load(repo_or_dir=repo, model="silero_vad")
    return loaded, "silero_torchhub"


def _silero_probabilities(
    stimulus: AudioStimulus,
    *,
    model: str,
    hop_s: float,
    win_s: float,
    local_files_only: bool,
    execution_mode: str,
    params: dict[str, object],
) -> FeatureSeries:
    import torch  # type: ignore

    backend_model, backend = _load_silero_backend(model, local_files_only=local_files_only)
    if hasattr(backend_model, "reset_states"):
        backend_model.reset_states()

    target_sr = 8000 if stimulus.sr_hz <= 8000 else 16000
    if target_sr not in _SILERO_SAMPLE_RATES:  # defensive; target_sr is intentionally constrained above.
        target_sr = 16000
    wav = np.nan_to_num(_mono(stimulus.samples), copy=False).astype(np.float32)
    wav = _resample_linear(wav, stimulus.sr_hz, target_sr)
    window = 512 if target_sr == 16000 else 256
    n_chunks = max(1, int(np.ceil(len(wav) / float(window))))
    probs = np.empty(n_chunks, dtype=np.float32)
    for i in range(n_chunks):
        start = i * window
        chunk = wav[start : start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk))).astype(np.float32)
        probs[i] = np.clip(_as_float(backend_model(torch.from_numpy(chunk.astype(np.float32)), target_sr)), 0.0, 1.0)
    if hasattr(backend_model, "reset_states"):
        backend_model.reset_states()

    native_hop_s = window / float(target_sr)
    native_times = stimulus.start_offset_s + (np.arange(n_chunks, dtype=np.float64) + 0.5) * native_hop_s
    duration_s = len(stimulus.samples) / float(stimulus.sr_hz)
    n_out = max(1, int(np.ceil(duration_s / max(hop_s, 1e-8))))
    out_times = times_from_hop(n_out, hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=win_s)
    if len(probs) == 1:
        out_probs = np.full(n_out, float(probs[0]), dtype=np.float32)
    else:
        out_probs = np.interp(out_times, native_times, probs, left=probs[0], right=probs[-1]).astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(
            "speech.neural_vad",
            params=params,
            extra={"backend": backend, "native_hop_s": native_hop_s, "model_sample_rate_hz": target_sr},
        ),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=out_probs[:, None].astype(np.float32),
        times_s=out_times,
        dims=("time", "feature"),
        coords={"feature": ["speech_probability"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def _fallback_neural_vad(
    stimulus: AudioStimulus,
    *,
    model: str,
    hop_s: float,
    win_s: float,
    execution_mode: str,
    params: dict[str, object],
) -> FeatureSeries:
    base = energy_vad(stimulus, hop_s=hop_s, win_s=win_s)
    prob = base.values[:, 0]
    smooth = np.convolve(prob, np.ones(3, dtype=np.float32) / 3.0, mode="same") if prob.size else prob
    values = smooth[:, None].astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata(
            "speech.neural_vad",
            params=params,
            extra={"backend": "energy_proxy"},
        ),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason="neural VAD backend unavailable",
    )
    return FeatureSeries(
        values=values,
        times_s=base.times_s,
        dims=("time", "feature"),
        coords={"feature": ["speech_probability"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def neural_vad(
    stimulus: AudioStimulus,
    *,
    model: str = "silero_vad",
    hop_s: float = 0.02,
    win_s: float = 0.03,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return neural-VAD speech probabilities with a deterministic fallback."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    params: dict[str, object] = {
        "model": model,
        "hop_s": hop_s,
        "win_s": win_s,
        "local_files_only": local_files_only,
    }
    try:
        return _silero_probabilities(
            stimulus,
            model=model,
            hop_s=hop_s,
            win_s=win_s,
            local_files_only=local_files_only,
            execution_mode=mode,
            params=params,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError("speech neural VAD extraction failed in strict mode.") from exc
        return _fallback_neural_vad(
            stimulus,
            model=model,
            hop_s=hop_s,
            win_s=win_s,
            execution_mode=mode,
            params=params,
        )
