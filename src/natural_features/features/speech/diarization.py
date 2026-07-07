"""Speaker diarization feature wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import TrackSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _mono
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.vad import neural_vad


def _pyannote_annotation(result: Any) -> Any:
    return getattr(result, "speaker_diarization", result)


def _iter_pyannote_turns(annotation: Any) -> list[tuple[float, float, str]]:
    turns = []
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        start = float(getattr(segment, "start"))
        end = float(getattr(segment, "end"))
        if end > start:
            turns.append((start, end, str(speaker)))
    turns.sort(key=lambda item: (item[0], item[1], item[2]))
    return turns


def _load_pyannote_pipeline(model: str, *, local_files_only: bool) -> tuple[Any, str]:
    from pyannote.audio import Pipeline  # type: ignore

    model_path = Path(model).expanduser()
    if model_path.exists():
        return Pipeline.from_pretrained(str(model_path)), "pyannote"
    if local_files_only:
        raise RuntimeError(f"pyannote pipeline '{model}' is not available as a local path.")
    return Pipeline.from_pretrained(model), "pyannote"


def _pyannote_diarization(
    stimulus: AudioStimulus,
    *,
    model: str,
    hop_s: float,
    local_files_only: bool,
    device: str | None,
    execution_mode: str,
    params: dict[str, object],
) -> TrackSeries:
    import torch  # type: ignore

    pipeline, backend = _load_pyannote_pipeline(model, local_files_only=local_files_only)
    resolved_device = None
    if device:
        requested = str(device)
        if requested == "auto":
            requested = "cuda" if getattr(torch, "cuda", None) is not None and torch.cuda.is_available() else "cpu"
        resolved_device = requested
        if hasattr(pipeline, "to"):
            pipeline.to(torch.device(requested))

    wav = _mono(stimulus.samples).astype(np.float32)
    audio = {"waveform": torch.from_numpy(wav[None, :]), "sample_rate": int(stimulus.sr_hz)}
    result = pipeline(audio)
    turns = _iter_pyannote_turns(_pyannote_annotation(result))
    speaker_ids = sorted({speaker for _start, _end, speaker in turns})
    duration_s = len(stimulus.samples) / float(stimulus.sr_hz)
    n_frames = max(1, int(np.ceil(duration_s / max(hop_s, 1e-8))))
    times = times_from_hop(n_frames, hop_s, start_offset_s=stimulus.start_offset_s, center=True, window_s=hop_s)
    values = np.zeros((n_frames, len(speaker_ids), 1), dtype=np.float32)
    rel_centers = times - float(stimulus.start_offset_s)
    frame_start = np.maximum(0.0, rel_centers - hop_s / 2.0)
    frame_end = rel_centers + hop_s / 2.0
    speaker_index = {speaker: i for i, speaker in enumerate(speaker_ids)}
    for start, end, speaker in turns:
        idx = speaker_index[speaker]
        mask = (frame_end > start) & (frame_start < end)
        values[mask, idx, 0] = 1.0

    md = add_execution_provenance(
        extractor_metadata(
            "speech.diarization",
            params=params,
            extra={
                "backend": backend,
                "speaker_labels": speaker_ids,
                "turn_count": len(turns),
                "device": resolved_device,
            },
        ),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return TrackSeries(
        times_s=times,
        track_id=np.asarray(speaker_ids, dtype=object),
        values=values,
        dims=("time", "track", "feature"),
        coords={"feature": ["speaker_activity"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def _fallback_diarization(
    stimulus: AudioStimulus,
    *,
    model: str,
    hop_s: float,
    local_files_only: bool,
    device: str | None,
    execution_mode: str,
) -> TrackSeries:
    params: dict[str, object] = {
        "model": model,
        "hop_s": hop_s,
        "local_files_only": local_files_only,
        "device": device,
    }
    vad = neural_vad(stimulus, hop_s=hop_s, local_files_only=True, execution_mode="fallback", strict_dependency=False)
    values = vad.values[:, None, :].astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata("speech.diarization", params=params, extra={"backend": "single_speaker_proxy"}),
        execution_mode=execution_mode,
        fallback_used=True,
        fallback_reason="pyannote pipeline unavailable",
    )
    return TrackSeries(
        times_s=vad.times_s,
        track_id=np.asarray(["speaker_0"], dtype=object),
        values=values,
        dims=("time", "track", "feature"),
        coords={"feature": ["speaker_probability"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )


def speaker_diarization(
    stimulus: AudioStimulus,
    *,
    model: str = "pyannote/speaker-diarization-community-1",
    hop_s: float = 0.02,
    local_files_only: bool = True,
    device: str | None = "cpu",
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> TrackSeries:
    """Return speaker activity tracks; fallback mode emits one speech track."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    params: dict[str, object] = {
        "model": model,
        "hop_s": hop_s,
        "local_files_only": local_files_only,
        "device": device,
    }
    try:
        return _pyannote_diarization(
            stimulus,
            model=model,
            hop_s=hop_s,
            local_files_only=local_files_only,
            device=device,
            execution_mode=mode,
            params=params,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError("speaker diarization failed in strict mode.") from exc
        return _fallback_diarization(
            stimulus,
            model=model,
            hop_s=hop_s,
            local_files_only=local_files_only,
            device=device,
            execution_mode=mode,
        )
