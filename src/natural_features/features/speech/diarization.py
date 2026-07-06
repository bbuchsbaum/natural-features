"""Speaker diarization feature wrappers."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import TrackSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.vad import neural_vad


def speaker_diarization(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.02,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> TrackSeries:
    """Return speaker activity tracks; fallback mode emits one speech track."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import pyannote.audio  # type: ignore  # noqa: F401
    except Exception as exc:
        if strict:
            raise RuntimeError("pyannote.audio is required for strict speaker diarization.") from exc
    else:
        if strict:
            raise RuntimeError("strict speaker diarization requires a configured pyannote pipeline.")
    vad = neural_vad(stimulus, hop_s=hop_s, strict_dependency=False)
    values = vad.values[:, None, :].astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata("speech.diarization", params={"hop_s": hop_s}, extra={"backend": "single_speaker_proxy"}),
        execution_mode=mode,
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
