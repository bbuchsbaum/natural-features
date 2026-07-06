"""Speech emotion feature wrappers."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio.prosody import prosody_features
from natural_features.features.common import extractor_metadata


def speech_emotion(
    stimulus: AudioStimulus,
    *,
    hop_s: float = 0.02,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return framewise speech-emotion proxies with strict/fallback semantics."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    try:
        import torch  # type: ignore  # noqa: F401
        import transformers  # type: ignore  # noqa: F401
    except Exception as exc:
        if strict:
            raise RuntimeError("transformers+torch are required for strict speech emotion extraction.") from exc
    else:
        if strict:
            raise RuntimeError(
                "strict speech emotion backend is not configured; use fallback mode or provide a supported model."
            )
    pros = prosody_features(stimulus, hop_s=hop_s, win_s=max(0.03, hop_s * 2))
    x = pros.values.astype(np.float32)
    if x.size == 0:
        values = np.zeros((0, 4), dtype=np.float32)
    else:
        rms = x[:, 0]
        f0 = x[:, 2]
        voiced = x[:, 3]
        centroid = x[:, 4]

        def norm(v: np.ndarray) -> np.ndarray:
            return (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-8)

        arousal = np.clip(0.55 * norm(rms) + 0.45 * norm(centroid), 0.0, 1.0)
        valence = np.clip(0.5 + 0.25 * norm(f0) - 0.2 * norm(centroid), 0.0, 1.0)
        dominance = np.clip(0.5 * norm(rms) + 0.5 * voiced, 0.0, 1.0)
        values = np.column_stack([arousal, valence, dominance, voiced]).astype(np.float32)
    md = add_execution_provenance(
        extractor_metadata("speech.emotion", params={"hop_s": hop_s}, extra={"backend": "prosody_proxy"}),
        execution_mode=mode,
        fallback_used=True,
        fallback_reason="speech emotion model unavailable",
    )
    return FeatureSeries(
        values=values,
        times_s=pros.times_s,
        dims=("time", "feature"),
        coords={"feature": ["arousal", "valence", "dominance", "voicing_strength"]},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s),
    )
