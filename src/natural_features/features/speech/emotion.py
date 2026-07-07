"""Speech emotion feature wrappers."""

from __future__ import annotations

import numpy as np

from natural_features.core.execution import add_execution_provenance, resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio.prosody import prosody_features
from natural_features.features.common import extractor_metadata


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.nanmax(logits)
    exp = np.exp(logits).astype(np.float32)
    return exp / np.maximum(exp.sum(dtype=np.float32), 1e-8)


def _transformers_emotion(
    stimulus: AudioStimulus,
    *,
    model: str,
    local_files_only: bool,
    execution_mode: str,
    params: dict[str, object],
) -> FeatureSeries:
    import torch  # type: ignore
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  # type: ignore

    fe = AutoFeatureExtractor.from_pretrained(model, local_files_only=local_files_only)
    net = AutoModelForAudioClassification.from_pretrained(model, local_files_only=local_files_only)
    wav = stimulus.samples.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    inputs = fe(wav, sampling_rate=stimulus.sr_hz, return_tensors="pt")
    with torch.no_grad():
        out = net(**inputs)
    logits = getattr(out, "logits", None)
    if logits is None:
        logits = out[0]
    arr = logits.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    probs = _softmax(arr)
    id2label = getattr(getattr(net, "config", None), "id2label", None) or {}
    labels = [str(id2label.get(i, id2label.get(str(i), f"emotion_{i}"))) for i in range(probs.shape[0])]
    md = add_execution_provenance(
        extractor_metadata(
            "speech.emotion",
            params=params,
            extra={"backend": "transformers_audio_classification", "labels": labels},
        ),
        execution_mode=execution_mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=probs.reshape(1, -1).astype(np.float32),
        times_s=np.asarray([stimulus.start_offset_s], dtype=np.float64),
        dims=("time", "feature"),
        coords={"feature": labels},
        metadata=md,
        timebase=TimebaseSpec(kind="audio_summary"),
    )


def _fallback_emotion(stimulus: AudioStimulus, *, hop_s: float, execution_mode: str, params: dict[str, object]) -> FeatureSeries:
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
        extractor_metadata("speech.emotion", params=params, extra={"backend": "prosody_proxy"}),
        execution_mode=execution_mode,
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


def speech_emotion(
    stimulus: AudioStimulus,
    *,
    model: str = "superb/wav2vec2-base-superb-er",
    hop_s: float = 0.02,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Return speech-emotion features with strict/fallback semantics."""

    mode, strict = resolve_execution_mode(execution_mode=execution_mode, strict_dependency=strict_dependency)
    params: dict[str, object] = {"model": model, "hop_s": hop_s, "local_files_only": local_files_only}
    try:
        return _transformers_emotion(
            stimulus,
            model=model,
            local_files_only=local_files_only,
            execution_mode=mode,
            params=params,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError("speech emotion extraction failed in strict mode.") from exc
        return _fallback_emotion(stimulus, hop_s=hop_s, execution_mode=mode, params=params)
