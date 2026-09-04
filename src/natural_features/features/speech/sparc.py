"""Optional SPARC acoustic-to-articulatory inversion wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path
import wave
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
from natural_features.core.timebase import TimebaseSpec, times_from_hop
from natural_features.features.audio.lowlevel import _mono
from natural_features.features.common import extractor_metadata

SPARC_HOP_S = 0.02
SPARC_EMA_LABELS = [
    "TDX",
    "TDY",
    "TBX",
    "TBY",
    "TTX",
    "TTY",
    "LIX",
    "LIY",
    "ULX",
    "ULY",
    "LLX",
    "LLY",
]


def _write_temp_wav(stimulus: AudioStimulus, path: Path) -> None:
    samples = _mono(stimulus.samples)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(stimulus.sr_hz))
        handle.writeframes(pcm.tobytes())


def _coerce_ema(payload: Any) -> np.ndarray:
    if isinstance(payload, dict):
        if "ema" not in payload:
            raise BackendInferenceError("sparc", "SPARC encode output missing 'ema'")
        ema = np.asarray(payload["ema"], dtype=np.float32)
    else:
        ema = np.asarray(payload, dtype=np.float32)
    if ema.ndim != 2 or ema.shape[1] != 12:
        raise BackendInferenceError(
            "sparc",
            f"SPARC ema must have shape (frames, 12); got {tuple(ema.shape)}",
        )
    return ema


def sparc_articulatory(
    stimulus: AudioStimulus,
    *,
    model: str = "feature_extraction",
    device: str = "cpu",
    local_files_only: bool = True,
    include_aux: bool = False,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
) -> FeatureSeries:
    """Invert audio to SPARC 12-channel template EMA at 50 Hz.

    This is a learned articulatory code, not measured EMA. SPARC has no velum
    or larynx channels.
    """

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode, strict_dependency=strict_dependency
    )
    try:
        from sparc import load_model  # type: ignore
    except ImportError as exc:
        raise BackendDependencyError(
            "sparc",
            "speech-articulatory-coding is required (pip install -e '.[sparc]')",
        ) from exc
    try:
        coder = load_model(model, device=device)
    except Exception as exc:
        if local_files_only:
            raise BackendLoadError(
                "sparc",
                f"Failed to load SPARC model {model!r} with local_files_only=True",
            ) from exc
        raise BackendLoadError(
            "sparc", f"Failed to load SPARC model {model!r}"
        ) from exc

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "clip.wav"
        _write_temp_wav(stimulus, wav_path)
        try:
            payload = coder.encode(str(wav_path))
        except Exception as exc:
            raise BackendInferenceError("sparc", "SPARC encode failed") from exc
    ema = _coerce_ema(payload)
    names = list(SPARC_EMA_LABELS)
    values = ema
    if include_aux and isinstance(payload, dict):
        aux_cols = []
        for key in ("loudness", "pitch", "periodicity"):
            if key in payload:
                col = np.asarray(payload[key], dtype=np.float32)
                if col.ndim == 1:
                    col = col.reshape(-1, 1)
                if col.shape[0] != ema.shape[0]:
                    raise BackendInferenceError(
                        "sparc",
                        f"SPARC {key} length {col.shape[0]} != ema frames {ema.shape[0]}",
                    )
                aux_cols.append(col)
                names.append(key)
        if aux_cols:
            values = np.concatenate([ema] + aux_cols, axis=1)
    times = times_from_hop(
        values.shape[0],
        SPARC_HOP_S,
        start_offset_s=stimulus.start_offset_s,
        center=True,
        window_s=SPARC_HOP_S,
    )
    md = add_execution_provenance(
        extractor_metadata(
            "speech.articulatory.sparc",
            params={
                "model": model,
                "device": device,
                "local_files_only": local_files_only,
                "include_aux": include_aux,
            },
            extra={"backend": "sparc_template_ema"},
        ),
        execution_mode=mode,
        fallback_used=False,
    )
    return FeatureSeries(
        values=values.astype(np.float32),
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=SPARC_HOP_S, sampling_rate_hz=1.0 / SPARC_HOP_S
        ),
    )
