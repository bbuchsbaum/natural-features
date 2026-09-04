"""Optional Churchwell et al. (2024) phonetic-posteriorgram backend."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

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
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.phonology import _resample_audio_linear

PPGS_SAMPLE_RATE_HZ = 16000
PPGS_HOP_SAMPLES = 160
PPGS_HOP_S = PPGS_HOP_SAMPLES / float(PPGS_SAMPLE_RATE_HZ)
PPGS_CHECKPOINT_REPO = "CameronChurchwell/ppgs"
PPGS_REPRESENTATIONS = ("mel", "w2v2fb")
PPGS_CHECKPOINTS = {
    "mel": "mel-800k.pt",
    "w2v2fb": "w2v2fb-425k.pt",
}

# CMU ARPABET inventory used by interactiveaudiolab/ppgs, with silence last.
# Upstream stores silence as pypar.SILENCE (a single space); we emit "silence".
PPGS_PHONE_LABELS: list[str] = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "silence",
]
PPGS_SILENCE_INDEX = len(PPGS_PHONE_LABELS) - 1


def _mono_float(samples: np.ndarray) -> np.ndarray:
    wav = np.asarray(samples, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _frame_count(n_samples: int, hop_samples: int = PPGS_HOP_SAMPLES) -> int:
    return max(1, int(n_samples) // int(hop_samples))


def leading_trailing_silence_frames(
    wav: np.ndarray,
    *,
    hop_samples: int = PPGS_HOP_SAMPLES,
    rms_floor: float = 1e-3,
    peak_fraction: float = 0.05,
) -> tuple[int, int]:
    """Count leading/trailing low-energy frames at the PPGs hop.

    Returns ``(0, 0)`` when the clip is entirely below ``rms_floor`` so we do
    not invent a speech island in silence-only audio.
    """

    n_frames = _frame_count(len(wav), hop_samples)
    rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        sl = wav[i * hop_samples : (i + 1) * hop_samples]
        if sl.size:
            rms[i] = float(np.sqrt(np.mean(np.square(sl, dtype=np.float64))))
    peak = float(rms.max()) if rms.size else 0.0
    if peak < rms_floor:
        return 0, 0
    thresh = max(rms_floor, peak_fraction * peak)
    speech = rms >= thresh
    if not np.any(speech):
        return 0, 0
    first = int(np.argmax(speech))
    last = int(n_frames - 1 - np.argmax(speech[::-1]))
    return first, n_frames - 1 - last


def pad_posteriors_with_silence(
    probs: np.ndarray,
    *,
    lead_frames: int,
    trail_frames: int,
    silence_index: int = PPGS_SILENCE_INDEX,
) -> np.ndarray:
    """Restore trimmed frames as one-hot silence rows."""

    if probs.ndim != 2:
        raise ValueError("probs must be a (time, phone) array")
    n_phones = int(probs.shape[1])
    if silence_index < 0 or silence_index >= n_phones:
        raise ValueError("silence_index is out of range")
    silence = np.zeros((1, n_phones), dtype=np.float32)
    silence[0, silence_index] = 1.0
    parts: list[np.ndarray] = []
    if lead_frames > 0:
        parts.append(np.repeat(silence, int(lead_frames), axis=0))
    parts.append(np.asarray(probs, dtype=np.float32))
    if trail_frames > 0:
        parts.append(np.repeat(silence, int(trail_frames), axis=0))
    return np.concatenate(parts, axis=0)


def _resolve_ppgs_checkpoint(
    representation: str,
    checkpoint: str | None,
    *,
    local_files_only: bool,
) -> str:
    if checkpoint is not None:
        path = str(checkpoint).strip()
        if not path:
            raise ValueError("checkpoint cannot be empty")
        return path
    filename = PPGS_CHECKPOINTS.get(representation)
    if filename is None:
        raise ValueError(f"no default checkpoint for representation {representation!r}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise BackendDependencyError(
            "ppgs", "huggingface-hub is required to resolve checkpoints"
        ) from exc
    try:
        return str(
            hf_hub_download(
                PPGS_CHECKPOINT_REPO,
                filename,
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        raise BackendLoadError(
            "ppgs",
            f"checkpoint '{filename}' from '{PPGS_CHECKPOINT_REPO}' is unavailable",
        ) from exc


@contextmanager
def _temporary_chunk_length(chunk_length: int | None) -> Iterator[None]:
    if chunk_length is None:
        yield
        return
    import ppgs

    previous = getattr(ppgs, "CHUNK_LENGTH", None)
    ppgs.CHUNK_LENGTH = int(chunk_length)
    try:
        yield
    finally:
        if previous is None:
            delattr(ppgs, "CHUNK_LENGTH")
        else:
            ppgs.CHUNK_LENGTH = previous


def _infer_ppg_matrix(
    wav: np.ndarray,
    sample_rate: int,
    *,
    representation: str,
    checkpoint: str,
    gpu: int | None,
    legacy_mode: bool,
    chunk_length: int | None,
) -> np.ndarray:
    import torch
    import ppgs

    audio = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).view(1, 1, -1)
    with _temporary_chunk_length(chunk_length):
        out = ppgs.from_audio(
            audio,
            sample_rate,
            representation=representation,
            checkpoint=checkpoint,
            gpu=gpu,
            legacy_mode=legacy_mode,
        )
    arr = np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] != len(PPGS_PHONE_LABELS):
        raise BackendInferenceError(
            "ppgs",
            f"unexpected posterior shape {arr.shape}; expected ({len(PPGS_PHONE_LABELS)}, frames)",
        )
    arr = arr.T
    arr = np.clip(arr, 0.0, None)
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-8)
    return arr.astype(np.float32)


def _trim_and_infer(
    wav: np.ndarray,
    *,
    representation: str,
    checkpoint: str,
    gpu: int | None,
    legacy_mode: bool,
    chunk_length: int | None,
    trim_edge_silence: bool,
) -> tuple[np.ndarray, int, int]:
    """Infer PPGs, optionally trimming edge silence and restoring the timeline.

    Churchwell confirmed that leading/trailing silence is out of distribution
    for the released checkpoints (interactiveaudiolab/ppgs#18). That is a
    training-data issue, not a one-line inference bug, so we do not fork the
    library. Trimming then padding silence-class frames keeps the original
    hop grid without retraining.
    """

    lead = trail = 0
    infer_wav = wav
    if trim_edge_silence:
        lead, trail = leading_trailing_silence_frames(wav)
        lead_samples = lead * PPGS_HOP_SAMPLES
        trail_samples = trail * PPGS_HOP_SAMPLES
        stop = len(wav) - trail_samples
        if stop > lead_samples:
            infer_wav = wav[lead_samples:stop]
        else:
            lead = trail = 0
            infer_wav = wav
    probs = _infer_ppg_matrix(
        infer_wav,
        PPGS_SAMPLE_RATE_HZ,
        representation=representation,
        checkpoint=checkpoint,
        gpu=gpu,
        legacy_mode=legacy_mode,
        chunk_length=chunk_length,
    )
    if not trim_edge_silence:
        return probs, 0, 0
    expected = _frame_count(len(wav))
    pad_lead = lead
    pad_trail = max(0, expected - probs.shape[0] - pad_lead)
    out = pad_posteriors_with_silence(
        probs, lead_frames=pad_lead, trail_frames=pad_trail
    )
    if out.shape[0] < expected:
        out = pad_posteriors_with_silence(
            out,
            lead_frames=0,
            trail_frames=expected - out.shape[0],
        )
    elif out.shape[0] > expected:
        out = out[:expected]
    return out, pad_lead, pad_trail


def ppg_phone_posteriors(
    stimulus: AudioStimulus,
    *,
    representation: str = "mel",
    checkpoint: str | None = None,
    gpu: int | None = None,
    local_files_only: bool = True,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
    trim_edge_silence: bool = True,
    legacy_mode: bool = False,
    chunk_length: int | None = None,
) -> FeatureSeries:
    """Framewise CMU-phone posteriors from interactiveaudiolab/ppgs.

    Parameters
    ----------
    representation:
        Upstream frontend. ``mel`` is the pip-default checkpoint; ``w2v2fb``
        is the more accurate wav2vec 2.0 frontend from the paper.
    checkpoint:
        Local checkpoint path. When omitted, the matching file is resolved
        from ``CameronChurchwell/ppgs`` honoring ``local_files_only``.
    gpu:
        CUDA device index, or ``None`` for CPU.
    trim_edge_silence:
        If true, run inference on the energy-trimmed interior and restore
        leading/trailing frames as the silence class. This is the in-wrapper
        mitigation for ppgs issue #18.
    legacy_mode:
        Forwarded to ``ppgs.from_audio`` (unchunked inference).
    chunk_length:
        Temporary override of ``ppgs.CHUNK_LENGTH`` during inference.
    """

    mode, _strict = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=strict_dependency,
    )
    representation = str(representation).strip().lower()
    if representation not in PPGS_REPRESENTATIONS:
        raise ValueError(f"representation must be one of {PPGS_REPRESENTATIONS}")
    if chunk_length is not None and int(chunk_length) <= 0:
        raise ValueError("chunk_length must be > 0 when provided")

    params: dict[str, Any] = {
        "representation": representation,
        "checkpoint": checkpoint,
        "gpu": gpu,
        "local_files_only": local_files_only,
        "trim_edge_silence": trim_edge_silence,
        "legacy_mode": legacy_mode,
        "chunk_length": chunk_length,
        "hop_s": PPGS_HOP_S,
        "sample_rate_hz": PPGS_SAMPLE_RATE_HZ,
    }

    try:
        import torch  # noqa: F401
        import ppgs  # noqa: F401
    except ImportError as exc:
        raise BackendDependencyError(
            "ppgs", "the ppgs package and torch are required"
        ) from exc

    checkpoint_path = _resolve_ppgs_checkpoint(
        representation,
        checkpoint,
        local_files_only=local_files_only,
    )
    try:
        wav = _mono_float(stimulus.samples)
        wav = _resample_audio_linear(
            wav, from_sr=int(stimulus.sr_hz), to_sr=PPGS_SAMPLE_RATE_HZ
        )
        probs, lead_frames, trail_frames = _trim_and_infer(
            wav,
            representation=representation,
            checkpoint=checkpoint_path,
            gpu=gpu,
            legacy_mode=legacy_mode,
            chunk_length=None if chunk_length is None else int(chunk_length),
            trim_edge_silence=trim_edge_silence,
        )
    except (BackendDependencyError, BackendLoadError, BackendInferenceError):
        raise
    except Exception as exc:
        raise BackendInferenceError("ppgs", "posterior inference failed") from exc

    times = times_from_hop(
        probs.shape[0], PPGS_HOP_S, start_offset_s=stimulus.start_offset_s
    )
    md = add_execution_provenance(
        extractor_metadata(
            "speech.phonology.ppg_posteriors",
            params=params,
            extra={
                "backend": "ppgs",
                "label_namespace": "arpabet",
                "namespace_version": "cmu-40",
                "checkpoint_path": checkpoint_path,
                "trimmed_lead_frames": int(lead_frames),
                "trimmed_trail_frames": int(trail_frames),
            },
        ),
        execution_mode=mode,
        fallback_used=False,
        backend="ppgs",
    )
    return FeatureSeries(
        values=probs.astype(np.float32),
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": list(PPGS_PHONE_LABELS)},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=PPGS_HOP_S, sampling_rate_hz=1.0 / PPGS_HOP_S
        ),
    )
