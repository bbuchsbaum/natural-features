"""High-level acoustic-phonetics workflow (Option 1: posterior -> articulatory)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from natural_features.core.execution import resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.phonology import (
    CTCModelRuntime,
    acoustic_phone_posteriors,
    articulatory_from_posteriors,
    ctc_phone_posteriors,
)
from natural_features.fmri.resample import resample_feature_series


@dataclass
class AcousticPhoneticsResult:
    posteriorgrams: FeatureSeries
    articulatory: FeatureSeries


def extract_acoustic_phonetics(
    audio: AudioStimulus | str | Path,
    *,
    hop_s: float = 0.02,
    posterior_backend: str = "ctc",
    ctc_model: str = "bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
    ctc_local_files_only: bool = True,
    ctc_device: str = "auto",
    ctc_chunk_seconds: float = 30.0,
    ctc_batch_size: int | None = None,
    ctc_runtime: CTCModelRuntime | None = None,
    ctc_progress_callback: Callable[[int, int], None] | None = None,
    execution_mode: str | None = None,
    ctc_strict_dependency: bool | None = None,
    resolution_s: float | None = None,
    resample_method: str = "mean",
    include_uncertainty: bool = True,
    renormalize_posteriors: bool = True,
) -> AcousticPhoneticsResult:
    """Extract time-aligned phone-like posteriors and articulatory probabilities.

    Parameters
    ----------
    audio:
        ``AudioStimulus`` instance or a path to a wav file.
    hop_s:
        Base posterior hop in seconds (used by ``acoustic`` backend).
    posterior_backend:
        ``"ctc"`` (preferred strict backend) or ``"acoustic"`` (lightweight fallback).
    ctc_model:
        Hugging Face CTC model id for posterior extraction.
    ctc_local_files_only:
        If true, only load local model files (no download attempts).
    ctc_device:
        ``auto`` selects CUDA, then MPS, then CPU; an explicit device may be set.
    ctc_chunk_seconds:
        Maximum stimulus duration per CTC inference chunk.
    ctc_batch_size:
        Chunks per forward pass; defaults to four on CUDA, two on MPS, and one
        on CPU.
    ctc_runtime:
        Optional preloaded model runtime shared by all stimuli in a pipeline run.
    ctc_progress_callback:
        Optional callback accepting completed and total chunk counts.
    ctc_strict_dependency:
        If true, fail when transformers/torch/model is unavailable.
    resolution_s:
        Optional output sampling resolution in seconds (for example 0.5, 1.0, 2.0).
        If omitted, features stay on the native posterior hop.
    resample_method:
        Resampling method when ``resolution_s`` is provided.
    include_uncertainty:
        Include ``posterior_entropy`` and ``posterior_peak`` in articulatory output.
    renormalize_posteriors:
        Renormalize posterior rows to sum to one before articulatory mapping.
    """
    mode, ctc_strict_dependency = resolve_execution_mode(
        execution_mode=execution_mode,
        strict_dependency=ctc_strict_dependency,
    )

    stim = audio if isinstance(audio, AudioStimulus) else AudioStimulus.from_wav(audio)
    if posterior_backend == "ctc":
        post = ctc_phone_posteriors(
            stim,
            model=ctc_model,
            local_files_only=ctc_local_files_only,
            execution_mode=mode,
            strict_dependency=ctc_strict_dependency,
            runtime=ctc_runtime,
            device=ctc_device,
            chunk_seconds=ctc_chunk_seconds,
            batch_size=ctc_batch_size,
            progress_callback=ctc_progress_callback,
        )
    elif posterior_backend == "acoustic":
        post = acoustic_phone_posteriors(stim, hop_s=hop_s)
    else:
        raise ValueError("posterior_backend must be one of {'ctc', 'acoustic'}")
    art = articulatory_from_posteriors(
        post,
        renormalize_posteriors=renormalize_posteriors,
        include_uncertainty=include_uncertainty,
    )
    if resolution_s is not None:
        if resolution_s <= 0:
            raise ValueError("resolution_s must be > 0 when provided")
        post = resample_feature_series(post, tr_s=resolution_s, method=resample_method)
        art = resample_feature_series(art, tr_s=resolution_s, method=resample_method)
    return AcousticPhoneticsResult(posteriorgrams=post, articulatory=art)
