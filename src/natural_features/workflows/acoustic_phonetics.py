"""High-level acoustic-phonetics workflow (Option 1: posterior -> articulatory)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from natural_features.core.execution import resolve_execution_mode
from natural_features.core.feature_bundle import inherit_temporal_contract
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.phonology import (
    CTCModelRuntime,
    acoustic_phone_posteriors,
    articulatory_from_posteriors,
    ctc_phone_posteriors,
)
from natural_features.features.speech.ppgs import ppg_phone_posteriors
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
    ctc_chunk_window_s: float = 30.0,
    ctc_chunk_overlap_s: float = 1.0,
    ctc_runtime: CTCModelRuntime | None = None,
    execution_mode: str | None = None,
    ctc_strict_dependency: bool | None = None,
    ppgs_representation: str = "mel",
    ppgs_checkpoint: str | None = None,
    ppgs_gpu: int | None = None,
    ppgs_local_files_only: bool = True,
    ppgs_trim_edge_silence: bool = True,
    ppgs_legacy_mode: bool = False,
    ppgs_chunk_length: int | None = None,
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
        ``"ctc"``, the lightweight ``"acoustic"`` surrogate, or ``"ppgs"``
        (Churchwell et al. 2024 phonetic posteriorgrams).
    ctc_model:
        Hugging Face CTC model id for posterior extraction.
    ctc_local_files_only:
        If true, only load local model files (no download attempts).
    ctc_device:
        ``auto`` selects CUDA, then MPS, then CPU; an explicit device may be set.
    ctc_chunk_window_s:
        Maximum stimulus duration for a single CTC forward pass. Longer audio is
        split with overlap.
    ctc_chunk_overlap_s:
        Overlap between adjacent CTC chunks. Interior overlap frames are dropped.
    ctc_runtime:
        Optional preloaded CTC runtime. When omitted, a process-level cache is used.
    ctc_strict_dependency:
        If true, fail when transformers/torch/model is unavailable.
    ppgs_representation:
        ``ppgs`` frontend: ``"mel"`` (default checkpoint) or ``"w2v2fb"``.
    ppgs_checkpoint:
        Optional local ``ppgs`` checkpoint path.
    ppgs_gpu:
        CUDA index for ``ppgs``, or ``None`` for CPU.
    ppgs_local_files_only:
        If true, only resolve a cached ``ppgs`` checkpoint.
    ppgs_trim_edge_silence:
        Trim leading/trailing silence before ``ppgs`` inference and restore
        those frames as the silence class (mitigation for ppgs issue #18).
    ppgs_legacy_mode:
        Use unchunked ``ppgs`` inference.
    ppgs_chunk_length:
        Optional temporary ``ppgs.CHUNK_LENGTH`` override.
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
            chunk_window_s=ctc_chunk_window_s,
            chunk_overlap_s=ctc_chunk_overlap_s,
        )
    elif posterior_backend == "acoustic":
        post = acoustic_phone_posteriors(stim, hop_s=hop_s)
    elif posterior_backend == "ppgs":
        post = ppg_phone_posteriors(
            stim,
            representation=ppgs_representation,
            checkpoint=ppgs_checkpoint,
            gpu=ppgs_gpu,
            local_files_only=ppgs_local_files_only,
            execution_mode=mode,
            trim_edge_silence=ppgs_trim_edge_silence,
            legacy_mode=ppgs_legacy_mode,
            chunk_length=ppgs_chunk_length,
        )
    else:
        raise ValueError("posterior_backend must be one of {'ctc', 'acoustic', 'ppgs'}")
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
    post = inherit_temporal_contract(post, [stim])
    art = inherit_temporal_contract(art, [stim])
    return AcousticPhoneticsResult(posteriorgrams=post, articulatory=art)
