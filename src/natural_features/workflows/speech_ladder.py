"""Speech representational ladder: A1–A3, P, G, M, and residual bands."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendLoadError,
)
from natural_features.core.execution import resolve_execution_mode
from natural_features.core.feature_bundle import (
    FeatureBundle,
    inherit_temporal_contract,
)
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.envelope import audio_envelope
from natural_features.features.audio.modulation import spectrotemporal_modulation
from natural_features.features.audio.phonetic import audio_formants, audio_harmonicity
from natural_features.features.speech.gestures import (
    articulatory_dynamics,
    articulatory_gestures,
)
from natural_features.features.speech.phonetic_cues import phonetic_cues
from natural_features.features.speech.phonology import (
    acoustic_phone_posteriors,
    ctc_phone_posteriors,
    distinctive_from_posteriors,
)
from natural_features.features.speech.ppgs import ppg_phone_posteriors
from natural_features.features.speech.sparc import sparc_articulatory
from natural_features.features.stats.residualize import residualize_feature_series
from natural_features.fmri.resample import resample_feature_series


def _ppgs_available() -> bool:
    try:
        import ppgs  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _concat_aligned(parts: list[FeatureSeries], *, hop_s: float) -> FeatureSeries:
    t0 = min(float(part.times_s[0]) for part in parts)
    t1 = max(float(part.times_s[-1]) for part in parts)
    n = int(np.floor((t1 - t0) / hop_s)) + 1
    grid = t0 + np.arange(n, dtype=np.float64) * hop_s
    resampled = [
        resample_feature_series(part, hop_s, method="linear", time_grid_s=grid)
        for part in parts
    ]
    from natural_features.fmri.design import concat_feature_series

    return concat_feature_series(resampled, standardize=False, add_intercept=False)


def extract_speech_ladder(
    audio: AudioStimulus | str | Path,
    *,
    hop_s: float = 0.01,
    posterior_backend: str = "auto",
    include_sparc: bool = True,
    include_residuals: bool = True,
    include_cues: bool = True,
    phones: EventSeries | None = None,
    formant_backend: str = "lpc_autocorr",
    analysis_hop_s: float = 0.02,
    execution_mode: str | None = None,
    strict_dependency: bool | None = None,
    ctc_model: str = "bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
    ctc_local_files_only: bool = True,
    ppgs_representation: str = "mel",
    ppgs_local_files_only: bool = True,
    sparc_model: str = "feature_extraction",
    sparc_local_files_only: bool = True,
) -> FeatureBundle:
    """Extract graded speech bands and optional residualized successors.

    Default posterior backend is PPGs when installed, otherwise CTC. SPARC is
    omitted when the extra is missing unless execution is strict.
    """

    mode, strict = resolve_execution_mode(
        execution_mode=execution_mode, strict_dependency=strict_dependency
    )
    stim = audio if isinstance(audio, AudioStimulus) else AudioStimulus.from_wav(audio)
    if posterior_backend == "auto":
        posterior_backend = "ppgs" if _ppgs_available() else "ctc"

    a1 = audio_envelope(stim, hop_s=hop_s)
    a2 = spectrotemporal_modulation(stim, hop_s=hop_s)
    formants = audio_formants(
        stim,
        hop_s=hop_s,
        backend=formant_backend,
        execution_mode=mode,
        strict_dependency=strict,
    )
    hnr = audio_harmonicity(stim, hop_s=hop_s)
    a3 = _concat_aligned([formants, hnr], hop_s=hop_s)

    if posterior_backend == "acoustic":
        post = acoustic_phone_posteriors(stim, hop_s=max(hop_s, 0.02))
    elif posterior_backend == "ctc":
        post = ctc_phone_posteriors(
            stim,
            model=ctc_model,
            local_files_only=ctc_local_files_only,
            execution_mode=mode,
            strict_dependency=strict,
        )
    elif posterior_backend == "ppgs":
        post = ppg_phone_posteriors(
            stim,
            representation=ppgs_representation,
            local_files_only=ppgs_local_files_only,
            execution_mode=mode,
        )
    else:
        raise ValueError(
            "posterior_backend must be one of {'auto', 'ctc', 'acoustic', 'ppgs'}"
        )
    p_features = distinctive_from_posteriors(post)

    features: dict[str, FeatureSeries | EventSeries] = {
        "a1": inherit_temporal_contract(a1, [stim]),
        "a2": inherit_temporal_contract(a2, [stim]),
        "a3": inherit_temporal_contract(a3, [stim]),
        "p_posteriors": inherit_temporal_contract(post, [stim]),
        "p_features": inherit_temporal_contract(p_features, [stim]),
    }
    if phones is not None:
        features["phones"] = phones
        if include_cues:
            features["a3_cues"] = inherit_temporal_contract(
                phonetic_cues(stim, phones, hop_s=hop_s), [stim]
            )
        gestures = articulatory_gestures(phones, hop_s=hop_s)
        features["g_gestures"] = inherit_temporal_contract(gestures, [stim])

    g_source: FeatureSeries | None = None
    if include_sparc:
        try:
            g_ema = sparc_articulatory(
                stim,
                model=sparc_model,
                local_files_only=sparc_local_files_only,
                execution_mode=mode,
                strict_dependency=strict,
            )
            features["g_ema"] = inherit_temporal_contract(g_ema, [stim])
            g_source = g_ema
        except (BackendDependencyError, BackendLoadError):
            if strict:
                raise
    if g_source is None and "g_gestures" in features:
        g_source = features["g_gestures"]  # type: ignore[assignment]
    if g_source is not None:
        features["m_dynamics"] = inherit_temporal_contract(
            articulatory_dynamics(g_source), [stim]
        )

    if include_residuals:
        residual_pairs: list[tuple[str, FeatureSeries, list[FeatureSeries]]] = [
            ("a2|a1", a2, [a1]),
            ("a3|a1+a2", a3, [a1, a2]),
            ("p|a", p_features, [a1, a2, a3]),
        ]
        if "g_ema" in features:
            residual_pairs.append(
                ("g_ema|a+p", features["g_ema"], [a1, a2, a3, p_features])  # type: ignore[list-item]
            )
        if "m_dynamics" in features and g_source is not None:
            residual_pairs.append(("m|g", features["m_dynamics"], [g_source]))  # type: ignore[list-item]
        for key, target, preds in residual_pairs:
            features[key] = residualize_feature_series(
                target, preds, hop_s=analysis_hop_s, method="linear"
            )

    return FeatureBundle(
        features=features,
        metadata={
            "workflow": "extract_speech_ladder",
            "posterior_backend": posterior_backend,
            "include_sparc": "g_ema" in features,
            "include_residuals": include_residuals,
            "execution_mode": mode,
        },
    )
