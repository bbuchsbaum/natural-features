from __future__ import annotations

import natural_features as nf
from natural_features import public_api

EXPECTED_STABLE_EXPORTS = [
    "FeatureSeries",
    "EventSeries",
    "TrackSeries",
    "FrameTimeline",
    "Timeline",
    "FeatureAlignment",
    "ExtractFeaturesResult",
    "AlignedFeatureSet",
    "VideoTextResult",
    "RunGrid",
    "ExperimentGrid",
    "build_experiment_grid",
    "query_feature_window",
    "query_feature_window_tr",
    "query_feature_zoo_window_tr",
    "extract_acoustic_phonetics",
    "available_features",
    "feature_catalog",
    "plan_features",
    "extract_features",
    "extract_audio_files",
    "extract_audio_dir",
    "extract_multiscale_language",
    "extract_video_text",
]


def test_root_api_reexports_public_api_contract() -> None:
    assert nf.API_COMPAT_VERSION == public_api.API_COMPAT_VERSION
    assert nf.STABLE_EXPORTS == public_api.STABLE_EXPORTS
    assert nf.EXPERIMENTAL_NAMESPACES == public_api.EXPERIMENTAL_NAMESPACES
    assert nf.__all__ == public_api.__all__


def test_stable_exports_are_exact_and_available() -> None:
    assert nf.STABLE_EXPORTS == EXPECTED_STABLE_EXPORTS
    for symbol in EXPECTED_STABLE_EXPORTS:
        assert hasattr(nf, symbol)


def test_experimental_namespaces_are_declared() -> None:
    assert nf.EXPERIMENTAL_NAMESPACES
    for ns in nf.EXPERIMENTAL_NAMESPACES:
        assert ns.startswith("natural_features.")


def test_api_compat_version_is_int_and_positive() -> None:
    assert isinstance(nf.API_COMPAT_VERSION, int)
    assert nf.API_COMPAT_VERSION >= 1
