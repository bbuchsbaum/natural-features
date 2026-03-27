"""natural_features package.

The root package re-exports only the versioned public API contract.
"""

from .public_api import (  # noqa: F401
    API_COMPAT_VERSION,
    EXPERIMENTAL_NAMESPACES,
    STABLE_EXPORTS,
    EventSeries,
    ExperimentGrid,
    FeatureSeries,
    RunGrid,
    TrackSeries,
    build_experiment_grid,
    extract_acoustic_phonetics,
    extract_audio_dir,
    extract_audio_files,
    extract_multiscale_language,
    query_feature_window,
    query_feature_window_tr,
    query_feature_zoo_window_tr,
)

__all__ = [
    "API_COMPAT_VERSION",
    "EXPERIMENTAL_NAMESPACES",
    "STABLE_EXPORTS",
    "FeatureSeries",
    "EventSeries",
    "TrackSeries",
    "RunGrid",
    "ExperimentGrid",
    "build_experiment_grid",
    "query_feature_window",
    "query_feature_window_tr",
    "query_feature_zoo_window_tr",
    "extract_acoustic_phonetics",
    "extract_audio_files",
    "extract_audio_dir",
    "extract_multiscale_language",
]
