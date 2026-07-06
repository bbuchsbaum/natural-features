"""Stable public API contract for natural_features.

Only symbols exported from this module are covered by the compatibility policy.
"""

# ruff: noqa: F401

from __future__ import annotations

from .core.feature_types import EventSeries, FeatureSeries, TrackSeries
from .fmri.query import (
    ExperimentGrid,
    RunGrid,
    build_experiment_grid,
    query_feature_window,
    query_feature_window_tr,
    query_feature_zoo_window_tr,
)
from .workflows.acoustic_phonetics import extract_acoustic_phonetics
from .workflows.audio_batch import extract_audio_dir, extract_audio_files
from .workflows.multiscale_language import extract_multiscale_language

# Increment only when stable public API contracts change in a breaking way.
API_COMPAT_VERSION = 1

STABLE_EXPORTS = [
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

EXPERIMENTAL_NAMESPACES = [
    "natural_features.features",
    "natural_features.flow",
    "natural_features.core.recipe",
    "natural_features.core.registry",
    "natural_features.workflows.extract_features",
]

__all__ = [
    "API_COMPAT_VERSION",
    "EXPERIMENTAL_NAMESPACES",
    "STABLE_EXPORTS",
    *STABLE_EXPORTS,
]
