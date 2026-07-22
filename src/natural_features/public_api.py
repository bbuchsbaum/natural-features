"""Stable public API contract for natural_features.

Only symbols exported from this module are covered by the compatibility policy.
"""

# ruff: noqa: F401

from __future__ import annotations

from .core.feature_types import EventSeries, FeatureSeries, TrackSeries
from .core.feature_bundle import FeatureBundle, TemporalPayload, temporal_object_in_clock
from .core.frame_timeline import FrameTimeline
from .core.timeline import FeatureAlignment, Timeline
from .core.timebase import ClockMap, ClockRef, SupportSpec, TemporalContext, TimebaseSpec
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
from .workflows.extract_features import (
    AlignedFeatureSet,
    ExtractFeaturesResult,
    available_features,
    extract_features,
    feature_catalog,
    plan_features,
)
from .workflows.multiscale_language import extract_multiscale_language
from .workflows.video_text import VideoTextResult, extract_video_text

# Increment only when stable public API contracts change in a breaking way.
API_COMPAT_VERSION = 2

STABLE_EXPORTS = [
    "FeatureSeries",
    "EventSeries",
    "TrackSeries",
    "ClockRef",
    "ClockMap",
    "SupportSpec",
    "TemporalContext",
    "TimebaseSpec",
    "FeatureBundle",
    "TemporalPayload",
    "temporal_object_in_clock",
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
