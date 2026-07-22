"""Core contracts and utilities."""

from .feature_types import EventSeries, FeatureSeries, TrackSeries
from .feature_bundle import FeatureBundle, TemporalPayload, temporal_object_in_clock
from .frame_timeline import FramePolicy, FrameTimeline
from .interchange import (
    as_event_table,
    as_feature_table,
    as_object_table,
    as_temporal_table,
    as_track_table,
    ensure_object_ids,
    merge_feature_tables,
    object_events,
)
from .recipe import RecipeDag, as_mermaid, execute_recipe, load_recipe, plan_dag, validate_recipe
from .registry import Registry
from .stimulus import (
    AudioStimulus,
    ImageStimulus,
    MultiModalStimulus,
    TextStimulus,
    VideoStimulus,
    image_from_array,
    image_from_file,
)
from .timeline import AlignmentPolicy, FeatureAlignment, Timeline, align_feature_to_timeline
from .timebase import ClockMap, ClockRef, SupportSpec, TemporalContext, TimebaseSpec

__all__ = [
    "AudioStimulus",
    "AlignmentPolicy",
    "ClockMap",
    "ClockRef",
    "EventSeries",
    "FeatureAlignment",
    "FeatureBundle",
    "as_mermaid",
    "as_event_table",
    "as_feature_table",
    "as_object_table",
    "as_temporal_table",
    "as_track_table",
    "ensure_object_ids",
    "execute_recipe",
    "FeatureSeries",
    "FramePolicy",
    "FrameTimeline",
    "ImageStimulus",
    "image_from_array",
    "image_from_file",
    "MultiModalStimulus",
    "merge_feature_tables",
    "object_events",
    "load_recipe",
    "plan_dag",
    "RecipeDag",
    "Registry",
    "TextStimulus",
    "SupportSpec",
    "TemporalContext",
    "TemporalPayload",
    "TimebaseSpec",
    "TrackSeries",
    "Timeline",
    "align_feature_to_timeline",
    "temporal_object_in_clock",
    "validate_recipe",
    "VideoStimulus",
]
