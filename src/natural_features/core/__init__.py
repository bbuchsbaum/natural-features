"""Core contracts and utilities."""

from .feature_types import EventSeries, FeatureSeries, TrackSeries
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

__all__ = [
    "AudioStimulus",
    "EventSeries",
    "as_mermaid",
    "as_event_table",
    "as_feature_table",
    "as_object_table",
    "as_temporal_table",
    "as_track_table",
    "ensure_object_ids",
    "execute_recipe",
    "FeatureSeries",
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
    "TrackSeries",
    "validate_recipe",
    "VideoStimulus",
]
