"""Validation utilities for canonical feature objects."""

from __future__ import annotations

from typing import Any

from .feature_types import EventSeries, FeatureSeries, TrackSeries

FeatureObject = FeatureSeries | EventSeries | TrackSeries


def validate_feature_object(obj: Any) -> FeatureObject:
    if isinstance(obj, (FeatureSeries, EventSeries, TrackSeries)):
        return obj
    raise TypeError(
        "Extractor outputs must be FeatureSeries, EventSeries, or TrackSeries; "
        f"got {type(obj)!r}"
    )

