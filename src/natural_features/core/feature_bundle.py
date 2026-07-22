"""Heterogeneous native-grid feature bundles and downstream payloads."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, TypeAlias

import numpy as np

from .feature_types import EventSeries, FeatureSeries, TrackSeries
from .timebase import ClockRef, STIMULUS_CLOCK, TemporalContext, TimebaseSpec

TemporalFeature: TypeAlias = FeatureSeries | EventSeries | TrackSeries


def inherit_temporal_contract(value: Any, sources: Iterable[Any]) -> Any:
    """Propagate source clocks and mappings through legacy extractors.

    Extractors that predate explicit clocks generally return objects on the
    legacy ``stimulus`` clock. When all temporal inputs use one other clock,
    the numeric coordinates are already expressed on that input clock, so this
    helper stamps the correct reference and merges the source contexts. It
    deliberately does not transform coordinates or resample values.
    """

    temporal_sources = [
        source
        for source in sources
        if hasattr(source, "clock") or hasattr(source, "temporal_context")
    ]
    contexts = [
        source.temporal_context
        for source in temporal_sources
        if isinstance(getattr(source, "temporal_context", None), TemporalContext)
    ]
    context = TemporalContext()
    if contexts:
        context = contexts[0].merged(*contexts[1:])
    clocks = {
        ClockRef(source.clock)
        for source in temporal_sources
        if getattr(source, "clock", None) is not None
    }
    source_clock = next(iter(clocks)) if len(clocks) == 1 else None

    if isinstance(value, dict):
        return {
            key: inherit_temporal_contract(item, temporal_sources)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [inherit_temporal_contract(item, temporal_sources) for item in value]
    if isinstance(value, tuple):
        return tuple(
            inherit_temporal_contract(item, temporal_sources) for item in value
        )
    if not isinstance(value, (FeatureSeries, EventSeries, TrackSeries)):
        return value

    combined = value.temporal_context.merged(context)
    timebase = value.timebase
    if (
        source_clock is not None
        and value.clock == STIMULUS_CLOCK
        and source_clock != STIMULUS_CLOCK
    ):
        timebase = timebase.with_reference(source_clock)
    return replace(value, timebase=timebase, temporal_context=combined)


def _combined_context(obj: TemporalFeature, context: TemporalContext | None) -> TemporalContext:
    if context is None:
        return obj.temporal_context
    return obj.temporal_context.merged(context)


def _transformed_timebase(timebase: TimebaseSpec, target: ClockRef, scale: float) -> TimebaseSpec:
    return TimebaseSpec(
        kind=timebase.kind,
        reference=target,
        sampling_rate_hz=(
            None
            if timebase.sampling_rate_hz is None
            else timebase.sampling_rate_hz / scale
        ),
        hop_s=None if timebase.hop_s is None else timebase.hop_s * scale,
        window_s=None if timebase.window_s is None else timebase.window_s * scale,
        stride_s=None if timebase.stride_s is None else timebase.stride_s * scale,
        alignment=timebase.alignment,
        support=timebase.support.scaled(scale),
    )


def temporal_object_in_clock(
    obj: TemporalFeature,
    target: ClockRef | str,
    *,
    context: TemporalContext | None = None,
) -> TemporalFeature:
    """Return a coordinate-transformed copy without resampling its values."""

    target_ref = ClockRef(target)
    combined = _combined_context(obj, context)
    mapping = combined.resolve(obj.clock, target_ref)
    timebase = _transformed_timebase(obj.timebase, target_ref, mapping.scale)
    if isinstance(obj, FeatureSeries):
        bounds = None
        if obj.time_bounds_s is not None:
            bounds = np.column_stack(
                [mapping.apply(obj.time_bounds_s[:, 0]), mapping.apply(obj.time_bounds_s[:, 1])]
            )
        return FeatureSeries(
            values=obj.values,
            times_s=mapping.apply(obj.times_s),
            dims=obj.dims,
            coords=obj.coords,
            metadata=dict(obj.metadata),
            schema=obj.schema,
            timebase=timebase,
            time_bounds_s=bounds,
            temporal_context=combined,
        )
    if isinstance(obj, EventSeries):
        return EventSeries(
            onset_s=mapping.apply(obj.onset_s),
            offset_s=mapping.apply(obj.offset_s),
            label=obj.label,
            confidence=obj.confidence,
            extra=dict(obj.extra),
            metadata=dict(obj.metadata),
            schema=obj.schema,
            timebase=timebase,
            temporal_context=combined,
        )
    bounds = None
    if obj.time_bounds_s is not None:
        bounds = np.column_stack(
            [mapping.apply(obj.time_bounds_s[:, 0]), mapping.apply(obj.time_bounds_s[:, 1])]
        )
    return TrackSeries(
        times_s=mapping.apply(obj.times_s),
        track_id=obj.track_id,
        values=obj.values,
        dims=obj.dims,
        coords=obj.coords,
        metadata=dict(obj.metadata),
        schema=obj.schema,
        timebase=timebase,
        time_bounds_s=bounds,
        temporal_context=combined,
    )


@dataclass(frozen=True)
class TemporalPayload:
    """Dependency-free native temporal payload for a modeling consumer."""

    name: str
    object_type: str
    clock: ClockRef
    timebase: dict[str, Any]
    temporal_context: dict[str, Any]
    metadata: dict[str, Any]
    schema: str
    values: np.ndarray | None = None
    times_s: np.ndarray | None = None
    time_bounds_s: np.ndarray | None = None
    onset_s: np.ndarray | None = None
    offset_s: np.ndarray | None = None
    label: np.ndarray | None = None
    confidence: np.ndarray | None = None
    track_id: np.ndarray | None = None
    dims: tuple[str, ...] | None = None
    coords: dict[str, list[Any]] = field(default_factory=dict)
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a protocol payload made only of mappings, arrays, and scalars."""

        return {
            "name": self.name,
            "object_type": self.object_type,
            "clock": str(self.clock),
            "timebase": dict(self.timebase),
            "temporal_context": dict(self.temporal_context),
            "metadata": dict(self.metadata),
            "schema": self.schema,
            "values": self.values,
            "times_s": self.times_s,
            "time_bounds_s": self.time_bounds_s,
            "onset_s": self.onset_s,
            "offset_s": self.offset_s,
            "label": self.label,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "dims": self.dims,
            "coords": dict(self.coords),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class FeatureBundle:
    """A collection of native-grid features sharing temporal mappings."""

    features: dict[str, TemporalFeature]
    temporal_context: TemporalContext = field(default_factory=TemporalContext)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        context = self.temporal_context
        if not isinstance(context, TemporalContext):
            context = TemporalContext.from_dict(context)
        features = dict(self.features)
        for name, obj in features.items():
            if not isinstance(obj, (FeatureSeries, EventSeries, TrackSeries)):
                raise TypeError(f"feature {name!r} is not a temporal feature object")
            obj.temporal_context.merged(context)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "temporal_context", context)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def in_clock(self, name: str, target: ClockRef | str) -> TemporalFeature:
        if name not in self.features:
            raise KeyError(f"unknown feature {name!r}")
        return temporal_object_in_clock(
            self.features[name],
            target,
            context=self.temporal_context,
        )

    def temporal_payload(
        self,
        name: str,
        *,
        target_clock: ClockRef | str | None = None,
    ) -> TemporalPayload:
        if name not in self.features:
            raise KeyError(f"unknown feature {name!r}")
        obj = self.features[name]
        if target_clock is not None:
            obj = self.in_clock(name, target_clock)
        context = obj.temporal_context.merged(self.temporal_context)
        common = {
            "name": name,
            "clock": obj.clock,
            "timebase": obj.timebase.to_dict(),
            "temporal_context": context.to_dict(),
            "metadata": {**self.metadata, **obj.metadata},
            "schema": obj.schema,
        }
        if isinstance(obj, FeatureSeries):
            return TemporalPayload(
                object_type="features",
                values=obj.values,
                times_s=obj.times_s,
                time_bounds_s=obj.temporal_bounds_s,
                dims=obj.dims,
                coords=dict(obj.coords),
                **common,
            )
        if isinstance(obj, EventSeries):
            return TemporalPayload(
                object_type="events",
                onset_s=obj.onset_s,
                offset_s=obj.offset_s,
                time_bounds_s=obj.temporal_bounds_s,
                label=obj.label,
                confidence=obj.confidence,
                extra=dict(obj.extra),
                **common,
            )
        return TemporalPayload(
            object_type="tracks",
            values=obj.values,
            times_s=obj.times_s,
            time_bounds_s=obj.temporal_bounds_s,
            track_id=obj.track_id,
            dims=obj.dims,
            coords=dict(obj.coords),
            **common,
        )


__all__ = [
    "FeatureBundle",
    "TemporalFeature",
    "TemporalPayload",
    "inherit_temporal_contract",
    "temporal_object_in_clock",
]
