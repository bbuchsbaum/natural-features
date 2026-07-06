"""Table interchange helpers for temporal feature objects."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata

OBJECT_CANONICAL_COLUMNS = [
    "source_id",
    "time_s",
    "onset_s",
    "offset_s",
    "duration_s",
    "object_id",
    "track_id",
    "object_type",
    "label",
    "confidence",
    "x",
    "y",
    "width",
    "height",
    "area",
    "coordinate_space",
]


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required for table interchange: pip install natural-features[storage]") from exc
    return pd


def _row_column(value: Any, n: int, name: str, *, default: Any = np.nan, dtype: Any | None = None) -> np.ndarray:
    if value is None:
        value = default
    arr = np.asarray(value if not isinstance(value, list) else value, dtype=dtype)
    if n == 0:
        return np.asarray([], dtype=arr.dtype if arr.ndim else dtype)
    if arr.ndim == 0 or arr.shape == ():
        arr = np.repeat(arr, n)
    if len(arr) != n:
        raise ValueError(f"{name} must have length 1 or {n}")
    return arr


def _metadata_source_id(metadata: dict[str, Any]) -> str | None:
    for key in ("source_id", "source", "source_file", "filename"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _object_id_prefix(object_type: Any) -> np.ndarray:
    raw = np.asarray(object_type, dtype=object)
    out = []
    for item in raw:
        label = re.sub(r"[^A-Za-z0-9]+", "_", str(item).lower()).strip("_")
        out.append(label or "object")
    return np.asarray(out, dtype=object)


def _make_object_ids(n: int, object_type: Any) -> np.ndarray:
    if n == 0:
        return np.asarray([], dtype=object)
    prefixes = _object_id_prefix(object_type)
    counts: dict[str, int] = {}
    ids = []
    for prefix in prefixes:
        counts[prefix] = counts.get(prefix, 0) + 1
        ids.append(f"{prefix}_{counts[prefix]:04d}")
    return np.asarray(ids, dtype=object)


def _provenance_columns(metadata: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    keys = [
        "extractor_id",
        "extractor_name",
        "params_hash",
        "code_version",
        "model_revision",
        "schema",
        "source",
        "execution_mode",
        "fallback_used",
        "fallback_reason",
        "backend",
    ]
    return {key: _row_column(metadata.get(key), n, key, default=np.nan, dtype=object) for key in keys if key in metadata}


def ensure_object_ids(
    events: EventSeries,
    *,
    object_type: str | list[str] | np.ndarray | None = None,
    source_id: str | list[str] | np.ndarray | None = None,
) -> EventSeries:
    if not isinstance(events, EventSeries):
        raise TypeError("events must be an EventSeries")
    n = len(events)
    extra = dict(events.extra)
    type_col = _row_column(
        extra.get("object_type", object_type if object_type is not None else events.metadata.get("object_type", "object")),
        n,
        "object_type",
        default="object",
        dtype=object,
    )
    source_col = _row_column(
        extra.get("source_id", source_id if source_id is not None else _metadata_source_id(events.metadata)),
        n,
        "source_id",
        default=None,
        dtype=object,
    )
    ids = _row_column(extra.get("object_id"), n, "object_id", default=None, dtype=object)
    missing = np.asarray([(x is None) or (isinstance(x, float) and np.isnan(x)) or str(x) == "" for x in ids])
    if np.any(missing):
        generated = _make_object_ids(n, type_col)
        ids[missing] = generated[missing]
    extra.update({"object_id": ids, "object_type": type_col, "source_id": source_col})
    return EventSeries(
        onset_s=events.onset_s,
        offset_s=events.offset_s,
        label=events.label,
        confidence=events.confidence,
        extra=extra,
        metadata=events.metadata,
        schema=events.schema,
        timebase=events.timebase,
    )


def object_events(
    onset_s: np.ndarray | list[float],
    offset_s: np.ndarray | list[float] | None = None,
    *,
    object_type: str | list[str] | np.ndarray = "object",
    object_id: str | list[str] | np.ndarray | None = None,
    track_id: str | list[str] | np.ndarray | None = None,
    source_id: str | list[str] | np.ndarray | None = None,
    time_s: float | list[float] | np.ndarray | None = None,
    label: str | list[str] | np.ndarray | None = None,
    confidence: float | list[float] | np.ndarray | None = None,
    x: float | list[float] | np.ndarray | None = None,
    y: float | list[float] | np.ndarray | None = None,
    width: float | list[float] | np.ndarray | None = None,
    height: float | list[float] | np.ndarray | None = None,
    area: float | list[float] | np.ndarray | None = None,
    coordinate_space: str | list[str] | np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    schema: str = "EventSeries/v1",
    timebase: TimebaseSpec | None = None,
) -> EventSeries:
    onset = np.asarray(onset_s, dtype=np.float64)
    offset = np.asarray(offset_s if offset_s is not None else onset_s, dtype=np.float64)
    n = len(onset)
    if len(offset) != n:
        raise ValueError(f"offset_s must have length {n}")
    extra_out = dict(extra or {})
    metadata = dict(metadata or extractor_metadata("object.events", params={}))
    geom_w = _row_column(width, n, "width", default=np.nan, dtype=float)
    geom_h = _row_column(height, n, "height", default=np.nan, dtype=float)
    if area is None and (np.any(np.isfinite(geom_w)) or np.any(np.isfinite(geom_h))):
        area = geom_w * geom_h
    extra_out.update(
        {
            "source_id": _row_column(source_id if source_id is not None else _metadata_source_id(metadata), n, "source_id", default=None, dtype=object),
            "time_s": _row_column(time_s if time_s is not None else onset, n, "time_s", default=np.nan, dtype=float),
            "object_id": _row_column(object_id, n, "object_id", default=None, dtype=object),
            "track_id": _row_column(track_id, n, "track_id", default=None, dtype=object),
            "object_type": _row_column(object_type, n, "object_type", default="object", dtype=object),
            "x": _row_column(x, n, "x", default=np.nan, dtype=float),
            "y": _row_column(y, n, "y", default=np.nan, dtype=float),
            "width": geom_w,
            "height": geom_h,
            "area": _row_column(area, n, "area", default=np.nan, dtype=float),
            "coordinate_space": _row_column(coordinate_space, n, "coordinate_space", default=None, dtype=object),
        }
    )
    events = EventSeries(
        onset_s=onset,
        offset_s=offset,
        label=None if label is None else _row_column(label, n, "label", dtype=object),
        confidence=None if confidence is None else _row_column(confidence, n, "confidence", dtype=float),
        extra=extra_out,
        metadata=metadata,
        schema=schema,
        timebase=timebase or TimebaseSpec(kind="events"),
    )
    return ensure_object_ids(events)


def as_feature_table(feature: FeatureSeries, *, include_metadata: bool = True) -> Any:
    pd = _require_pandas()
    values = feature.values.reshape(feature.values.shape[0], -1)
    names = feature.coords.get("feature", [f"f{i}" for i in range(values.shape[1])])
    if len(names) != values.shape[1]:
        names = [f"f{i}" for i in range(values.shape[1])]
    rows = {
        "time_s": np.repeat(feature.times_s, values.shape[1]),
        "feature": np.tile(np.asarray(names, dtype=object), len(feature.times_s)),
        "value": values.reshape(-1),
    }
    table = pd.DataFrame(rows)
    if include_metadata:
        for key, value in _provenance_columns({**feature.metadata, "schema": feature.schema}, len(table)).items():
            table[key] = value
    return table


def as_event_table(events: EventSeries, *, include_metadata: bool = True) -> Any:
    pd = _require_pandas()
    n = len(events)
    table = pd.DataFrame(
        {
            "onset_s": events.onset_s,
            "offset_s": events.offset_s,
            "duration_s": events.offset_s - events.onset_s,
            "label": _row_column(events.label, n, "label", default=None, dtype=object),
            "confidence": _row_column(events.confidence, n, "confidence", default=np.nan, dtype=float),
        }
    )
    for key, value in events.extra.items():
        arr = np.asarray(value)
        if len(arr) == n:
            table[key] = arr
    if include_metadata:
        for key, value in _provenance_columns({**events.metadata, "schema": events.schema}, n).items():
            table[key] = value
    return table


def as_track_table(tracks: TrackSeries, *, include_metadata: bool = True) -> Any:
    pd = _require_pandas()
    values = tracks.values.reshape(tracks.values.shape[0], tracks.values.shape[1], -1)
    feature_names = tracks.coords.get("feature", [f"f{i}" for i in range(values.shape[2])])
    rows = []
    for ti, time_s in enumerate(tracks.times_s):
        for ki, track_id in enumerate(tracks.track_id):
            for fi, feature_name in enumerate(feature_names):
                rows.append({"time_s": time_s, "track_id": track_id, "feature": feature_name, "value": values[ti, ki, fi]})
    table = pd.DataFrame(rows)
    if include_metadata:
        for key, value in _provenance_columns({**tracks.metadata, "schema": tracks.schema}, len(table)).items():
            table[key] = value
    return table


def as_object_table(obj: EventSeries | TrackSeries, *, include_extra: bool = True, include_metadata: bool = True) -> Any:
    pd = _require_pandas()
    if isinstance(obj, EventSeries):
        events = ensure_object_ids(obj)
        n = len(events)
        data = {
            "source_id": _row_column(events.extra.get("source_id"), n, "source_id", default=None, dtype=object),
            "time_s": _row_column(events.extra.get("time_s", events.onset_s), n, "time_s", default=np.nan, dtype=float),
            "onset_s": events.onset_s,
            "offset_s": events.offset_s,
            "duration_s": events.offset_s - events.onset_s,
            "object_id": _row_column(events.extra.get("object_id"), n, "object_id", default=None, dtype=object),
            "track_id": _row_column(events.extra.get("track_id"), n, "track_id", default=None, dtype=object),
            "object_type": _row_column(events.extra.get("object_type"), n, "object_type", default="object", dtype=object),
            "label": _row_column(events.label, n, "label", default=None, dtype=object),
            "confidence": _row_column(events.confidence, n, "confidence", default=np.nan, dtype=float),
            "x": _row_column(events.extra.get("x"), n, "x", default=np.nan, dtype=float),
            "y": _row_column(events.extra.get("y"), n, "y", default=np.nan, dtype=float),
            "width": _row_column(events.extra.get("width"), n, "width", default=np.nan, dtype=float),
            "height": _row_column(events.extra.get("height"), n, "height", default=np.nan, dtype=float),
            "area": _row_column(events.extra.get("area"), n, "area", default=np.nan, dtype=float),
            "coordinate_space": _row_column(events.extra.get("coordinate_space"), n, "coordinate_space", default=None, dtype=object),
        }
        table = pd.DataFrame(data)
        if include_extra:
            for key, value in events.extra.items():
                if key in OBJECT_CANONICAL_COLUMNS:
                    continue
                arr = np.asarray(value)
                if len(arr) == n:
                    table[key] = arr
        if include_metadata:
            for key, value in _provenance_columns({**events.metadata, "schema": events.schema}, n).items():
                table[key] = value
        return table
    if isinstance(obj, TrackSeries):
        rows = []
        source = _metadata_source_id(obj.metadata)
        for time_s in obj.times_s:
            for track_id in obj.track_id:
                rows.append(
                    {
                        "source_id": source,
                        "time_s": time_s,
                        "onset_s": np.nan,
                        "offset_s": np.nan,
                        "duration_s": np.nan,
                        "object_id": track_id,
                        "track_id": track_id,
                        "object_type": "track",
                        "label": track_id,
                        "confidence": np.nan,
                        "x": np.nan,
                        "y": np.nan,
                        "width": np.nan,
                        "height": np.nan,
                        "area": np.nan,
                        "coordinate_space": None,
                    }
                )
        table = pd.DataFrame(rows, columns=OBJECT_CANONICAL_COLUMNS)
        if include_metadata:
            for key, value in _provenance_columns({**obj.metadata, "schema": obj.schema}, len(table)).items():
                table[key] = value
        return table
    raise TypeError("obj must be an EventSeries or TrackSeries")


def as_temporal_table(obj: FeatureSeries | EventSeries | TrackSeries, *, include_metadata: bool = True) -> Any:
    if isinstance(obj, FeatureSeries):
        return as_feature_table(obj, include_metadata=include_metadata)
    if isinstance(obj, EventSeries):
        return as_event_table(obj, include_metadata=include_metadata)
    if isinstance(obj, TrackSeries):
        return as_track_table(obj, include_metadata=include_metadata)
    raise TypeError("Unsupported temporal object")


def _flatten_outputs(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, (FeatureSeries, EventSeries, TrackSeries)):
        return [(prefix or "default", value)]
    if isinstance(value, dict):
        out: list[tuple[str, Any]] = []
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_flatten_outputs(item, name))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for i, item in enumerate(value):
            name = f"{prefix}.{i}" if prefix else str(i)
            out.extend(_flatten_outputs(item, name))
        return out
    return []


def merge_feature_tables(
    value: Any,
    *,
    format: str = "long",
    include_metadata: bool = True,
    include_objects: bool = True,
) -> Any:
    pd = _require_pandas()
    items = _flatten_outputs(value)
    if not items:
        return pd.DataFrame()
    if format not in {"long", "wide"}:
        raise ValueError("format must be 'long' or 'wide'")
    if format == "long":
        tables = []
        for name, obj in items:
            if include_objects and isinstance(obj, EventSeries):
                table = as_object_table(obj, include_metadata=include_metadata)
                output_type = "objects"
            elif isinstance(obj, FeatureSeries):
                table = as_feature_table(obj, include_metadata=include_metadata)
                output_type = "features"
            elif isinstance(obj, EventSeries):
                table = as_event_table(obj, include_metadata=include_metadata)
                output_type = "events"
            elif isinstance(obj, TrackSeries):
                table = as_track_table(obj, include_metadata=include_metadata)
                output_type = "tracks"
            else:
                continue
            if "feature_id" not in table:
                table["feature_id"] = name
            table["output_name"] = name
            table["output_type"] = output_type
            tables.append(table)
        return pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()

    bad = [name for name, obj in items if not isinstance(obj, FeatureSeries)]
    if bad:
        raise ValueError(f"Wide format only supports FeatureSeries outputs. Incompatible outputs: {', '.join(bad)}")
    wide = None
    for name, obj in items:
        assert isinstance(obj, FeatureSeries)
        vals = obj.values.reshape(obj.values.shape[0], -1)
        feature_names = obj.coords.get("feature", [f"f{i}" for i in range(vals.shape[1])])
        columns = [f"{name}__{feature}" for feature in feature_names]
        table = pd.DataFrame(vals, columns=columns)
        table.insert(0, "time_s", obj.times_s)
        wide = table if wide is None else pd.merge(wide, table, on="time_s", how="outer")
    return wide if wide is not None else pd.DataFrame()
