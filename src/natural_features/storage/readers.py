"""Artifact readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.timebase import TemporalContext, TimebaseSpec


def _require_zarr() -> Any:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("zarr is required for zarr reads: pip install natural-features[storage]") from exc
    return zarr


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet reads: pip install natural-features[storage]"
        ) from exc
    return pq


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_storage_array(value: Any, *, encoded: bool) -> np.ndarray:
    array = np.asarray(value)
    if not encoded:
        return array
    decoded = [json.loads(str(item)) for item in array.reshape(-1)]
    return np.asarray(decoded, dtype=object).reshape(array.shape)


def read_feature_series(path: str | Path) -> FeatureSeries:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        meta = _read_meta(p.with_name("feature_series.meta.json"))
        encoded = set(meta.get("json_encoded_arrays", []))
        return FeatureSeries(
            values=_decode_storage_array(data["values"], encoded="values" in encoded),
            times_s=data["times_s"],
            dims=tuple(meta.get("dims", ("time", "feature"))),
            coords=meta.get("coords", {}),
            metadata=meta.get("metadata", {}),
            schema=meta.get("schema", "FeatureSeries/v1"),
            timebase=TimebaseSpec.from_dict(meta.get("timebase"), default_kind="frames"),
            time_bounds_s=data["time_bounds_s"] if "time_bounds_s" in data.files else None,
            temporal_context=TemporalContext.from_dict(meta.get("temporal_context")),
        )
    if p.suffix == ".zarr":
        zarr = _require_zarr()
        root = zarr.open_group(str(p), mode="r")
        encoded = set(root.attrs.get("json_encoded_arrays", []))
        return FeatureSeries(
            values=_decode_storage_array(
                root["values"],
                encoded="values" in encoded,
            ),
            times_s=np.asarray(root["times_s"]),
            dims=tuple(root.attrs.get("dims", ("time", "feature"))),
            coords=dict(root.attrs.get("coords", {})),
            metadata=dict(root.attrs.get("metadata", {})),
            schema=str(root.attrs.get("schema", "FeatureSeries/v1")),
            timebase=TimebaseSpec.from_dict(dict(root.attrs.get("timebase", {})), default_kind="frames"),
            time_bounds_s=np.asarray(root["time_bounds_s"]) if "time_bounds_s" in root else None,
            temporal_context=TemporalContext.from_dict(dict(root.attrs.get("temporal_context", {}))),
        )
    raise ValueError(f"Unsupported feature path: {path}")


def read_event_series(path: str | Path) -> EventSeries:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        meta = _read_meta(p.with_name("event_series.meta.json"))
        encoded = set(meta.get("json_encoded_arrays", []))
        keys = set(data.files) - {"onset_s", "offset_s", "label", "confidence"}
        extra = {
            key: _decode_storage_array(data[key], encoded=key in encoded)
            for key in keys
        }
        label = (
            _decode_storage_array(data["label"], encoded="label" in encoded)
            if "label" in data.files
            else None
        )
        conf = data["confidence"] if "confidence" in data.files else None
        return EventSeries(
            onset_s=data["onset_s"],
            offset_s=data["offset_s"],
            label=label,
            confidence=conf,
            extra=extra,
            metadata=meta.get("metadata", {}),
            schema=meta.get("schema", "EventSeries/v1"),
            timebase=TimebaseSpec.from_dict(meta.get("timebase"), default_kind="events"),
            temporal_context=TemporalContext.from_dict(meta.get("temporal_context")),
        )
    if p.suffix == ".parquet":
        pq = _require_pyarrow()
        table = pq.read_table(p)
        data = table.to_pydict()
        meta = _read_meta(p.with_name("event_series.meta.json"))
        cols = set(data.keys()) - {"onset_s", "offset_s", "label", "confidence"}
        extra = {k: np.asarray(data[k]) for k in cols}
        return EventSeries(
            onset_s=np.asarray(data["onset_s"], dtype=np.float64),
            offset_s=np.asarray(data["offset_s"], dtype=np.float64),
            label=np.asarray(data["label"]) if "label" in data else None,
            confidence=np.asarray(data["confidence"], dtype=np.float64) if "confidence" in data else None,
            extra=extra,
            metadata=meta.get("metadata", {}),
            schema=meta.get("schema", "EventSeries/v1"),
            timebase=TimebaseSpec.from_dict(meta.get("timebase"), default_kind="events"),
            temporal_context=TemporalContext.from_dict(meta.get("temporal_context")),
        )
    raise ValueError(f"Unsupported event path: {path}")


def read_track_series(path: str | Path) -> TrackSeries:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        meta = _read_meta(p.with_name("track_series.meta.json"))
        encoded = set(meta.get("json_encoded_arrays", []))
        return TrackSeries(
            values=_decode_storage_array(data["values"], encoded="values" in encoded),
            times_s=data["times_s"],
            track_id=_decode_storage_array(
                data["track_id"],
                encoded="track_id" in encoded,
            ),
            dims=tuple(meta.get("dims", ("time", "track", "feature"))),
            coords=meta.get("coords", {}),
            metadata=meta.get("metadata", {}),
            schema=meta.get("schema", "TrackSeries/v1"),
            timebase=TimebaseSpec.from_dict(meta.get("timebase"), default_kind="frames"),
            time_bounds_s=data["time_bounds_s"] if "time_bounds_s" in data.files else None,
            temporal_context=TemporalContext.from_dict(meta.get("temporal_context")),
        )
    if p.suffix == ".zarr":
        zarr = _require_zarr()
        root = zarr.open_group(str(p), mode="r")
        encoded = set(root.attrs.get("json_encoded_arrays", []))
        return TrackSeries(
            values=_decode_storage_array(
                root["values"],
                encoded="values" in encoded,
            ),
            times_s=np.asarray(root["times_s"]),
            track_id=_decode_storage_array(
                root["track_id"],
                encoded="track_id" in encoded,
            ),
            dims=tuple(root.attrs.get("dims", ("time", "track", "feature"))),
            coords=dict(root.attrs.get("coords", {})),
            metadata=dict(root.attrs.get("metadata", {})),
            schema=str(root.attrs.get("schema", "TrackSeries/v1")),
            timebase=TimebaseSpec.from_dict(dict(root.attrs.get("timebase", {})), default_kind="frames"),
            time_bounds_s=np.asarray(root["time_bounds_s"]) if "time_bounds_s" in root else None,
            temporal_context=TemporalContext.from_dict(dict(root.attrs.get("temporal_context", {}))),
        )
    raise ValueError(f"Unsupported track path: {path}")
