"""Artifact readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.timebase import TimebaseSpec


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


def read_feature_series(path: str | Path) -> FeatureSeries:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        meta = _read_meta(p.with_name("feature_series.meta.json"))
        return FeatureSeries(
            values=data["values"],
            times_s=data["times_s"],
            dims=tuple(meta.get("dims", ("time", "feature"))),
            coords=meta.get("coords", {}),
            metadata=meta.get("metadata", {}),
            timebase=TimebaseSpec(**meta.get("timebase", {"kind": "frames"})),
        )
    if p.suffix == ".zarr":
        zarr = _require_zarr()
        root = zarr.open_group(str(p), mode="r")
        return FeatureSeries(
            values=np.asarray(root["values"]),
            times_s=np.asarray(root["times_s"]),
            dims=tuple(root.attrs.get("dims", ("time", "feature"))),
            coords=dict(root.attrs.get("coords", {})),
            metadata=dict(root.attrs.get("metadata", {})),
            timebase=TimebaseSpec(**dict(root.attrs.get("timebase", {"kind": "frames"}))),
        )
    raise ValueError(f"Unsupported feature path: {path}")


def read_event_series(path: str | Path) -> EventSeries:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        meta = _read_meta(p.with_name("event_series.meta.json"))
        keys = set(data.files) - {"onset_s", "offset_s", "label", "confidence"}
        extra = {k: data[k] for k in keys}
        label = data["label"] if "label" in data.files else None
        conf = data["confidence"] if "confidence" in data.files else None
        return EventSeries(
            onset_s=data["onset_s"],
            offset_s=data["offset_s"],
            label=label,
            confidence=conf,
            extra=extra,
            metadata=meta.get("metadata", {}),
            timebase=TimebaseSpec(**meta.get("timebase", {"kind": "events"})),
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
            timebase=TimebaseSpec(**meta.get("timebase", {"kind": "events"})),
        )
    raise ValueError(f"Unsupported event path: {path}")


def read_track_series(path: str | Path) -> TrackSeries:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        meta = _read_meta(p.with_name("track_series.meta.json"))
        return TrackSeries(
            values=data["values"],
            times_s=data["times_s"],
            track_id=data["track_id"],
            dims=tuple(meta.get("dims", ("time", "track", "feature"))),
            coords=meta.get("coords", {}),
            metadata=meta.get("metadata", {}),
            timebase=TimebaseSpec(**meta.get("timebase", {"kind": "frames"})),
        )
    if p.suffix == ".zarr":
        zarr = _require_zarr()
        root = zarr.open_group(str(p), mode="r")
        return TrackSeries(
            values=np.asarray(root["values"]),
            times_s=np.asarray(root["times_s"]),
            track_id=np.asarray(root["track_id"]),
            dims=tuple(root.attrs.get("dims", ("time", "track", "feature"))),
            coords=dict(root.attrs.get("coords", {})),
            metadata=dict(root.attrs.get("metadata", {})),
            timebase=TimebaseSpec(**dict(root.attrs.get("timebase", {"kind": "frames"}))),
        )
    raise ValueError(f"Unsupported track path: {path}")

