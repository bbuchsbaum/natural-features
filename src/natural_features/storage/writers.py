"""Artifact writers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.util.io import atomic_numpy_savez, atomic_write_json


def _require_zarr() -> Any:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("zarr is required for zarr writes: pip install natural-features[storage]") from exc
    return zarr


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet writes: pip install natural-features[storage]"
        ) from exc
    return pa, pq


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload, sort_keys=True, indent=2)


def _replace_path(tmp: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.replace(tmp, dst)


def write_feature_series(obj: FeatureSeries, out_dir: str | Path, *, fmt: str = "zarr") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        path = out / "feature_series.npz"
        atomic_numpy_savez(path, values=obj.values, times_s=obj.times_s)
        _write_json(out / "feature_series.meta.json", {"dims": list(obj.dims), "coords": obj.coords, "metadata": obj.metadata})
        return path
    if fmt == "zarr":
        zarr = _require_zarr()
        path = out / "feature_series.zarr"
        tmp = out / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            root = zarr.open_group(str(tmp), mode="w")
            root.create_dataset("values", data=obj.values, chunks=True)
            root.create_dataset("times_s", data=obj.times_s, chunks=True)
            root.attrs["dims"] = list(obj.dims)
            root.attrs["coords"] = obj.coords
            root.attrs["metadata"] = obj.metadata
            root.attrs["schema"] = obj.schema
            root.attrs["timebase"] = obj.timebase.__dict__
            _replace_path(tmp, path)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return path
    raise ValueError(f"Unsupported format for FeatureSeries: {fmt}")


def write_event_series(obj: EventSeries, out_dir: str | Path, *, fmt: str = "parquet") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        pa, pq = _require_pyarrow()
        cols: dict[str, Any] = {
            "onset_s": pa.array(obj.onset_s),
            "offset_s": pa.array(obj.offset_s),
        }
        if obj.label is not None:
            cols["label"] = pa.array(obj.label)
        if obj.confidence is not None:
            cols["confidence"] = pa.array(obj.confidence)
        for key, arr in obj.extra.items():
            cols[key] = pa.array(arr)
        table = pa.table(cols)
        path = out / "event_series.parquet"
        tmp = out / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            pq.write_table(table, tmp)
            _replace_path(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        _write_json(out / "event_series.meta.json", {"metadata": obj.metadata, "schema": obj.schema, "timebase": obj.timebase.__dict__})
        return path
    if fmt == "npz":
        path = out / "event_series.npz"
        payload: dict[str, Any] = {
            "onset_s": obj.onset_s,
            "offset_s": obj.offset_s,
        }
        if obj.label is not None:
            payload["label"] = np.asarray(obj.label)
        if obj.confidence is not None:
            payload["confidence"] = np.asarray(obj.confidence)
        payload.update({k: np.asarray(v) for k, v in obj.extra.items()})
        atomic_numpy_savez(path, **payload)
        _write_json(out / "event_series.meta.json", {"metadata": obj.metadata, "schema": obj.schema, "timebase": obj.timebase.__dict__})
        return path
    raise ValueError(f"Unsupported format for EventSeries: {fmt}")


def write_track_series(obj: TrackSeries, out_dir: str | Path, *, fmt: str = "zarr") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        path = out / "track_series.npz"
        atomic_numpy_savez(path, values=obj.values, times_s=obj.times_s, track_id=obj.track_id)
        _write_json(out / "track_series.meta.json", {"dims": list(obj.dims), "coords": obj.coords, "metadata": obj.metadata})
        return path
    if fmt == "zarr":
        zarr = _require_zarr()
        path = out / "track_series.zarr"
        tmp = out / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            root = zarr.open_group(str(tmp), mode="w")
            root.create_dataset("values", data=obj.values, chunks=True)
            root.create_dataset("times_s", data=obj.times_s, chunks=True)
            root.create_dataset("track_id", data=obj.track_id, chunks=True)
            root.attrs["dims"] = list(obj.dims)
            root.attrs["coords"] = obj.coords
            root.attrs["metadata"] = obj.metadata
            root.attrs["schema"] = obj.schema
            root.attrs["timebase"] = obj.timebase.__dict__
            _replace_path(tmp, path)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return path
    raise ValueError(f"Unsupported format for TrackSeries: {fmt}")
