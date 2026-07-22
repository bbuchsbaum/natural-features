"""Artifact writers."""

from __future__ import annotations

import json
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


def _zarr_create_array(root: Any, name: str, data: np.ndarray) -> None:
    """Write an array through the Zarr v2/v3 compatible group surface."""

    array = np.asarray(data)
    create_array = getattr(root, "create_array", None)
    if create_array is not None:
        string_dtype = getattr(getattr(np, "dtypes", None), "StringDType", None)
        if string_dtype is not None and array.dtype.kind in {"U", "S"}:
            array = np.asarray(array, dtype=string_dtype())
        create_array(name, data=array)
        return
    root.create_dataset(name, data=array, chunks=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _storage_safe_array(value: Any) -> tuple[np.ndarray, bool]:
    """Encode object arrays as portable JSON strings without pickle."""

    array = np.asarray(value)
    if array.dtype != object:
        return array, False
    encoded = [
        json.dumps(item, default=_json_default, sort_keys=True, separators=(",", ":"))
        for item in array.reshape(-1)
    ]
    return np.asarray(encoded, dtype=np.str_).reshape(array.shape), True


def _temporal_meta(obj: FeatureSeries | EventSeries | TrackSeries) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": obj.metadata,
        "schema": obj.schema,
        "timebase": obj.timebase.to_dict(),
        "temporal_context": obj.temporal_context.to_dict(),
    }
    if isinstance(obj, (FeatureSeries, TrackSeries)):
        payload.update({"dims": list(obj.dims), "coords": obj.coords})
    return payload


def write_feature_series(obj: FeatureSeries, out_dir: str | Path, *, fmt: str = "zarr") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        path = out / "feature_series.npz"
        values, values_encoded = _storage_safe_array(obj.values)
        arrays: dict[str, Any] = {"values": values, "times_s": obj.times_s}
        if obj.time_bounds_s is not None:
            arrays["time_bounds_s"] = obj.time_bounds_s
        atomic_numpy_savez(path, **arrays)
        meta = _temporal_meta(obj)
        meta["json_encoded_arrays"] = ["values"] if values_encoded else []
        _write_json(out / "feature_series.meta.json", meta)
        return path
    if fmt == "zarr":
        zarr = _require_zarr()
        path = out / "feature_series.zarr"
        tmp = out / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            root = zarr.open_group(str(tmp), mode="w")
            values, values_encoded = _storage_safe_array(obj.values)
            _zarr_create_array(root, "values", values)
            _zarr_create_array(root, "times_s", obj.times_s)
            if obj.time_bounds_s is not None:
                _zarr_create_array(root, "time_bounds_s", obj.time_bounds_s)
            root.attrs["dims"] = list(obj.dims)
            root.attrs["coords"] = obj.coords
            root.attrs["metadata"] = obj.metadata
            root.attrs["schema"] = obj.schema
            root.attrs["timebase"] = obj.timebase.to_dict()
            root.attrs["temporal_context"] = obj.temporal_context.to_dict()
            root.attrs["json_encoded_arrays"] = ["values"] if values_encoded else []
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
        _write_json(out / "event_series.meta.json", _temporal_meta(obj))
        return path
    if fmt == "npz":
        path = out / "event_series.npz"
        payload: dict[str, Any] = {
            "onset_s": obj.onset_s,
            "offset_s": obj.offset_s,
        }
        encoded_arrays: list[str] = []
        if obj.label is not None:
            payload["label"], encoded = _storage_safe_array(obj.label)
            if encoded:
                encoded_arrays.append("label")
        if obj.confidence is not None:
            payload["confidence"] = np.asarray(obj.confidence)
        for key, value in obj.extra.items():
            payload[key], encoded = _storage_safe_array(value)
            if encoded:
                encoded_arrays.append(key)
        atomic_numpy_savez(path, **payload)
        meta = _temporal_meta(obj)
        meta["json_encoded_arrays"] = encoded_arrays
        _write_json(out / "event_series.meta.json", meta)
        return path
    raise ValueError(f"Unsupported format for EventSeries: {fmt}")


def write_track_series(obj: TrackSeries, out_dir: str | Path, *, fmt: str = "zarr") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        path = out / "track_series.npz"
        values, values_encoded = _storage_safe_array(obj.values)
        track_id, track_id_encoded = _storage_safe_array(obj.track_id)
        arrays = {"values": values, "times_s": obj.times_s, "track_id": track_id}
        if obj.time_bounds_s is not None:
            arrays["time_bounds_s"] = obj.time_bounds_s
        atomic_numpy_savez(path, **arrays)
        meta = _temporal_meta(obj)
        meta["json_encoded_arrays"] = [
            name
            for name, encoded in (
                ("values", values_encoded),
                ("track_id", track_id_encoded),
            )
            if encoded
        ]
        _write_json(out / "track_series.meta.json", meta)
        return path
    if fmt == "zarr":
        zarr = _require_zarr()
        path = out / "track_series.zarr"
        tmp = out / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            root = zarr.open_group(str(tmp), mode="w")
            values, values_encoded = _storage_safe_array(obj.values)
            track_id, track_id_encoded = _storage_safe_array(obj.track_id)
            _zarr_create_array(root, "values", values)
            _zarr_create_array(root, "times_s", obj.times_s)
            _zarr_create_array(root, "track_id", track_id)
            if obj.time_bounds_s is not None:
                _zarr_create_array(root, "time_bounds_s", obj.time_bounds_s)
            root.attrs["dims"] = list(obj.dims)
            root.attrs["coords"] = obj.coords
            root.attrs["metadata"] = obj.metadata
            root.attrs["schema"] = obj.schema
            root.attrs["timebase"] = obj.timebase.to_dict()
            root.attrs["temporal_context"] = obj.temporal_context.to_dict()
            root.attrs["json_encoded_arrays"] = [
                name
                for name, encoded in (
                    ("values", values_encoded),
                    ("track_id", track_id_encoded),
                )
                if encoded
            ]
            _replace_path(tmp, path)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return path
    raise ValueError(f"Unsupported format for TrackSeries: {fmt}")
