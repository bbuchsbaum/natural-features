#!/usr/bin/env python3
"""Run a pinned, manifest-driven singing-phone benchmark without downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TemporalContext, TimebaseSpec
from natural_features.features.audio._music_contract import asset_digest
from natural_features.features.speech.phonology import ctc_phone_posteriors
from natural_features.features.speech.singing_benchmark import evaluate_singing_phones


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _checked(root: Path, raw: str, expected: str | None) -> tuple[Path, str]:
    path = _path(root, raw)
    if expected is None:
        raise ValueError(f"an explicit SHA256 is required for {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if expected is not None and actual.lower() != str(expected).lower():
        raise ValueError(f"SHA256 mismatch for {path}")
    return path, actual


def _checked_snapshot(root: Path, raw: str, expected: str | None) -> tuple[Path, str]:
    path = _path(root, raw)
    if expected is None:
        raise ValueError(f"an explicit snapshot digest is required for {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"model snapshot directory does not exist: {path}")
    actual = asset_digest(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(f"snapshot digest mismatch for {path}")
    return path, actual


def _load_audio(path: Path) -> tuple[AudioStimulus, int]:
    """Read integer or IEEE-float WAV through libsndfile, preserving source SR."""
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("soundfile is required to read qualification audio") from error
    samples, sr_hz = sf.read(path, dtype="float32", always_2d=False)
    source_sr_hz = int(sr_hz)
    return AudioStimulus.from_array(samples, source_sr_hz), source_sr_hz


def _artifact_series(path: Path) -> FeatureSeries:
    with np.load(path, allow_pickle=False) as data:
        labels = [str(x) for x in data["labels"].tolist()]
        bounds = data["bounds_s"] if "bounds_s" in data.files else None
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data.files else {"extractor_id": "retained_ctc_artifact", "params_hash": _sha256(path)}
        timebase = TimebaseSpec.from_dict(json.loads(str(data["timebase_json"].item())), default_kind="frames") if "timebase_json" in data.files else TimebaseSpec(kind="frames")
        context = TemporalContext.from_dict(json.loads(str(data["temporal_context_json"].item()))) if "temporal_context_json" in data.files else TemporalContext()
        return FeatureSeries(values=data["values"], times_s=data["times_s"], coords={"feature": labels}, time_bounds_s=bounds, metadata=metadata, timebase=timebase, temporal_context=context)


def _save_artifact(path: Path, series: FeatureSeries) -> None:
    np.savez_compressed(path, times_s=series.times_s, values=series.values, labels=np.asarray(series.coords["feature"]), bounds_s=series.time_bounds_s, metadata_json=json.dumps(series.metadata, sort_keys=True, default=str), timebase_json=json.dumps(series.timebase.to_dict(), sort_keys=True), temporal_context_json=json.dumps(series.temporal_context.to_dict(), sort_keys=True))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_manifest(
    manifest_path: Path,
    *,
    root: Path,
    artifact_dir: Path,
    on_item: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != 1:
        raise ValueError("singing-phone manifest_version must be 1")
    state = str(payload.get("state", ""))
    if state != "development":
        raise ValueError("v1 is development-only; qualification requires a separately verified protocol")
    token_roles = payload["token_roles"]
    inventory = payload["canonical_inventory"]
    items = list(payload.get("items", []))
    if not items:
        raise ValueError("singing-phone manifest has no items")
    ids = [str(dict(raw).get("id", f"item_{i:03d}")) for i, raw in enumerate(items)]
    if len(ids) != len(set(ids)):
        raise ValueError("singing-phone manifest item ids must be unique")
    if any(Path(item_id).name != item_id or item_id in {"", ".", ".."} for item_id in ids):
        raise ValueError("singing-phone artifact ids must be simple file names")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failed = False
    report: dict[str, Any] = {
        "manifest_sha256": _sha256(manifest_path), "state": state,
        "model_selection": "manifest-pinned", "network_downloads": False,
        "items": results, "n_failed": 0, "runtime_failure": False,
        "runtime": dict(runtime or {}),
        "acceptance_gates": "none; report is descriptive unless an external frozen protocol supplies gates",
    }
    for index, raw in enumerate(items):
        item = dict(raw)
        item_id = str(item.get("id", f"item_{index:03d}"))
        result: dict[str, Any] = {"id": item_id, "condition": item.get("condition"), "singer": item.get("singer"), "song": item.get("song"), "partition": item.get("partition"), "provenance": dict(item.get("provenance", {}))}
        try:
            source_hashes: dict[str, str] = {}
            source_hashes["reference_intervals"] = _json_digest(item["reference_intervals"])
            if "posterior_npz" in item:
                artifact, digest = _checked(root, str(item["posterior_npz"]), item.get("posterior_sha256"))
                source_hashes["posterior_npz"] = digest
                series = _artifact_series(artifact)
                result["artifacts"] = {"replay_posterior": {"path": str(artifact), "sha256": digest}}
            else:
                audio_path, digest = _checked(root, str(item["audio_path"]), item.get("audio_sha256"))
                source_hashes["audio"] = digest
                model_path, model_digest = _checked_snapshot(root, str(item["model_snapshot"]), item.get("model_snapshot_sha256"))
                source_hashes["model_snapshot"] = model_digest
                stimulus, source_sr_hz = _load_audio(audio_path)
                result["source_sample_rate_hz"] = source_sr_hz
                series = ctc_phone_posteriors(stimulus, model=str(model_path), local_files_only=True, drop_special_tokens=False, device=str(item.get("device", "auto")))
                generated = artifact_dir / f"{item_id}.npz"
                _save_artifact(generated, series)
                result["artifacts"] = {"generated_posterior": {"path": str(generated), "sha256": _sha256(generated)}}
            result["source_hashes"] = source_hashes
            result["native_metadata"] = series.metadata
            result["evaluation"] = evaluate_singing_phones(series, token_roles=item.get("token_roles", token_roles), canonical_inventory=item.get("canonical_inventory", inventory), reference_intervals=item["reference_intervals"], window_bounds_s=item["window_bounds_s"], bin_bounds_s=item.get("bin_bounds_s"), native_hop_s=item.get("native_hop_s"), reference_nonphone_labels=item.get("reference_nonphone_labels"))
        except Exception as error:  # preserve errors so incomplete runs cannot look clean
            failed = True
            result["error"] = {"type": type(error).__name__, "message": str(error)}
        results.append(result)
        report["n_failed"] = sum("error" in item_result for item_result in results)
        report["runtime_failure"] = failed
        if on_item is not None:
            on_item(report, result)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    root = args.root or args.manifest.parent
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.set_num_threads(1)
        runtime = {
            "torch_version": getattr(torch, "__version__", None),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        }
    except ImportError:
        runtime = {"torch_version": None, "torch_num_threads": None}
    def checkpoint(report: dict[str, Any], item: dict[str, Any]) -> None:
        _atomic_json(args.out_json, report)
        print(f"{len(report['items'])}: {item['id']}", flush=True)
    report = run_manifest(args.manifest, root=root, artifact_dir=args.artifact_dir, on_item=checkpoint, runtime=runtime)
    _atomic_json(args.out_json, report)
    return 1 if report["runtime_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
