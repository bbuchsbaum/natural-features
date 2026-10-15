#!/usr/bin/env python3
"""Run the frozen VocalF0/v1 qualification protocol with resumable artifacts."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import uuid
import zipfile

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from natural_features.core.feature_types import FeatureSeries  # noqa: E402
from natural_features.core.stimulus import AudioStimulus  # noqa: E402
from natural_features.core.timebase import TemporalContext, TimebaseSpec  # noqa: E402
from natural_features.features.audio.vocal_f0_benchmark import score_vocal_f0  # noqa: E402


VOCADITO_URL = "https://zenodo.org/records/5578807/files/vocadito.zip?download=1"
VOCADITO_MD5 = "dea40fd18f14d899643c4ba221b33a46"
BACKING = {
    "vibe": (
        "https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace.hq.ogg",
        "73d6443ef90a7c022f164e5aa90e56c2291585930b39b1656d0765abbc1f1779",
    ),
    "sugar": (
        "https://librosa.org/data/audio/Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.hq.ogg",
        "f062221a56a227cdb7c067cf2e6ac0e250a50012f7693ca0c8e31f05f83e49b1",
    ),
}
POSITIVE_CONDITIONS = ("solo", "mixture_0db", "mixture_minus6db")
NEGATIVE_CONDITIONS = ("backing", "silence")


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _download_verified(url: str, path: Path, expected: str, algorithm: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and _digest(path, algorithm) == expected:
        return
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as stream:
            shutil.copyfileobj(response, stream)
            stream.flush()
            os.fsync(stream.fileno())
        actual = _digest(tmp, algorithm)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _prepare_assets(assets: Path, allow_download: bool) -> None:
    archive = assets / "vocadito.zip"
    if allow_download:
        _download_verified(VOCADITO_URL, archive, VOCADITO_MD5, "md5")
    elif archive.is_file() and _digest(archive, "md5") != VOCADITO_MD5:
        raise ValueError("existing vocadito.zip MD5 does not match the frozen protocol")
    if archive.is_file() and not list(assets.rglob("vocadito_1.wav")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(assets)
    for name, (url, checksum) in BACKING.items():
        path = assets / f"{name}.ogg"
        if allow_download:
            _download_verified(url, path, checksum, "sha256")
        elif not path.is_file() or _digest(path) != checksum:
            raise ValueError(f"backing asset {path} is missing or has the wrong SHA256")


def _find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def _verify_vocadito_member(archive: Path, local: Path) -> None:
    if not archive.is_file() or _digest(archive, "md5") != VOCADITO_MD5:
        raise ValueError("a verified vocadito.zip is required to qualify extracted audio or references")
    with zipfile.ZipFile(archive) as zf:
        members = [member for member in zf.namelist() if Path(member).name == local.name]
        if len(members) != 1:
            raise ValueError(f"verified archive has no unique member for {local.name}")
        expected = hashlib.sha256(zf.read(members[0])).hexdigest()
    if _digest(local) != expected:
        raise ValueError(f"extracted file does not match verified archive member: {local}")


def _error_detail(exc: Exception) -> str:
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        chain.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(chain)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency is part of test environment
        raise RuntimeError("soundfile is required to read qualification audio") from exc
    values, sr_hz = sf.read(path, always_2d=True, dtype="float32")
    if not np.all(np.isfinite(values)) or values.shape[0] == 0:
        raise ValueError(f"invalid audio: {path}")
    return values.mean(axis=1), int(sr_hz)


def _to_16k_mono(samples: np.ndarray, sr_hz: int, seconds: float | None = None) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("audio must be finite mono or channel-last stereo")
    if sr_hz != 16000:
        factor = math.gcd(sr_hz, 16000)
        values = resample_poly(values, 16000 // factor, sr_hz // factor)
    if seconds is not None:
        values = values[: int(round(seconds * 16000))]
    if values.size == 0:
        raise ValueError("audio is empty after resampling")
    return values.astype(np.float32)


def _reference_mask(reference_times: np.ndarray, reference_hz: np.ndarray, n: int) -> np.ndarray:
    if len(reference_times) < 2:
        raise ValueError("reference needs at least two native timestamps for RMS masking")
    spacing = float(np.median(np.diff(reference_times)))
    samples_t = np.arange(n, dtype=np.float64) / 16000.0
    insertion = np.searchsorted(reference_times, samples_t)
    left = np.clip(insertion - 1, 0, len(reference_times) - 1)
    right = np.clip(insertion, 0, len(reference_times) - 1)
    choose_left = np.abs(samples_t - reference_times[left]) <= np.abs(reference_times[right] - samples_t)
    nearest = np.where(choose_left, left, right)
    return (np.abs(samples_t - reference_times[nearest]) <= spacing / 2) & (reference_hz[nearest] > 0)


def _rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if selected.size == 0:
        raise ValueError("reference-voiced sample mask is empty")
    return float(np.sqrt(np.mean(selected**2)))


def _mixture(source: np.ndarray, backing: np.ndarray, mask: np.ndarray, db: float) -> tuple[np.ndarray, dict]:
    source_rms = _rms(source, mask)
    backing_rms = _rms(backing, mask)
    if backing_rms <= 0:
        raise ValueError("backing has zero RMS on the reference-voiced mask")
    scale = source_rms / (10.0 ** (db / 20.0) * backing_rms)
    scaled_backing = backing * scale
    mix = source + scaled_backing
    peak_scale = min(1.0, 0.95 / max(float(np.max(np.abs(source))), float(np.max(np.abs(scaled_backing))), float(np.max(np.abs(mix))), 1e-12))
    return (mix * peak_scale).astype(np.float32), {
        "source_rms_masked": source_rms,
        "backing_rms_masked_before_scaling": backing_rms,
        "backing_scale": scale,
        "target_vocal_to_backing_db": db,
        "common_peak_scale": peak_scale,
    }


def _cache_key(audio: np.ndarray, label: str, protocol_hash: str, extractor_hash: str, device: str) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(audio).tobytes())
    digest.update(json.dumps({"label": label, "sr_hz": 16000, "protocol": protocol_hash, "extractor": extractor_hash, "device": device}, sort_keys=True).encode())
    return digest.hexdigest()


def _save_feature(path: Path, key: str, feature: FeatureSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        np.savez_compressed(tmp, cache_key=key, values=feature.values, times_s=feature.times_s,
                            bounds=feature.time_bounds_s, feature_names_json=json.dumps(feature.coords["feature"]),
                            timebase_json=json.dumps(feature.timebase.to_dict()), temporal_context_json=json.dumps(feature.temporal_context.to_dict()),
                            source_metadata_json=json.dumps(feature.metadata, sort_keys=True))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_feature(path: Path, key: str) -> FeatureSeries | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["cache_key"].item()) != key:
                return None
            metadata = json.loads(str(archive["source_metadata_json"].item()))
            versions = metadata.get("versions", {})
            if metadata.get("backend") != "demucs_htdemucs_crepe_full" or versions.get("demucs") != "4.0.1" or versions.get("torchcrepe") != "0.0.24":
                return None
            return FeatureSeries(values=archive["values"], times_s=archive["times_s"],
                                 coords={"feature": json.loads(str(archive["feature_names_json"].item()))},
                                 metadata=metadata, time_bounds_s=archive["bounds"],
                                 timebase=TimebaseSpec.from_dict(json.loads(str(archive["timebase_json"].item())), default_kind="frames"),
                                 temporal_context=TemporalContext.from_dict(json.loads(str(archive["temporal_context_json"].item()))))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _extract_cached(audio: np.ndarray, label: str, args: argparse.Namespace, protocol_hash: str, extractor_hash: str) -> tuple[FeatureSeries, dict]:
    runtime_identity = json.dumps({"device": args.device, "threads": args.threads,
                                   "versions": {name: version(name) for name in ("torch", "demucs", "torchcrepe", "scipy")}}, sort_keys=True)
    key = _cache_key(audio, label, protocol_hash, extractor_hash, runtime_identity)
    path = args.artifact_dir / f"{key}.npz"
    cached = _load_feature(path, key) if path.exists() else None
    if cached is not None:
        return cached, {"cache_key": key, "artifact": str(path), "artifact_sha256": _digest(path), "cached": True}
    from natural_features.features.audio.vocal_f0 import audio_vocal_f0
    feature = audio_vocal_f0(AudioStimulus.from_array(audio, 16000), separator_checkpoint=args.separator_checkpoint,
                             cache_dir=str(args.artifact_dir / "model_cache"), local_files_only=not args.download,
                             device=args.device)
    _save_feature(path, key, feature)
    return feature, {"cache_key": key, "artifact": str(path), "artifact_sha256": _digest(path), "cached": False}


def _macro(rows: list[dict], names: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in names:
        values = [row["metrics"].get(name) for row in rows]
        out[name] = None if len(values) != 6 or any(value is None for value in values) else float(np.mean(values))
    return out


def _scoring_reference(condition, reference_times, reference_hz, feature, duration):
    if condition in NEGATIVE_CONDITIONS:
        # Every native center in real audio is known to contain no singer.
        # The unrelated vocal annotation grid must not weight these negatives.
        times = feature.times_s[(feature.times_s >= 0) & (feature.times_s < duration)]
        return times, np.zeros(len(times))
    return reference_times, reference_hz


def _gate(report: dict, gates: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    summary = report.get("summary", {})
    for condition in POSITIVE_CONDITIONS:
        row = summary.get(condition, {})
        for metric, operator, threshold in (
            ("raw_pitch_accuracy", ">=", gates["raw_pitch_accuracy_min"]),
            ("voicing_recall", ">=", gates["voicing_recall_min"]),
            ("voicing_false_alarm_rate", "<=", gates["voicing_false_alarm_rate_max"]),
            ("median_abs_cents_error", "<=", gates["median_abs_cents_error_max"]),
            ("octave_error_rate", "<=", gates["octave_error_rate_max"]),
        ):
            value = row.get(metric)
            if value is None or not ((value >= threshold) if operator == ">=" else (value <= threshold)):
                failures.append(f"{condition}.{metric} {operator} {threshold} failed: {value}")
    for condition, gate_name in (("backing", "backing_false_alarm_rate_max"), ("silence", "silence_false_alarm_rate_max")):
        value = summary.get(condition, {}).get("voicing_false_alarm_rate")
        if value is None or value > gates[gate_name]:
            failures.append(f"{condition}.voicing_false_alarm_rate <= {gates[gate_name]} failed: {value}")
    analytic = report.get("analytic", {})
    if not analytic.get("complete", False):
        failures.append("analytic pitch checks incomplete")
    elif analytic.get("worst_median_abs_cents_error") is None or analytic["worst_median_abs_cents_error"] > gates["analytic_median_abs_cents_error_max"]:
        failures.append("analytic median cents gate failed")
    if report.get("errors"):
        failures.append("inference or input errors recorded")
    if report.get("completed_jobs") != 30:
        failures.append(f"incomplete jobs: {report.get('completed_jobs')}/30")
    return not failures, failures


def _analytic_feature(audio: np.ndarray, args: argparse.Namespace) -> FeatureSeries:
    from natural_features.features.audio import vocal_f0 as vf
    runtime = vf._load_runtime(args.separator_checkpoint, str(args.artifact_dir / "model_cache"), not args.download, args.device)
    hop = 160
    f0, confidence = runtime.pitch(audio, hop, 50.0, 1000.0)
    n = len(f0)
    if confidence.shape != f0.shape or not np.all(np.isfinite(f0)) or not np.all(np.isfinite(confidence)):
        raise ValueError("runtime.pitch produced invalid analytic output")
    voiced = confidence >= 0.21
    values = np.column_stack([np.where(voiced, f0, np.nan), confidence, np.ones(n), np.ones(n), voiced, ~voiced, np.zeros(n), np.zeros(n)])
    return FeatureSeries(values=values, times_s=np.arange(n) * hop / 16000.0,
                         coords={"feature": ["f0_hz", "confidence", "source_support", "source_supported", "voiced", "missing", "range_edge", "octave_jump"]},
                         metadata={"extractor_id": "analytic-runtime-pitch", "params_hash": "v1"})


def _analytic_cases(args: argparse.Namespace) -> dict:
    duration = 1.0
    times = np.arange(0.0, duration, 0.01)
    samples_t = np.arange(int(duration * 16000)) / 16000.0
    specs = {
        "tone_100": (100.0, np.full_like(times, 100.0), np.sin(2 * np.pi * 100.0 * samples_t)),
        "tone_150": (150.0, np.full_like(times, 150.0), np.sin(2 * np.pi * 150.0 * samples_t)),
        "tone_220": (220.0, np.full_like(times, 220.0), np.sin(2 * np.pi * 220.0 * samples_t)),
        "tone_440": (440.0, np.full_like(times, 440.0), np.sin(2 * np.pi * 440.0 * samples_t)),
        "glide_110_440": (None, 110.0 * (4.0 ** times), np.sin(2 * np.pi * 110.0 * (4.0 ** samples_t - 1.0) / math.log(4.0))),
        "missing_fundamental_220": (220.0, np.full_like(times, 220.0), sum(np.sin(2 * np.pi * 220.0 * k * samples_t) / k for k in (2, 3, 4))),
    }
    cases: dict[str, dict] = {}
    for name, (_, reference, audio) in specs.items():
        feature = _analytic_feature((0.1 * audio).astype(np.float32), args)
        metrics = score_vocal_f0(times, reference, feature)
        valid = (times >= 0.05) & (times <= 0.95)
        # score_vocal_f0 retains all frames for RPA/recall; its median is recomputed
        # here only over the required interior region.
        joint_times = feature.times_s[(feature.values[:, 4] > 0) & np.isfinite(feature.values[:, 0])]
        cents: list[float] = []
        for t, hz in zip(joint_times, feature.values[(feature.values[:, 4] > 0) & np.isfinite(feature.values[:, 0]), 0], strict=True):
            idx = int(np.searchsorted(times, t))
            candidates = [i for i in (idx - 1, idx) if 0 <= i < len(times)]
            if candidates:
                closest = min(candidates, key=lambda i: abs(times[i] - t))
                if abs(times[closest] - t) <= 0.005001 and valid[closest]:
                    cents.append(abs(1200.0 * math.log2(float(hz) / float(reference[closest]))))
        metrics["median_abs_cents_error_interior"] = None if not cents else float(np.median(cents))
        cases[name] = metrics
        print(f"analytic {name}: {metrics['median_abs_cents_error_interior']}", flush=True)
    medians = [row["median_abs_cents_error_interior"] for row in cases.values()]
    return {"complete": all(value is not None for value in medians), "cases": cases,
            "worst_median_abs_cents_error": None if any(value is None for value in medians) else float(max(medians))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--separator-checkpoint", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.assets = args.assets.resolve()
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass
    protocol_path = ROOT / "tests/benchmarks/thresholds/vocal_f0_v1.json"
    protocol_hash = _digest(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    extractor_path = ROOT / "src/natural_features/features/audio/vocal_f0.py"
    extractor_hash = _digest(extractor_path)
    report: dict = {"schema": "VocalF0QualificationReport/v1", "protocol_path": str(protocol_path), "protocol_sha256": protocol_hash,
                    "extractor_source_sha256": extractor_hash,
                    "runner_source_sha256": _digest(Path(__file__)),
                    "scorer_source_sha256": _digest(ROOT / "src/natural_features/features/audio/vocal_f0_benchmark.py"),
                    "extractor_revision": "vocal-f0-v1", "arguments": {"device": args.device, "threads": args.threads, "download": args.download},
                    "per_excerpt": [], "summary": {}, "analytic": {"complete": False}, "errors": [], "completed_jobs": 0, "qualified_scope": "unqualified"}
    _atomic_json(args.out_json, report)
    try:
        _prepare_assets(args.assets, args.download)
        backing_paths = {name: args.assets / f"{name}.ogg" for name in BACKING}
        for path in backing_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"missing backing asset {path}; pass --download to fetch verified assets")
        for excerpt in range(1, 7):
            wav = _find_one(args.assets, f"vocadito_{excerpt}.wav")
            annotation = _find_one(args.assets, f"vocadito_{excerpt}_f0.csv")
            _verify_vocadito_member(args.assets / "vocadito.zip", wav)
            _verify_vocadito_member(args.assets / "vocadito.zip", annotation)
            source, source_sr = _read_audio(wav)
            source = _to_16k_mono(source, source_sr, 10.0)
            reference = np.loadtxt(annotation, delimiter=",")
            if reference.ndim != 2 or reference.shape[1] < 2:
                raise ValueError(f"invalid reference annotation: {annotation}")
            ref_times, ref_hz = reference[:, 0], reference[:, 1]
            within_clip = ref_times < (len(source) / 16000.0)
            ref_times, ref_hz = ref_times[within_clip], ref_hz[within_clip]
            which_backing = "vibe" if excerpt % 2 else "sugar"
            backing, backing_sr = _read_audio(backing_paths[which_backing])
            backing = _to_16k_mono(backing, backing_sr, len(source) / 16000.0)
            if len(backing) < len(source):
                raise ValueError(f"backing asset is shorter than excerpt {excerpt}")
            backing = backing[:len(source)]
            mask = _reference_mask(ref_times, ref_hz, len(source))
            conditions: dict[str, tuple[np.ndarray, dict]] = {"solo": (source, {"recipe": "source only"}), "backing": (backing, {"recipe": "backing only", "backing": which_backing}), "silence": (np.zeros_like(source), {"recipe": "zero signal"})}
            for name, db in (("mixture_0db", 0.0), ("mixture_minus6db", -6.0)):
                audio, recipe = _mixture(source, backing, mask, db)
                recipe["backing"] = which_backing
                conditions[name] = (audio, recipe)
            for condition, (audio, recipe) in conditions.items():
                try:
                    feature, artifact = _extract_cached(audio, f"{excerpt}:{condition}", args, protocol_hash, extractor_hash)
                    score_times, score_reference = _scoring_reference(
                        condition, ref_times, ref_hz, feature, len(audio) / 16000,
                    )
                    metrics = score_vocal_f0(score_times, score_reference, feature)
                    row = {"excerpt_id": excerpt, "condition": condition, "metrics": metrics, "artifact": artifact,
                           "input_hashes": {"source_audio_sha256": _digest(wav), "reference_sha256": _digest(annotation),
                                            "backing_sha256": _digest(backing_paths[which_backing]),
                                            "rendered_16khz_mono_sha256": hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()},
                           "mixture_recipe": recipe}
                    report["per_excerpt"].append(row)
                    report["completed_jobs"] += 1
                    print(f"excerpt {excerpt} {condition}: done", flush=True)
                except Exception as exc:
                    detail = _error_detail(exc)
                    report["errors"].append({"excerpt_id": excerpt, "condition": condition, "error": detail})
                    print(f"excerpt {excerpt} {condition}: ERROR {detail}", flush=True)
                _atomic_json(args.out_json, report)
        for condition in (*POSITIVE_CONDITIONS, *NEGATIVE_CONDITIONS):
            report["summary"][condition] = _macro([row for row in report["per_excerpt"] if row["condition"] == condition], ["raw_pitch_accuracy", "voicing_recall", "voicing_false_alarm_rate", "median_abs_cents_error", "octave_error_rate", "gross_pitch_error_rate"])
        try:
            report["analytic"] = _analytic_cases(args)
        except Exception as exc:
            detail = _error_detail(exc)
            report["errors"].append({"analytic": detail})
            print(f"analytic: ERROR {detail}", flush=True)
        passed, failures = _gate(report, protocol["gates"])
        report["gate"] = {"passed": passed, "failures": failures}
        report["qualified_scope"] = protocol["scope"] if passed else "unqualified"
    except Exception as exc:
        detail = _error_detail(exc)
        report["errors"].append({"setup": detail})
        report["gate"] = {"passed": False, "failures": ["setup failed"]}
        print(f"setup: ERROR {detail}", flush=True)
    _atomic_json(args.out_json, report)
    return 0 if report.get("gate", {}).get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
