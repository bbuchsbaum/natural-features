"""Experimental, separately typed evidence for vocal activity and vocal pitch.

External model code and weights are explicit local, hash-pinned assets. Nothing
is downloaded or substituted during extraction. Qualification is separate.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.metadata import version
import importlib.util
from pathlib import Path

import numpy as np

from natural_features.core.backend_errors import BackendDependencyError, BackendInferenceError, BackendLoadError
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio._music_contract import computation_context, mono_audio
from natural_features.features.audio.vocal_f0_benchmark import _nearest_index
from natural_features.features.common import extractor_metadata

ACTIVITY_CODE_SHA256 = "ad2710170f5c30e2916ac75297ea2f0de73f3b05fb076bfec6bda36437e09d1c"
ACTIVITY_WEIGHTS_SHA256 = "f09cd6a55c83c13640622ce67480443ce282e3124dd22fccee8a898bb2ba9dee"
ACTIVITY_CONFIG_SHA256 = "3e27dd2bf027ac10998add291b33fac77bb9f53822b1dab3601f902a89086f77"
ACTIVITY_REVISION = "CPJKU/veracity@0983900f136173015f3c5d0b116be014edd33905"
RMVPE_CODE_SHA256 = "6dfe9b90bc2b283d20680d97129d1d0ec5d5af6d5323ece67d1b3623d95ee3de"
RMVPE_WEIGHTS_SHA256 = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"
RMVPE_REVISION = "RVC@9f2f0559e6932c10c48642d404e7d2e771d9db43;HF:lj1995/VoiceConversionWebUI@e6d0c1a17da07c33557852f9dfa2bd44cc75737d"


def _verified(path, expected):
    path = Path(path).expanduser().resolve()
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BackendLoadError("vocal_evidence", f"required local asset is unavailable: {path}") from exc
    if actual != expected:
        raise BackendLoadError("vocal_evidence", f"asset identity mismatch: {path}")
    return path


@lru_cache(maxsize=2)
def _rmvpe_runtime(source_root, device):
    try:
        import torch
        import librosa  # noqa: F401
    except (ImportError, OSError) as exc:
        raise BackendDependencyError("rmvpe", "torch, librosa and scipy are required") from exc
    root = Path(source_root)
    code = _verified(root / "rmvpe.py", RMVPE_CODE_SHA256)
    weights = _verified(root / "rmvpe.pt", RMVPE_WEIGHTS_SHA256)
    try:
        source = code.read_text()
        # Isolate the fixed non-JIT CPU path. No architecture, frontend or
        # decoder operation is altered; optional global Intel/JIT hooks vanish.
        begin, end = source.index("from infer.lib import jit"), source.index("import torch.nn as nn")
        source = source[:begin] + source[end:]
        source = source.replace('torch.load(model_path, map_location="cpu")',
                                'torch.load(model_path, map_location="cpu", weights_only=True)')
        spec = importlib.util.spec_from_loader("_nf_rvc_rmvpe", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(compile(source, str(code), "exec"), module.__dict__)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return module.RMVPE(str(weights), is_half=False, device=device, use_jit=False)
    except Exception as exc:
        raise BackendLoadError("rmvpe", "pinned RVC RMVPE model could not be loaded") from exc


def audio_rmvpe(stimulus, *, source_root: str, threshold=0.03, fmin=50.0, fmax=1000.0, device="cpu"):
    """Fixed RVC RMVPE on supplied audio, without implicit source separation.

    A candidate outside the declared range is rejected, not forced to a nearby
    bin. The salience threshold is separate from vocal presence. This function
    supports CPU FP32 only; it never silently changes inference or chunks audio.
    """
    from natural_features.features.audio.vocal_f0 import _resample, _frame_rms
    if device != "cpu":
        raise ValueError("the pinned experimental RMVPE adapter currently supports CPU only")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 32.71 <= fmin < fmax <= 2006):
        raise ValueError("require 32.71 <= fmin < fmax <= 2006")
    mono = mono_audio(stimulus)
    root = Path(source_root).resolve()
    _verified(root / "rmvpe.py", RMVPE_CODE_SHA256)
    _verified(root / "rmvpe.pt", RMVPE_WEIGHTS_SHA256)
    runtime = _rmvpe_runtime(str(root), device)
    try:
        import torch
        audio = _resample(mono, stimulus.sr_hz, 16000)
        with torch.inference_mode():
            mel = runtime.mel_extractor(torch.from_numpy(audio.copy())[None], center=True)
            salience = runtime.mel2hidden(mel)[0].cpu().numpy()
        centers = np.arange(1 + len(audio) // 160) * 160
        if salience.shape != (len(centers), 360) or not np.isfinite(salience).all() or np.any((salience < 0) | (salience > 1)):
            raise ValueError("RMVPE returned invalid native salience")
        # Decode raw candidates with the upstream local-average decoder, retaining
        # rejection separately. The all-zero limiting case has no pitch candidate.
        confidence = salience.max(axis=1)
        candidate = np.full(len(centers), np.nan)
        has_energy = confidence > 0
        candidate[has_energy] = runtime.decode(salience[has_energy], thred=0)
        if not np.isfinite(candidate[has_energy]).all() or np.any(candidate[has_energy] <= 0):
            raise ValueError("RMVPE returned invalid pitch candidates")
    except Exception as exc:
        raise BackendInferenceError("rmvpe", "RMVPE inference failed") from exc
    duration = len(mono) / stimulus.sr_hz
    times = centers / 16000
    available = times < duration
    within = (candidate >= fmin) & (candidate <= fmax)
    voiced = available & within & (confidence > threshold)
    pitch = np.where(voiced, candidate, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        edge = (1200 * np.log2(candidate / fmin) < 20) | (1200 * np.log2(fmax / candidate) < 20)
        delta = abs(1200 * np.diff(np.log2(pitch)))
    jump = np.r_[False, voiced[1:] & voiced[:-1] & (abs(delta - 1200) <= 100)]
    padded = (centers < 512) | ((centers + 512) / 16000 > duration)
    unknown_source = np.full(len(centers), np.nan)
    values = np.column_stack([pitch, confidence, unknown_source, unknown_source, voiced,
                              ~voiced, edge, jump, padded, candidate, _frame_rms(audio, centers), available])
    names = ["f0_hz", "confidence", "source_support", "source_supported", "voiced", "missing",
             "range_edge", "octave_jump", "padded", "candidate_f0_hz", "input_rms", "available"]
    bounds = stimulus.start_offset_s + np.clip(np.column_stack([centers - 512, centers + 512]) / 16000, 0, duration)
    context = computation_context(stimulus, policy="bidirectional",
        bounds_s=[[stimulus.start_offset_s, stimulus.start_offset_s + duration]],
        preprocessing={"resampler": "scipy.signal.resample_poly; default Kaiser5; zero boundary",
                       "original_sr_hz": stimulus.sr_hz, "model_sr_hz": 16000,
                       "stft": "1024-point periodic Hann; centered reflect padding; hop160",
                       "mel_pad_frames": int((-len(centers)) % 32), "input_context": "whole supplied audio; bidirectional GRU",
                       "short_input_policy": "typed inference error when reflect padding is unavailable"})
    return FeatureSeries(values=values, times_s=stimulus.start_offset_s + times, coords={"feature": names},
        time_bounds_s=bounds, temporal_context=stimulus.temporal_context,
        timebase=TimebaseSpec(kind="audio_hop", reference=stimulus.clock, hop_s=0.01, window_s=0.064),
        metadata=extractor_metadata("audio.vocal_f0.rmvpe", code_version="vocal-evidence-v1",
            params={"threshold": threshold, "fmin": fmin, "fmax": fmax, "device": device},
            model_revision=f"{RMVPE_REVISION};sha256:{RMVPE_WEIGHTS_SHA256}",
            extra={"qualification": "experimental; not qualified", "source_code_sha256": RMVPE_CODE_SHA256,
                   "confidence_semantics": "maximum RMVPE salience; not calibrated vocal presence",
                   "source_support_semantics": "NaN: no independent vocal-presence evidence",
                   "source_selection": "supplied waveform; no implicit separation or singer identity",
                   "compatibility_change": "omit optional JIT/Intel imports; use weights_only state-dictionary load",
                   "missing_f0": "NaN; never encoded as phonetic absence",
                   "computation_context": context,
                   "versions": {name: version(name) for name in ("torch", "librosa", "scipy")}}))


@lru_cache(maxsize=2)
def _activity_runtime(source_root, device):
    try:
        import torch
        import torchaudio  # noqa: F401
        import librosa  # noqa: F401
        import resampy  # noqa: F401
    except (ImportError, OSError) as exc:
        raise BackendDependencyError("vocal_activity", "torch, torchaudio, librosa and resampy are required") from exc
    root = Path(source_root)
    code = _verified(root / "svd/core/model.py", ACTIVITY_CODE_SHA256)
    weights = _verified(root / "dataset/jamendo/models/model_log_0mean/model.pt", ACTIVITY_WEIGHTS_SHA256)
    _verified(root / "dataset/jamendo/models/model_log_0mean/model.vars", ACTIVITY_CONFIG_SHA256)
    try:
        # The pinned architecture does not use audio I/O. Remove its obsolete
        # global backend-selection call, without changing any network operation.
        source = code.read_text().replace("torchaudio.set_audio_backend('sox_io')", "")
        spec = importlib.util.spec_from_loader("_nf_cpjku_activity", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(compile(source, str(code), "exec"), module.__dict__)
        config = {"n_mels": 80, "sample_rate": 22050, "f_min": 27.5, "f_max": 8000,
                  "n_stft": 513, "frame_len": 1024, "filterbank": "mel_orig", "spec_len": 372,
                  "magscale": "log", "arch.batch_norm": 1, "arch.convdrop": "none",
                  "arch": "ismir2015", "arch.firstconv_zeromean": "0mean"}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            net = module.SingingVoiceDetector(config)
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        return net.to(device).eval()
    except Exception as exc:
        raise BackendLoadError("vocal_activity", "pinned CPJKU model could not be loaded") from exc


def audio_vocal_activity(stimulus, *, source_root: str, device="cpu"):
    """CPJKU singing-activity score; not pitched voicing or calibrated confidence.

    The 115-frame convolutional context spans about 1.67 seconds. Native centers
    occur every 315 samples at 22050 Hz. Padding and real observed support remain
    explicit; no pitch or phone output is accepted/rejected by this function.
    """
    mono = mono_audio(stimulus)
    root = Path(source_root).resolve()
    _verified(root / "svd/core/model.py", ACTIVITY_CODE_SHA256)
    _verified(root / "dataset/jamendo/models/model_log_0mean/model.pt", ACTIVITY_WEIGHTS_SHA256)
    _verified(root / "dataset/jamendo/models/model_log_0mean/model.vars", ACTIVITY_CONFIG_SHA256)
    net = _activity_runtime(str(root), device)
    try:
        import torch
        import librosa
        from torchaudio.transforms import Spectrogram
        audio = librosa.resample(mono, orig_sr=stimulus.sr_hz, target_sr=22050, res_type="kaiser_best")
        with torch.inference_mode():
            spec = Spectrogram(n_fft=1024, hop_length=315, power=1.0,
                               center=True, pad_mode="reflect")(torch.from_numpy(audio))
            spec = spec.T[None, None].to(device)
            padding = torch.zeros((1, 1, 57, 513), device=device)
            probability = net.probabilities(torch.cat([padding, spec, padding], dim=2)).cpu().numpy()
        centers = np.arange(1 + len(audio) // 315) * 315
        if probability.shape != centers.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("invalid native activity output")
    except Exception as exc:
        raise BackendInferenceError("vocal_activity", "CPJKU activity inference failed") from exc
    duration = len(mono) / stimulus.sr_hz
    times = centers / 22050
    radius = (57 * 315 + 512) / 22050
    available = times < duration
    padded = (times - radius < 0) | (times + radius > duration)
    bounds = stimulus.start_offset_s + np.clip(np.column_stack([times - radius, times + radius]), 0, duration)
    context = computation_context(stimulus, policy="bidirectional",
                                  bounds_s=[[stimulus.start_offset_s, stimulus.start_offset_s + duration]],
                                  preprocessing={"resampler": "librosa/resampy kaiser_best",
                                                 "model_sr_hz": 22050, "hop_samples": 315,
                                                 "stft": "1024-point periodic Hann; centered reflect padding",
                                                 "context_frames": 115, "magnitude_padding_each_side": 57,
                                                 "source_sr_hz": stimulus.sr_hz,
                                                 "short_input_policy": "typed inference error when reflect padding is unavailable"})
    return FeatureSeries(values=np.column_stack([probability, available, padded]),
                         times_s=stimulus.start_offset_s + times,
                         coords={"feature": ["vocal_activity_score", "available", "padded"]},
                         time_bounds_s=bounds, temporal_context=stimulus.temporal_context,
                         timebase=TimebaseSpec(kind="audio_hop", reference=stimulus.clock, hop_s=1 / 70),
                         metadata=extractor_metadata("audio.vocal_activity.cpjku", code_version="vocal-evidence-v1",
                             model_revision=f"{ACTIVITY_REVISION};sha256:{ACTIVITY_WEIGHTS_SHA256}",
                             params={"device": device}, extra={"qualification": "experimental; not qualified",
                                 "confidence_semantics": "sigmoid singing-activity score; not calibrated vocal probability",
                                 "source_code_sha256": ACTIVITY_CODE_SHA256,
                                 "compatibility_change": "omit obsolete torchaudio.set_audio_backend call only",
                                 "computation_context": context,
                                 "versions": {name: version(name) for name in ("torch", "torchaudio", "librosa", "resampy")}}))


def pitch_with_activity(estimate: FeatureSeries, activity: FeatureSeries, *, threshold=0.51,
                        max_time_error_s=1 / 140 + 1e-8):
    """Explicit experimental pitch intersection; never a phonetic-support mask."""
    from dataclasses import replace
    if estimate.clock != activity.clock:
        raise ValueError("pitch and vocal activity must share a clock")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if not np.isfinite(max_time_error_s) or max_time_error_s < 0:
        raise ValueError("max_time_error_s must be finite and nonnegative")
    names, anames = list(estimate.coords["feature"]), list(activity.coords["feature"])
    score = np.full(len(estimate.times_s), np.nan)
    for j, time in enumerate(estimate.times_s):
        i = _nearest_index(activity.times_s, time, max_time_error_s)
        if i is not None and activity.values[i, anames.index("available")] > 0:
            score[j] = activity.values[i, anames.index("vocal_activity_score")]
    if np.any(np.isinf(score)) or np.any((score[np.isfinite(score)] < 0) | (score[np.isfinite(score)] > 1)):
        raise ValueError("invalid vocal activity scores")
    values = np.array(estimate.values, copy=True)
    accepted = (values[:, names.index("voiced")] > 0) & (score > threshold)
    values[:, names.index("voiced")] = accepted
    values[:, names.index("missing")] = ~accepted
    values[~accepted, names.index("f0_hz")] = np.nan
    values[:, names.index("octave_jump")] = 0
    if len(values) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            jumps = abs(1200 * np.diff(np.log2(values[:, names.index("f0_hz")])))
        values[1:, names.index("octave_jump")] = accepted[1:] & accepted[:-1] & (abs(jumps - 1200) <= 100)
    contexts = [parent.metadata.get("computation_context") for parent in (estimate, activity)]
    if any(not isinstance(ctx, dict) or "input_bounds_s" not in ctx for ctx in contexts):
        raise ValueError("both inputs must declare their computation context")
    if any(ctx.get("clock") != str(estimate.clock) for ctx in contexts):
        raise ValueError("parent computation contexts must use the output clock")
    input_bounds = np.asarray([ctx["input_bounds_s"] for ctx in contexts], dtype=float)
    if input_bounds.shape != (2, 2) or not np.isfinite(input_bounds).all() or np.any(input_bounds[:, 1] < input_bounds[:, 0]):
        raise ValueError("invalid parent computation context")
    union = [float(input_bounds[:, 0].min()), float(input_bounds[:, 1].max())]
    context = {"schema": "MusicComputationContext/v1", "clock": str(estimate.clock),
               "policy": "bidirectional", "input_bounds_s": union, "dependency_bounds_s": [union],
               "preprocessing": {"operation": "pitch/activity intersection; conservative union of parent inputs"}}
    metadata = extractor_metadata("audio.vocal_f0.activity_intersection", code_version="vocal-evidence-v1",
        params={"threshold": threshold, "max_time_error_s": max_time_error_s,
                "pitch_params_hash": estimate.metadata.get("params_hash"),
                "activity_params_hash": activity.metadata.get("params_hash"),
                "pitch_extractor_id": estimate.metadata.get("extractor_id"),
                "activity_extractor_id": activity.metadata.get("extractor_id"),
                "pitch_code_version": estimate.metadata.get("code_version"),
                "activity_code_version": activity.metadata.get("code_version")},
        model_revision=f"pitch:{estimate.metadata.get('model_revision')};activity:{activity.metadata.get('model_revision')}",
        extra={"computation_context": context, "qualification": "experimental; not qualified",
               "source_pitch_metadata": estimate.metadata, "source_activity_metadata": activity.metadata,
               "activity_intersection": {"threshold": threshold, "comparison": "strict >",
                   "max_time_error_s": max_time_error_s, "matched_native_frames": int(np.isfinite(score).sum()),
                   "phonetic_support": "not defined or modified"}})
    return replace(estimate, values=values, metadata=metadata)
