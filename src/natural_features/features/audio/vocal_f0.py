"""Source-selected vocal F0 with fixed Demucs and CREPE assets.

Source support is an energy heuristic, not a calibrated singer probability.
Qualification is recorded separately from extraction; see docs/vocal_f0.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import version
from math import gcd
from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np

from natural_features.core.backend_errors import BackendDependencyError, BackendInferenceError, BackendLoadError
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio._music_contract import computation_context, mono_audio
from natural_features.features.common import extractor_metadata

SEPARATOR_SHA256 = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
CREPE_SHA256 = "133225604dedd2e4005f8bbd1bd0a2ec073ba8b7a6cd31ff6d5edbbfa3539986"
SEPARATOR_FILE = "955717e8-8726e21a.th"
SEPARATOR_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/" + SEPARATOR_FILE
FEATURES = ["f0_hz", "confidence", "source_support", "source_supported", "voiced", "missing", "range_edge", "octave_jump", "padded"]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resample(samples, source_sr, target_sr):
    if source_sr == target_sr:
        return np.asarray(samples, dtype=np.float32)
    from scipy.signal import resample_poly
    factor = gcd(source_sr, target_sr)
    return resample_poly(samples, target_sr // factor, source_sr // factor, axis=0).astype(np.float32)


def _deterministic_decode(probabilities):
    """CREPE's local weighted decoder without random pitch-bin dithering."""
    import torch
    bins = probabilities.argmax(dim=1)
    index = torch.arange(360, device=probabilities.device)[None, :, None]
    weights = torch.sigmoid(probabilities).masked_fill((index - bins[:, None]).abs() > 4, 0)
    cents = 20.0 * index + 1997.3794084376191
    pitch = 10 * 2 ** ((weights * cents).sum(dim=1) / weights.sum(dim=1) / 1200)
    return bins, pitch


@dataclass
class _Runtime:
    torch: object
    crepe: object
    separator: object
    pitch_model: object
    device: str
    versions: dict[str, str]

    @property
    def sources(self):
        return self.separator.sources

    def separate(self, stereo):
        from demucs.apply import apply_model
        torch = self.torch
        mix = torch.as_tensor(stereo.T.copy(), device=self.device)
        ref = mix.mean(0)
        mean, scale = ref.mean(), ref.std(unbiased=False)
        if scale <= 1e-10:
            return np.zeros((len(stereo), len(self.sources)), np.float32)
        with torch.inference_mode():
            stems = apply_model(
                self.separator, ((mix - mean) / scale)[None], shifts=0,
                split=True, overlap=0.25, device=self.device, num_workers=0,
            )[0]
            # DC is not evidence of a periodic source. Adding the mixture mean
            # to every stem can turn tiny leakage into false source support.
            restored = stems * scale
            restored = restored - restored.mean(dim=-1, keepdim=True)
        return restored.mean(1).T.cpu().numpy()

    def pitch(self, audio, hop, fmin, fmax):
        torch = self.torch
        pitches, confidence = [], []
        with torch.inference_mode():
            for frames in self.crepe.preprocess(
                torch.as_tensor(audio[None].copy()), 16000, hop,
                batch_size=256, device=self.device, pad=True,
            ):
                probs = self.pitch_model(frames).T[None]
                f0, strength = self.crepe.core.postprocess(
                    probs, fmin, fmax, _deterministic_decode, return_periodicity=True,
                )
                pitches.append(f0[0].cpu().numpy())
                confidence.append(strength[0].cpu().numpy())
        return np.concatenate(pitches), np.concatenate(confidence)


@lru_cache(maxsize=2)
def _cached_runtime(checkpoint, device):
    import torch
    import torchcrepe
    from demucs.states import load_model
    try:
        # Only the exact upstream content is unpickled. No arbitrary checkpoint
        # loading and no implicit download fallback are supported.
        with torch.random.fork_rng(devices=[]):
            model = load_model(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
            pitch = torchcrepe.Crepe("full")
        weights = Path(torchcrepe.__file__).parent / "assets" / "full.pth"
        pitch.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        if (model.samplerate != 44100 or model.audio_channels != 2 or "vocals" not in model.sources
                or float(model.segment) != 7.8):
            raise ValueError("unexpected separator sample rate, channels or source labels")
        return _Runtime(torch, torchcrepe, model.to(device).eval(), pitch.to(device).eval(), device,
                        {name: version(name) for name in ("demucs", "torchcrepe", "torch", "scipy")})
    except Exception as exc:
        raise BackendLoadError("vocal_f0", "failed to load the pinned separator/pitch models") from exc


def _load_runtime(separator_checkpoint, cache_dir, local_files_only, device):
    try:
        import torch
        import torchcrepe
        import demucs  # noqa: F401
        import scipy  # noqa: F401
        if version("demucs") != "4.0.1" or version("torchcrepe") != "0.0.24":
            raise ImportError("requires demucs==4.0.1 and torchcrepe==0.0.24")
    except (ImportError, OSError) as exc:
        raise BackendDependencyError("vocal_f0", "install natural-features[vocal_f0]") from exc
    cache = Path(cache_dir).expanduser() if cache_dir else Path(torch.hub.get_dir()) / "checkpoints"
    path = Path(separator_checkpoint).expanduser() if separator_checkpoint else cache / SEPARATOR_FILE
    if not path.is_file():
        if separator_checkpoint or local_files_only:
            raise BackendLoadError("vocal_f0", f"pinned Demucs checkpoint is not available: {path}")
        cache.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                temporary = Path(stream.name)
                with urlopen(SEPARATOR_URL, timeout=60) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
            if _sha256(temporary) != SEPARATOR_SHA256:
                raise ValueError("downloaded separator SHA256 mismatch")
            temporary.replace(path)
        except Exception as exc:
            raise BackendLoadError("vocal_f0", "pinned separator download failed") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    weights = Path(torchcrepe.__file__).parent / "assets" / "full.pth"
    try:
        if not weights.is_file():
            raise FileNotFoundError("packaged CREPE weights are missing")
        if _sha256(path) != SEPARATOR_SHA256 or _sha256(weights) != CREPE_SHA256:
            raise BackendLoadError("vocal_f0", "model SHA256 mismatch; only the pinned assets are supported")
    except OSError as exc:
        raise BackendLoadError("vocal_f0", "model assets could not be inspected") from exc
    return _cached_runtime(str(path.resolve()), device)


def _frame_rms(audio, centers):
    # Same 1024-sample, centered zero-padding contract as CREPE.
    padded = np.pad(np.asarray(audio, dtype=np.float64) ** 2, (512, 512))
    total = np.concatenate([[0.0], np.cumsum(padded)])
    return np.sqrt(np.maximum(0, total[centers + 1024] - total[centers]) / 1024)


def audio_vocal_f0(
    stimulus: AudioStimulus, *, separator_checkpoint: str | None = None,
    cache_dir: str | None = None, local_files_only: bool = True, device: str = "cpu",
    hop_s: float = 0.01, fmin: float = 50.0, fmax: float = 1000.0,
    confidence_threshold: float = 0.21, source_threshold: float = 0.1,
    min_source_rms: float = 0.001,
) -> FeatureSeries:
    """Estimate dominant F0 of Demucs' selected vocal stem, with missingness.

    Supported F0 is NaN when periodicity, stem-energy support or amplitude is
    insufficient. ``confidence`` is CREPE periodicity; ``source_support`` is
    vocal-stem/sum-of-stem frame energy, not voice probability. Model,
    decoder and threshold choices must be qualified for the intended domain.
    This does not resolve individual singers in duets/choirs.
    """
    mono = mono_audio(stimulus)
    if not isinstance(local_files_only, bool) or not isinstance(device, str) or not device:
        raise ValueError("local_files_only must be boolean and device a non-empty string")
    if not np.isfinite(hop_s) or hop_s <= 0:
        raise ValueError("hop_s must be finite and positive")
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 32.71 <= fmin < fmax <= 2006):
        raise ValueError("require 32.71 <= fmin < fmax <= 2006")
    for name, threshold in (("confidence_threshold", confidence_threshold), ("source_threshold", source_threshold)):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not np.isfinite(min_source_rms) or min_source_rms <= 0:
        raise ValueError("min_source_rms must be finite and positive")
    runtime = _load_runtime(separator_checkpoint, cache_dir, local_files_only, device)
    x = np.asarray(stimulus.samples, dtype=np.float32)
    if x.ndim == 1:
        x = np.repeat(x[:, None], 2, axis=1)
    elif x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] != 2:
        raise ValueError("vocal separation accepts mono or stereo audio")
    separator_input = _resample(x, stimulus.sr_hz, 44100)
    pitch_mix = _resample(mono, stimulus.sr_hz, 16000)
    try:
        stems = runtime.separate(separator_input)
        if np.shape(stems) != (len(separator_input), len(runtime.sources)) or not np.all(np.isfinite(stems)):
            raise ValueError("separator returned invalid stems")
        pitch_stems = _resample(stems, 44100, 16000)[:len(pitch_mix)]
        pitch_input = pitch_stems[:, list(runtime.sources).index("vocals")]
        if len(pitch_input) != len(pitch_mix):
            raise ValueError("separator resampling returned too few samples")
        hop = max(1, round(16000 * hop_s))
        centers = np.arange(1 + len(pitch_mix) // hop) * hop
        f0, confidence = runtime.pitch(pitch_input, hop, fmin, fmax)
        if f0.shape != centers.shape or confidence.shape != centers.shape:
            raise ValueError("pitch output count disagrees with the native sample grid")
        if not np.all(np.isfinite(f0)) or np.any(f0 <= 0) or not np.all(np.isfinite(confidence)):
            raise ValueError("pitch backend returned nonfinite or nonpositive output")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("pitch confidence is outside [0, 1]")
    except Exception as exc:
        raise BackendInferenceError("vocal_f0", "source-specific pitch inference failed") from exc
    source_rms = _frame_rms(pitch_input, centers)
    total_energy = sum(_frame_rms(pitch_stems[:, j], centers) ** 2 for j in range(pitch_stems.shape[1]))
    support = source_rms ** 2 / np.maximum(total_energy, 1e-20)
    duration = len(mono) / stimulus.sr_hz
    source = (support >= source_threshold) & (source_rms >= min_source_rms) & (centers / 16000 < duration)
    voiced = source & (confidence >= confidence_threshold)
    edge = (1200 * np.log2(f0 / fmin) < 20) | (1200 * np.log2(fmax / f0) < 20)
    jump = np.zeros(len(f0), dtype=bool)
    intervals = np.abs(1200 * np.diff(np.log2(f0)))
    jump[1:] = voiced[1:] & voiced[:-1] & (np.abs(intervals - 1200) <= 100)
    padded = (centers < 512) | ((centers + 512) / 16000 > duration)
    values = np.column_stack([np.where(voiced, f0, np.nan), confidence, support, source, voiced, ~voiced, edge, jump, padded])
    start = stimulus.start_offset_s
    bounds = start + np.clip(np.column_stack([centers - 512, centers + 512]) / 16000, 0, duration)
    context = computation_context(
        stimulus, policy="bidirectional", bounds_s=[[start, start + duration]],
        preprocessing={"resampler": "scipy.signal.resample_poly; default Kaiser 5; constant zero boundary",
                       "original_sr_hz": stimulus.sr_hz, "separator_sr_hz": 44100, "pitch_sr_hz": 16000,
                       "original_samples": len(mono), "separator_samples": len(separator_input),
                       "pitch_samples": len(pitch_input), "mono_policy": "mono duplicated; stereo retained; selected stem mean",
                       "resampling_tail_s": len(pitch_mix) / 16000 - duration,
                       "separator_normalization": "whole-input mono mean/population-std; per-stem DC removed; shifts=0; split=True; overlap=0.25",
                       "separator_segment_s": 7.8,
                       "pitch_zero_padding_samples_each_side": 512},
    )
    md = extractor_metadata(
        "audio.vocal_f0", code_version="vocal-f0-v1",
        params={"hop_s": hop_s, "fmin": fmin, "fmax": fmax, "confidence_threshold": confidence_threshold,
                "source_threshold": source_threshold, "min_source_rms": min_source_rms},
        model_revision=f"demucs:sha256:{SEPARATOR_SHA256};crepe:sha256:{CREPE_SHA256}",
        extra={"backend": "demucs_htdemucs_crepe_full", "fallback_used": False, "execution_mode": "strict",
               "source_selection": "Demucs vocals; dominant periodic pitch within that stem; no singer identity",
               "confidence_semantics": "CREPE periodicity; not calibrated vocal probability",
               "source_support_semantics": "vocal/sum-of-separated-stems frame energy; heuristic, not probability",
               "missing_f0": "NaN; never encoded as acoustic silence or phonology",
               "octave_jump_semantics": "adjacent supported frames differ by 1200 +/- 100 cents; not proven octave error",
               "pitch_decoder": "local weighted +/-4 bins; undithered centers; no temporal smoothing",
               "effective_hop_s": hop / 16000, "frontend_support": "1024-sample CREPE window on resampled vocal stem",
               "computation_context": context, "qualification": "not implied by extraction; see benchmark report",
               "versions": runtime.versions},
    )
    return FeatureSeries(
        values=values.astype(np.float32), times_s=start + centers / 16000,
        coords={"feature": FEATURES}, metadata=md,
        timebase=TimebaseSpec(kind="audio_hop", reference=stimulus.clock, hop_s=hop / 16000, window_s=1024 / 16000),
        time_bounds_s=bounds, temporal_context=stimulus.temporal_context,
    )
