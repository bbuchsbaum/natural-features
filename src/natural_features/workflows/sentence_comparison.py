"""Experimental native acoustic blocks and explicit item-feature scaling.

RDM audit/inference policy lives in the companion example. No model dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.audio.lowlevel import mel, rms
from natural_features.features.audio import lowlevel, modulation, prosody
from natural_features.features.audio.modulation import (
    log_cochleagram,
    modulation_power_spectrum,
)
from natural_features.features.audio.prosody import audio_pitch
from natural_features.features.speech.phone_sequences import ItemFeatures
from natural_features.features.speech.pronunciation import digest, provenance


@dataclass(frozen=True)
class AcousticConfig:
    sample_rate: int = 16000
    fmin: float = 50.0
    fmax: float = 7500.0
    n_mels: int = 32
    frame_s: float = 0.025
    hop_s: float = 0.01
    modulation_window_s: float = 2.0
    modulation_hop_s: float = 0.5
    coch_hop_s: float = 0.005
    n_log_bins: int = 64
    spectral_edges: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    temporal_edges: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    include_pitch: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.sample_rate) is not int
            or self.sample_rate < 1000
            or not 0 < self.fmin < self.fmax <= self.sample_rate / 2
        ):
            raise ValueError("invalid shared sample rate or frequency limits")
        for n in (self.n_mels, self.n_log_bins):
            if type(n) is not int or n < 4:
                raise ValueError("filter and log-bin counts must be integers >= 4")
        for value in (
            self.frame_s,
            self.hop_s,
            self.modulation_window_s,
            self.modulation_hop_s,
            self.coch_hop_s,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("window and hop lengths must be positive and finite")
        if self.hop_s > self.frame_s or self.coch_hop_s > self.frame_s:
            raise ValueError("frame hops may not leave unsupported gaps")
        for edges in (self.spectral_edges, self.temporal_edges):
            if (
                len(edges) < 2
                or not np.isfinite(edges).all()
                or edges[0] <= 0
                or np.any(np.diff(edges) <= 0)
            ):
                raise ValueError(
                    "modulation edges must be finite, positive and increasing"
                )


@dataclass(frozen=True)
class NativeAcousticBlock:
    series: FeatureSeries
    valid: np.ndarray
    weights_s: np.ndarray
    coverage_s: float


@dataclass(frozen=True)
class SentenceAcoustics:
    native: dict[str, dict[str, NativeAcousticBlock]]
    blocks: dict[str, ItemFeatures]
    metadata: dict


def overlap_weights(bounds: np.ndarray, *, start: float, stop: float) -> np.ndarray:
    """Integrate 1 / active-window count over each clipped support interval.

    Weights sum to the union duration, not the sum of overlapping durations.
    This defines a duration-weighted average of window estimates, not a claim
    that windowed estimates equal instantaneous measurements.
    """
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] < bounds[:, 0])
    ):
        raise ValueError("bounds must be finite ordered intervals")
    if not np.isfinite([start, stop]).all() or stop < start:
        raise ValueError("invalid clip bounds")
    if not len(bounds):
        return np.zeros(0)
    clipped = np.clip(bounds, start, stop)
    edges, inverse = np.unique(clipped.ravel(), return_inverse=True)
    indices = inverse.reshape(-1, 2)
    changes = np.zeros(len(edges))
    np.add.at(changes, indices[:, 0], 1)
    np.add.at(changes, indices[:, 1], -1)
    counts = np.cumsum(changes)[:-1]
    increments = np.divide(
        np.diff(edges), counts, out=np.zeros_like(counts), where=counts > 0
    )
    integral = np.r_[0.0, np.cumsum(increments)]
    return integral[indices[:, 1]] - integral[indices[:, 0]]


def _native(
    stim: AudioStimulus,
    values: np.ndarray,
    names: Sequence[str],
    bounds: np.ndarray,
    md: dict,
    *,
    hop_s: float,
    window_s: float,
    valid: np.ndarray | None = None,
) -> NativeAcousticBlock:
    start = stim.start_offset_s
    stop = start + len(stim.samples) / stim.sr_hz
    series = FeatureSeries(
        values=values,
        times_s=bounds.mean(axis=1),
        coords={"feature": list(names)},
        metadata=md,
        time_bounds_s=bounds,
        temporal_context=stim.temporal_context,
        timebase=TimebaseSpec(
            kind="audio_hop",
            reference=stim.clock,
            hop_s=hop_s,
            sampling_rate_hz=1 / hop_s,
            window_s=window_s,
        ),
    )
    validity = (
        np.isfinite(values)
        if valid is None
        else np.asarray(valid, dtype=bool) & np.isfinite(values)
    )
    weights = overlap_weights(bounds, start=start, stop=stop)
    return NativeAcousticBlock(series, validity, weights, float(weights.sum()))


def _summary(block: NativeAcousticBlock) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(block.series.values, dtype=float)
    weights = block.weights_s[:, None] * block.valid
    total = weights.sum(axis=0)
    mean = np.divide(
        (np.where(block.valid, values, 0) * weights).sum(axis=0),
        total,
        out=np.full(values.shape[1], np.nan),
        where=total > 0,
    )
    residual = np.where(block.valid, values - mean, 0)
    variance = np.divide(
        (residual**2 * weights).sum(axis=0),
        total,
        out=np.full(values.shape[1], np.nan),
        where=total > 0,
    )
    return np.r_[mean, np.sqrt(variance)], np.tile(total > 0, 2)


def extract_sentence_acoustics(
    items: Mapping[str, AudioStimulus], *, config: AcousticConfig | None = None
) -> SentenceAcoustics:
    """Keep native data; include silence; mark padded/short/undefined data invalid.

    Spectral shape subtracts each log-mel frame's across-band mean. Raw log-mel
    summaries remain separately available. No speech detection or resampling.
    """
    config = AcousticConfig() if config is None else config
    if not items:
        raise ValueError("at least one audio item is required")
    native = {}
    rows: dict[str, list[np.ndarray]] = {}
    masks: dict[str, list[np.ndarray]] = {}
    coordinates: dict[str, tuple[str, ...]] = {}
    sources = {}
    sr = config.sample_rate
    hop = max(1, round(sr * config.hop_s))
    win = max(1, round(sr * config.frame_s))
    coch_hop = max(1, round(sr * config.coch_hop_s))
    w = max(1, round(config.modulation_window_s * sr / coch_hop))
    mh = max(1, round(config.modulation_hop_s * sr / coch_hop))
    if w < 4:
        raise ValueError(
            "modulation window must contain at least four cochleagram frames"
        )
    for item_id, stim in items.items():
        if stim.sr_hz != sr:
            raise ValueError(
                f"{item_id}: expected {sr} Hz; explicitly resample outside this workflow"
            )
        if not np.isfinite(stim.samples).all() or not np.isfinite(stim.start_offset_s):
            raise ValueError(f"{item_id}: audio samples and origin must be finite")
        x = np.asarray(stim.samples, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        mono = AudioStimulus(
            x,
            sr,
            start_offset_s=stim.start_offset_s,
            clock=stim.clock,
            temporal_context=stim.temporal_context,
        )
        sample_hash = hashlib.sha256(x.astype("<f4").tobytes()).hexdigest()
        sources[item_id] = {
            "mono_samples_sha256": sample_hash,
            "sample_count": len(x),
            "sample_rate": sr,
            "input_samples_sha256": hashlib.sha256(
                np.asarray(stim.samples, dtype="<f4").tobytes()
            ).hexdigest(),
            "input_shape": list(stim.samples.shape),
            "clock": str(stim.clock),
            "start_offset_s": stim.start_offset_s,
        }
        source_path = Path(stim.source) if stim.source else None
        if source_path and source_path.is_file():
            sources[item_id]["audio_file_sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
        nonzero = bool(np.any(x != 0))
        mel_series = mel(
            mono,
            hop_s=hop / sr,
            win_s=win / sr,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
        )
        rms_series = rms(mono, hop_s=hop / sr, win_s=win / sr)
        starts = np.arange(len(mel_series.values)) * hop
        bounds = stim.start_offset_s + np.column_stack((starts, starts + win)) / sr
        frame_ok = (starts + win <= len(x)) & nonzero
        entries = {}
        for name, fs, vals in (
            ("spectral_raw", mel_series, mel_series.values),
            (
                "spectral_shape",
                mel_series,
                mel_series.values - mel_series.values.mean(axis=1, keepdims=True),
            ),
            ("envelope", rms_series, rms_series.values),
        ):
            md = {
                **fs.metadata,
                "aggregation_input": name,
                "transform": "subtract_frame_band_mean"
                if name == "spectral_shape"
                else "none",
                "effective_hop_s": hop / sr,
                "effective_window_s": win / sr,
                "right_padding_samples": max(0, win - len(x)),
                "source": sources[item_id],
            }
            if name == "spectral_shape":
                md = provenance(
                    "audio.sentence.spectral_shape",
                    {
                        **fs.metadata.get("params", {}),
                        "transform": "subtract_frame_band_mean",
                        "config": asdict(config),
                    },
                    source_metadata=md,
                )
            ok = starts + win <= len(x) if name == "envelope" else frame_ok
            entries[name] = _native(
                stim,
                vals,
                fs.coords["feature"],
                bounds,
                md,
                hop_s=hop / sr,
                window_s=win / sr,
                valid=np.broadcast_to(ok[:, None], vals.shape),
            )
        coch, octaves = log_cochleagram(
            mono,
            hop_s=coch_hop / sr,
            win_s=win / sr,
            n_channels=64,
            n_log_bins=config.n_log_bins,
            fmin=config.fmin,
            fmax=config.fmax,
        )
        _, mod_names = modulation_power_spectrum(
            np.zeros((w, config.n_log_bins)),
            oct_per_bin=octaves,
            frame_rate_hz=sr / coch_hop,
            spec_edges=np.array(config.spectral_edges),
            temp_edges=np.array(config.temporal_edges),
        )
        mod_starts = np.arange(0, max(0, len(coch) - w + 1), mh)
        # Require real sample support, including the final STFT analysis window.
        mod_starts = mod_starts[(mod_starts + w - 1) * coch_hop + win <= len(x)]
        mod_values = np.array(
            [
                modulation_power_spectrum(
                    coch[s : s + w],
                    oct_per_bin=octaves,
                    frame_rate_hz=sr / coch_hop,
                    spec_edges=np.array(config.spectral_edges),
                    temp_edges=np.array(config.temporal_edges),
                )[0]
                for s in mod_starts
            ]
        ).reshape(-1, len(mod_names))
        mod_bounds = (
            stim.start_offset_s
            + np.column_stack(
                (mod_starts * coch_hop, (mod_starts + w - 1) * coch_hop + win)
            )
            / sr
        )
        # Fixed two-cycle support policy; missing frequencies are not padded in.
        supported = np.array(
            [
                t * w * coch_hop / sr >= 2
                and config.temporal_edges[i + 1] <= sr / coch_hop / 2
                and config.spectral_edges[j + 1] <= 0.5 / octaves
                for j in range(len(config.spectral_edges) - 1)
                for _ in range(2)
                for i, t in enumerate(config.temporal_edges[:-1])
            ]
        )
        mod_md = provenance(
            "audio.modulation.mps.sentence",
            asdict(config),
            source=sources[item_id],
            backend="ERB_triangular_STFT_logfrequency_2DFFT",
            spectral_units="cycles_per_octave",
            temporal_units="Hz",
            minimum_cycles=2,
            effective_fft_duration_s=w * coch_hop / sr,
            short_clip_policy="unavailable_no_padding",
            silence_policy="all_zero_unavailable",
        )
        entries["modulation"] = _native(
            stim,
            mod_values,
            mod_names,
            mod_bounds,
            mod_md,
            hop_s=mh * coch_hop / sr,
            window_s=((w - 1) * coch_hop + win) / sr,
            valid=np.broadcast_to(supported & nonzero, mod_values.shape),
        )
        if config.include_pitch:
            pitch = audio_pitch(mono, hop_s=hop / sr)
            complete = pitch.time_bounds_s[:, 1] <= stim.start_offset_s + len(x) / sr
            voiced = (pitch.values[:, 0] > 0) & complete
            entries["pitch"] = _native(
                stim,
                pitch.values,
                pitch.coords["feature"],
                pitch.time_bounds_s,
                {**pitch.metadata, "source": sources[item_id]},
                hop_s=hop / sr,
                window_s=pitch.metadata["effective_window_s"],
                valid=np.column_stack((voiced, complete)),
            )
        native[item_id] = entries
        for name, block in entries.items():
            vals, valid = _summary(block)
            rows.setdefault(name, []).append(vals)
            masks.setdefault(name, []).append(valid)
            coordinates[name] = tuple(
                f"{n}.{stat}"
                for stat in ("mean", "sd")
                for n in block.series.coords["feature"]
            )
        controls = [len(x) / sr, float(np.sqrt(np.mean(x.astype(float) ** 2)))]
        control_names = ["duration_s", "whole_clip_rms"]
        if config.include_pitch:
            pitch_block = entries["pitch"]
            coverage = np.sum(pitch_block.weights_s * pitch_block.valid[:, 1])
            controls.append(
                float(
                    np.sum(pitch_block.weights_s * pitch_block.valid[:, 0]) / coverage
                )
                if coverage
                else np.nan
            )
            control_names.append("voiced_duration_fraction")
        rows.setdefault("controls", []).append(np.array(controls))
        masks.setdefault("controls", []).append(np.isfinite(controls))
        coordinates["controls"] = tuple(control_names)
    md = provenance(
        "audio.sentence_blocks",
        asdict(config),
        sources=sources,
        silence="included_no_VAD",
        channel_mix="arithmetic_mean",
        resampling="none",
        weighting="integral_inverse_overlap_count",
        standard_deviation="population",
        native_coverage={
            key: {name: b.coverage_s for name, b in blocks.items()}
            for key, blocks in native.items()
        },
        valid_weight_s={
            key: {
                name: (b.weights_s[:, None] * b.valid).sum(axis=0).tolist()
                for name, b in blocks.items()
            }
            for key, blocks in native.items()
        },
    )
    md["implementation_sha256"][Path(__file__).name] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    for module in (lowlevel, modulation, prosody):
        md["implementation_sha256"][module.__name__] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    blocks = {
        name: ItemFeatures(
            tuple(items),
            coordinates[name],
            np.array(values),
            np.array(masks[name]),
            {**md, "block": name},
        )
        for name, values in rows.items()
    }
    return SentenceAcoustics(native, blocks, md)


@dataclass(frozen=True)
class FeatureScaler:
    names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    retained: tuple[int, ...]
    reference_ids: tuple[str, ...]
    reference_sha256: str
    feature_schema_sha256: str

    @staticmethod
    def schema_digest(features: ItemFeatures) -> str:
        return digest(
            {
                "schema": features.schema,
                "names": features.names,
                **{
                    k: features.metadata.get(k)
                    for k in ("extractor_name", "block", "params", "model", "revision")
                },
            }
        )

    @classmethod
    def fit(
        cls, features: ItemFeatures, *, reference_ids: Sequence[str]
    ) -> FeatureScaler:
        if len(reference_ids) < 2 or len(set(reference_ids)) != len(reference_ids):
            raise ValueError("at least two distinct scaling reference IDs are required")
        try:
            ix = [features.item_ids.index(i) for i in reference_ids]
        except ValueError as exc:
            raise ValueError("unknown scaling reference ID") from exc
        data = features.values[ix]
        available = features.valid[ix].all(axis=0)
        data = np.where(available, data, 0.0)
        mean, scale = data.mean(axis=0), data.std(axis=0)
        keep = available & (
            scale > np.finfo(float).eps * np.maximum(1.0, np.abs(data).max(axis=0))
        )
        return cls(
            features.names,
            tuple(mean),
            tuple(np.where(keep, scale, 1)),
            tuple(np.flatnonzero(keep).tolist()),
            tuple(reference_ids),
            digest(
                {
                    "ids": list(reference_ids),
                    "values": data.tolist(),
                    "available": available.tolist(),
                }
            ),
            cls.schema_digest(features),
        )

    def transform(self, features: ItemFeatures) -> ItemFeatures:
        if (
            features.names != self.names
            or self.schema_digest(features) != self.feature_schema_sha256
        ):
            raise ValueError("scaler feature schema mismatch")
        values = (features.values - np.array(self.mean)) / np.array(self.scale)
        indices = list(self.retained)
        return ItemFeatures(
            features.item_ids,
            tuple(self.names[i] for i in indices),
            values[:, indices],
            features.valid[:, indices],
            {"source": features.metadata, "scaler": self.to_dict()},
        )

    def to_dict(self) -> dict:
        return {
            "schema": "FeatureScaler/v1",
            **asdict(self),
            "ddof": 0,
            "excluded_policy": "invalid_on_any_reference_or_numerically_constant",
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeatureScaler:
        if data.get("schema") != "FeatureScaler/v1" or data.get("ddof") != 0:
            raise ValueError("unsupported scaler schema")
        obj = cls(
            *(
                tuple(data[k])
                for k in ("names", "mean", "scale", "retained", "reference_ids")
            ),
            data["reference_sha256"],
            data["feature_schema_sha256"],
        )
        if (
            len(obj.names) != len(obj.mean)
            or len(obj.names) != len(obj.scale)
            or not np.isfinite(obj.mean).all()
            or not np.isfinite(obj.scale).all()
            or any(s <= 0 for s in obj.scale)
        ):
            raise ValueError("invalid scaler coefficients")
        if len(set(obj.retained)) != len(obj.retained) or any(
            type(i) is not int or not 0 <= i < len(obj.names) for i in obj.retained
        ):
            raise ValueError("invalid retained scaler coordinates")
        return obj


def combine_scaled_blocks(
    blocks: Mapping[str, ItemFeatures], *, weights: Mapping[str, float]
) -> ItemFeatures:
    """Explicit sqrt(weight / dimension) scaling; Euclidean distance balances blocks."""
    if (
        not blocks
        or set(blocks) != set(weights)
        or any(not np.isfinite(w) or w <= 0 for w in weights.values())
    ):
        raise ValueError("every block requires an explicit finite positive weight")
    first = next(iter(blocks.values()))
    if any(b.item_ids != first.item_ids or not b.names for b in blocks.values()):
        raise ValueError(
            "blocks must have identical item order and nonempty coordinates"
        )
    if any("scaler" not in b.metadata for b in blocks.values()):
        raise ValueError("fit and apply a scaler to each block before combination")
    values = np.concatenate(
        [
            b.values * np.sqrt(weights[name] / (len(b.names) * sum(weights.values())))
            for name, b in blocks.items()
        ],
        axis=1,
    )
    return ItemFeatures(
        first.item_ids,
        tuple(f"{name}:{c}" for name, b in blocks.items() for c in b.names),
        values,
        np.concatenate([b.valid for b in blocks.values()], axis=1),
        {
            "weights": dict(weights),
            "normalization": "sqrt(weight / (dimension * sum_weights))",
            "blocks": {n: b.metadata for n, b in blocks.items()},
        },
    )


def feature_distances(features: ItemFeatures, *, metric: str = "cosine") -> np.ndarray:
    if metric not in {"cosine", "euclidean", "rms_euclidean"}:
        raise ValueError("unsupported feature distance")
    result = np.full((len(features.item_ids), len(features.item_ids)), np.nan)
    eligible = features.eligible.copy()
    if metric == "cosine":
        eligible &= np.linalg.norm(features.values, axis=1) > 0
    for i in np.flatnonzero(eligible):
        for j in np.flatnonzero(eligible):
            a, b = features.values[i], features.values[j]
            if metric == "cosine":
                result[i, j] = np.clip(
                    1 - np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), 0, 2
                )
            else:
                result[i, j] = np.linalg.norm(a - b) / (
                    np.sqrt(len(a)) if metric == "rms_euclidean" else 1
                )
    np.fill_diagonal(result, np.where(eligible, 0.0, np.nan))
    return result
