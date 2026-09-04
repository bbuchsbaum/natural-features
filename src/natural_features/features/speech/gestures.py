"""Canonical overlapping gesture activations and articulatory dynamics."""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.phonology import (
    _features_for_label,
    _normalize_phone_label,
)

GESTURE_CHANNELS = ["lips", "tongue_tip", "tongue_body", "velum", "glottis"]

_PHONE_GESTURES: dict[str, dict[str, float]] = {
    "P": {"lips": 1.0},
    "B": {"lips": 1.0, "glottis": 1.0},
    "M": {"lips": 1.0, "velum": 1.0, "glottis": 1.0},
    "F": {"lips": 0.8},
    "V": {"lips": 0.8, "glottis": 1.0},
    "W": {"lips": 0.9, "tongue_body": 0.4, "glottis": 1.0},
    "T": {"tongue_tip": 1.0},
    "D": {"tongue_tip": 1.0, "glottis": 1.0},
    "N": {"tongue_tip": 1.0, "velum": 1.0, "glottis": 1.0},
    "S": {"tongue_tip": 0.7},
    "Z": {"tongue_tip": 0.7, "glottis": 1.0},
    "TH": {"tongue_tip": 0.8},
    "DH": {"tongue_tip": 0.8, "glottis": 1.0},
    "L": {"tongue_tip": 0.9, "glottis": 1.0},
    "R": {"tongue_tip": 0.6, "tongue_body": 0.4, "glottis": 1.0},
    "CH": {"tongue_tip": 0.9},
    "JH": {"tongue_tip": 0.9, "glottis": 1.0},
    "SH": {"tongue_tip": 0.7, "tongue_body": 0.3},
    "ZH": {"tongue_tip": 0.7, "tongue_body": 0.3, "glottis": 1.0},
    "Y": {"tongue_body": 0.6, "glottis": 1.0},
    "K": {"tongue_body": 1.0},
    "G": {"tongue_body": 1.0, "glottis": 1.0},
    "NG": {"tongue_body": 1.0, "velum": 1.0, "glottis": 1.0},
    "HH": {"glottis": 0.3},
    "IY": {"tongue_body": 0.7, "glottis": 1.0},
    "IH": {"tongue_body": 0.6, "glottis": 1.0},
    "EH": {"tongue_body": 0.5, "glottis": 1.0},
    "AE": {"tongue_body": 0.4, "glottis": 1.0},
    "EY": {"tongue_body": 0.55, "glottis": 1.0},
    "AY": {"tongue_body": 0.5, "glottis": 1.0},
    "AH": {"tongue_body": 0.4, "glottis": 1.0},
    "ER": {"tongue_body": 0.45, "tongue_tip": 0.3, "glottis": 1.0},
    "AA": {"tongue_body": 0.35, "glottis": 1.0},
    "AO": {"tongue_body": 0.4, "lips": 0.4, "glottis": 1.0},
    "OW": {"tongue_body": 0.45, "lips": 0.5, "glottis": 1.0},
    "AW": {"tongue_body": 0.4, "lips": 0.4, "glottis": 1.0},
    "OY": {"tongue_body": 0.5, "lips": 0.4, "glottis": 1.0},
    "UH": {"tongue_body": 0.5, "lips": 0.5, "glottis": 1.0},
    "UW": {"tongue_body": 0.55, "lips": 0.6, "glottis": 1.0},
}


def _raised_cosine(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float32)
    return (
        0.5 * (1.0 - np.cos(np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float64)))
    ).astype(np.float32)


def articulatory_gestures(
    phones: EventSeries,
    *,
    hop_s: float = 0.01,
    overlap_s: float = 0.03,
    duration_s: float | None = None,
) -> FeatureSeries:
    """Overlapping raised-cosine activations from phone labels.

    This is an interpretable control layer, not waveform-conditioned inversion.
    """

    if len(phones) == 0:
        n = 1
        start = 0.0
    else:
        start = max(0.0, float(phones.onset_s[0]) - overlap_s)
        stop = float(phones.offset_s[-1]) + overlap_s
        if duration_s is not None:
            stop = max(stop, float(duration_s))
        n = max(1, int(np.round((stop - start) / hop_s)))
    values = np.zeros((n, len(GESTURE_CHANNELS)), dtype=np.float32)
    times = start + (np.arange(n, dtype=np.float64) + 0.5) * hop_s
    labels = phones.label if phones.label is not None else np.array([], dtype=object)
    ix = {name: i for i, name in enumerate(GESTURE_CHANNELS)}
    for i in range(len(phones)):
        lab = _normalize_phone_label(str(labels[i]) if i < len(labels) else "")
        targets = dict(_PHONE_GESTURES.get(lab, {}))
        feats = _features_for_label(lab)
        if "voiced" in feats:
            targets.setdefault("glottis", 1.0)
        if "nasal" in feats:
            targets.setdefault("velum", 1.0)
        on = float(phones.onset_s[i]) - overlap_s
        off = float(phones.offset_s[i]) + overlap_s
        mask = (times >= on) & (times < off)
        if not np.any(mask):
            continue
        window = _raised_cosine(int(np.sum(mask)))
        for name, strength in targets.items():
            col = ix.get(name)
            if col is None:
                continue
            values[mask, col] = np.maximum(
                values[mask, col], window * np.float32(strength)
            )
    md = extractor_metadata(
        "speech.articulatory.gestures",
        params={"hop_s": hop_s, "overlap_s": overlap_s},
        extra={"backend": "canonical_raised_cosine"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": list(GESTURE_CHANNELS)},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )


def articulatory_dynamics(
    articulatory: FeatureSeries,
    *,
    speed_onset_quantile: float = 0.7,
) -> FeatureSeries:
    """Velocity, acceleration, speed, and speed onset/offset of a G series."""

    if articulatory.values.ndim != 2:
        raise ValueError("articulatory must be a 2-D FeatureSeries")
    g = np.asarray(articulatory.values, dtype=np.float64)
    times = np.asarray(articulatory.times_s, dtype=np.float64)
    vel = np.gradient(g, times, axis=0)
    acc = np.gradient(vel, times, axis=0)
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    thresh = float(np.quantile(speed, speed_onset_quantile)) if speed.size else 0.0
    onset = np.zeros((g.shape[0], 1), dtype=np.float32)
    offset = np.zeros((g.shape[0], 1), dtype=np.float32)
    above = speed[:, 0] > thresh
    onset[1:, 0] = np.logical_and(above[1:], np.logical_not(above[:-1])).astype(
        np.float32
    )
    offset[1:, 0] = np.logical_and(np.logical_not(above[1:]), above[:-1]).astype(
        np.float32
    )
    base_names = [
        str(x)
        for x in articulatory.coords.get(
            "feature", [f"g{i}" for i in range(g.shape[1])]
        )
    ]
    names = (
        [f"vel_{n}" for n in base_names]
        + [f"acc_{n}" for n in base_names]
        + [
            "speed",
            "gesture_onset",
            "gesture_offset",
        ]
    )
    values = np.concatenate(
        [
            vel.astype(np.float32),
            acc.astype(np.float32),
            speed.astype(np.float32),
            onset,
            offset,
        ],
        axis=1,
    )
    hop = None
    if len(times) > 1:
        hop = float(np.median(np.diff(times)))
    md = extractor_metadata(
        "speech.articulatory.dynamics",
        params={"speed_onset_quantile": speed_onset_quantile},
        extra={
            "backend": "finite_difference",
            "source_extractor": articulatory.metadata.get("extractor_name", "unknown"),
        },
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": names},
        metadata=md,
        timebase=articulatory.timebase
        if hop is None
        else TimebaseSpec(
            kind="audio_hop", hop_s=hop, sampling_rate_hz=1.0 / max(hop, 1e-6)
        ),
    )
