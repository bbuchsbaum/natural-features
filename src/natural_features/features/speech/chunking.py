"""Chunk planning and stitching utilities for long-audio speech pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from natural_features.core.feature_types import EventSeries
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.contracts import normalize_alignment_qc


@dataclass(frozen=True)
class AudioChunk:
    index: int
    sample_start: int
    sample_end: int
    start_s: float
    end_s: float


def plan_audio_chunks(
    *,
    n_samples: int,
    sr_hz: int,
    window_s: float,
    overlap_s: float = 0.0,
    start_offset_s: float = 0.0,
) -> list[AudioChunk]:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if sr_hz <= 0:
        raise ValueError("sr_hz must be > 0")
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    if overlap_s < 0:
        raise ValueError("overlap_s must be >= 0")
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be < window_s")

    win = max(1, int(round(window_s * sr_hz)))
    overlap = int(round(overlap_s * sr_hz))
    hop = max(1, win - overlap)
    chunks: list[AudioChunk] = []
    i = 0
    idx = 0
    while i < n_samples:
        j = min(i + win, n_samples)
        start_s = float(start_offset_s + (i / sr_hz))
        end_s = float(start_offset_s + (j / sr_hz))
        chunks.append(
            AudioChunk(
                index=idx,
                sample_start=i,
                sample_end=j,
                start_s=start_s,
                end_s=end_s,
            )
        )
        if j >= n_samples:
            break
        i += hop
        idx += 1
    return chunks


def stitch_word_events(
    parts: list[EventSeries],
    *,
    dedupe_tolerance_s: float = 0.03,
) -> tuple[EventSeries, int]:
    if not parts:
        md = extractor_metadata("speech.asr.chunk_stitch", params={"dedupe_tolerance_s": dedupe_tolerance_s})
        return EventSeries(
            onset_s=np.array([], dtype=np.float64),
            offset_s=np.array([], dtype=np.float64),
            label=np.array([], dtype=object),
            confidence=np.array([], dtype=np.float32),
            metadata=md,
        ), 0

    on = np.concatenate([p.onset_s for p in parts]).astype(np.float64)
    off = np.concatenate([p.offset_s for p in parts]).astype(np.float64)
    labels = np.concatenate([p.label if p.label is not None else np.array([""] * len(p), dtype=object) for p in parts]).astype(object)
    conf = np.concatenate(
        [p.confidence if p.confidence is not None else np.ones(len(p), dtype=np.float32) for p in parts]
    ).astype(np.float32)

    order = np.argsort(on, kind="mergesort")
    on = on[order]
    off = off[order]
    labels = labels[order]
    conf = conf[order]

    keep = np.ones(len(on), dtype=bool)
    stitch_conflicts = 0
    for i in range(1, len(on)):
        if not keep[i - 1]:
            continue
        same_label = str(labels[i]) == str(labels[i - 1])
        near_same_start = abs(float(on[i] - on[i - 1])) <= dedupe_tolerance_s
        overlap = float(on[i]) <= float(off[i - 1])
        if same_label and (near_same_start or overlap):
            # Keep the higher-confidence representative.
            if float(conf[i]) > float(conf[i - 1]):
                keep[i - 1] = False
            else:
                keep[i] = False
            stitch_conflicts += 1

    on = on[keep]
    off = np.maximum(off[keep], on)
    labels = labels[keep]
    conf = conf[keep]

    md = dict(parts[-1].metadata) if parts else extractor_metadata("speech.asr.chunk_stitch", params={})
    md["stitch_conflicts"] = int(stitch_conflicts)
    out = EventSeries(
        onset_s=on,
        offset_s=off,
        label=labels,
        confidence=conf,
        metadata=md,
    )
    return out, int(stitch_conflicts)


def aggregate_chunk_qc(
    per_chunk: list[dict[str, Any]],
    *,
    chunk_count: int,
    stitch_conflicts: int,
) -> dict[str, Any]:
    n_words = int(sum(int(q.get("n_words", 0)) for q in per_chunk))
    dropped = int(sum(int(q.get("dropped_words", 0)) for q in per_chunk))
    low = int(sum(int(q.get("low_confidence_words", 0)) for q in per_chunk))
    coverage = [float(q.get("coverage_fraction", 0.0)) for q in per_chunk if "coverage_fraction" in q]
    qc = normalize_alignment_qc(
        {
            "mode": "chunked",
            "fallback_used": bool(any(bool(q.get("fallback_used", False)) for q in per_chunk)),
            "n_words": n_words,
            "dropped_words": dropped,
            "low_confidence_words": low,
            "chunk_count": int(chunk_count),
            "stitch_conflicts": int(stitch_conflicts),
            "coverage_fraction": float(np.mean(coverage)) if coverage else 0.0,
        }
    )
    return qc
