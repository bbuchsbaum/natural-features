"""Bounded, dependency-light descriptors of musical structure.

These descriptors deliberately report estimates and explicit missingness; they do
not assign chord labels or infer musical sections.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio._music_contract import (
    computation_context,
    mono_audio,
    music_series,
    positive_seconds,
)
from natural_features.features.audio.music import PITCH_CLASSES, _chroma_filterbank
from natural_features.util.hashing import stable_hash

__all__ = ["music_chord_profiles", "music_tonal_novelty", "music_sequence_recurrence"]

_EPS = 1e-12


def _finite_positive(value: float, name: str) -> float:
    return positive_seconds(value, name)


def _frame_chroma(
    stimulus: AudioStimulus,
    *,
    hop_s: float,
    win_s: float,
    activity_rms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Full-support chroma frames and their physical support.

    This frontend intentionally does not use ``_chroma_matrix``: that helper's
    timestamps are based on requested rather than realized integer hop lengths.
    """
    hop_s = _finite_positive(hop_s, "hop_s")
    win_s = _finite_positive(win_s, "win_s")
    activity_rms = _finite_positive(activity_rms, "activity_rms")
    x = mono_audio(stimulus)
    sr = int(stimulus.sr_hz)
    h = max(1, int(round(hop_s * sr)))
    w = max(1, int(round(win_s * sr)))
    if len(x) < w:
        empty = np.zeros((0, 12), dtype=np.float32)
        return empty, np.zeros(0, bool), np.zeros(0), np.zeros((0, 2)), h, w
    starts = np.arange(0, len(x) - w + 1, h, dtype=np.int64)
    frames = x[starts[:, None] + np.arange(w)[None, :]].astype(np.float64)
    # Activity is deliberately measured on the unwindowed input waveform.
    active = np.sqrt(np.mean(frames * frames, axis=1)) > activity_rms
    window = np.hanning(w + 1)[:-1]
    spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1)) ** 2
    freqs = np.fft.rfftfreq(w, 1.0 / sr)
    chroma = spectrum @ _chroma_filterbank(freqs).T
    norm = np.linalg.norm(chroma, axis=1)
    # Activity is governed by waveform RMS; a finite nonzero chroma response is
    # sufficient regardless of its scale, preserving gain invariance away from
    # the declared waveform threshold.
    active &= np.isfinite(norm) & (norm > 0.0)
    chroma[active] /= norm[active, None]
    chroma[~active] = 0.0
    origin = float(stimulus.start_offset_s)
    bounds = origin + np.column_stack((starts, starts + w)) / sr
    times = origin + (starts + w / 2.0) / sr
    return chroma.astype(np.float32), active, times, bounds, h, w


def _triad_templates() -> tuple[np.ndarray, list[str]]:
    templates = []
    names = []
    for root, name in enumerate(PITCH_CLASSES):
        for suffix, intervals in (("major", (0, 4, 7)), ("minor", (0, 3, 7))):
            row = np.zeros(12, dtype=np.float64)
            row[(root + np.asarray(intervals)) % 12] = 1.0
            templates.append(row / np.linalg.norm(row))
            names.append(f"{name}_{suffix}")
    return np.asarray(templates), names


def _chord_profile_scores(chroma: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Score already-normalized synthetic or frontend chroma against triads."""
    templates, names = _triad_templates()
    return np.asarray(chroma, dtype=np.float64) @ templates.T, names


def _cosine_novelty(left: np.ndarray, right: np.ndarray) -> float:
    """Float64 cosine distance for two chroma blocks (also testable in isolation)."""
    a = np.asarray(left, dtype=np.float64).mean(axis=0)
    b = np.asarray(right, dtype=np.float64).mean(axis=0)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an == 0.0 or bn == 0.0:
        return 0.0
    return 1.0 - float(np.clip(np.dot(a / an, b / bn), 0.0, 1.0))


def _delay_stack_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    """Cosine similarity of ordered, normalized delay stacks."""
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    c = np.asarray(candidate, dtype=np.float64).reshape(-1)
    qn, cn = np.linalg.norm(q), np.linalg.norm(c)
    return 0.0 if qn == 0.0 or cn == 0.0 else float(np.dot(q, c) / (qn * cn))


def _series(
    stimulus: AudioStimulus, values: np.ndarray, times: np.ndarray, bounds: np.ndarray,
    *, extractor: str, params: dict, names: list[str], policy: str, dependencies: np.ndarray,
    hop_s: float, extra: dict | None = None,
):
    context = computation_context(
        stimulus, policy=policy, bounds_s=dependencies,
        preprocessing={"frontend": "full_support_chroma", "activity": "unwindowed_rms"},
    )
    return music_series(
        stimulus, values=values, times_s=times, bounds_s=bounds, names=names,
        extractor=extractor, params=params, context=context, hop_s=hop_s,
        extra={"requires_validity": True, **(extra or {})},
        alignment="offset" if policy == "past_only" else "center",
    )


def _output_pair_id(stimulus: AudioStimulus, family: str, params: dict) -> str:
    return stable_hash({
        "samples": np.asarray(stimulus.samples), "sr_hz": stimulus.sr_hz,
        "clock": str(stimulus.clock), "start_offset_s": stimulus.start_offset_s,
        "extractor_family": family, "params": params,
    }, length=24)


def music_chord_profiles(
    stimulus: AudioStimulus, *, hop_s: float = 0.1, win_s: float = 0.25,
    activity_rms: float = 1e-6,
) -> dict[str, object]:
    """Raw cosine similarities to 24 normalized major/minor triad templates."""
    chroma, active, times, bounds, h, _w = _frame_chroma(
        stimulus, hop_s=hop_s, win_s=win_s, activity_rms=activity_rms
    )
    raw_scores, names = _chord_profile_scores(chroma)
    scores = raw_scores.astype(np.float32)
    scores[~active] = 0.0
    normalized = scores / np.maximum(scores.sum(axis=1, keepdims=True), _EPS)
    entropy = -(normalized * np.log(np.maximum(normalized, _EPS))).sum(axis=1) / np.log(24.0)
    ordered = np.sort(scores, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if len(scores) else np.zeros(0)
    diag = np.column_stack((active, active, entropy * active, margin * active)).astype(np.float32)
    params = {"hop_s": hop_s, "win_s": win_s, "activity_rms": activity_rms}
    realized = h / float(stimulus.sr_hz)
    pair_id = _output_pair_id(stimulus, "audio.music.chord_profiles", params)
    return {
        "default": _series(stimulus, scores, times, bounds, extractor="audio.music.chord_profiles",
                           params=params, names=names, policy="local", dependencies=bounds,
                           hop_s=realized, extra={"validity_output": "diagnostics", "validity_column": "valid",
                                                   "output_pair_id": pair_id}),
        "diagnostics": _series(stimulus, diag, times, bounds, extractor="audio.music.chord_profiles.diagnostics",
                               params=params, names=["valid", "activity", "score_entropy", "top_two_margin"],
                               policy="local", dependencies=bounds, hop_s=realized,
                               extra={"output_pair_id": pair_id}),
    }


def music_tonal_novelty(
    stimulus: AudioStimulus, *, hop_s: float = 0.1, win_s: float = 0.25,
    window_s: float = 1.0, mode: Literal["centered", "past_only"] = "centered",
    activity_rms: float = 1e-6,
) -> dict[str, object]:
    """Cosine distance between adjacent bounded chroma windows."""
    if mode not in {"centered", "past_only"}:
        raise ValueError("mode must be 'centered' or 'past_only'")
    window_s = _finite_positive(window_s, "window_s")
    chroma, active, centers, frame_bounds, h, _w = _frame_chroma(
        stimulus, hop_s=hop_s, win_s=win_s, activity_rms=activity_rms
    )
    n = len(chroma)
    block = max(1, int(round(window_s * stimulus.sr_hz / h)))
    values = np.zeros((n, 1), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    # Centered output uses split s for [s-block,s) and [s,s+block). Causal
    # output instead ends each row at its current last frame i, so appending
    # audio cannot revise an earlier placeholder, value, or diagnostic.
    split = np.arange(n, dtype=int)
    if mode == "centered":
        output_times = float(stimulus.start_offset_s) + ((split - 0.5) * h + _w / 2.0) / stimulus.sr_hz
    else:
        output_times = frame_bounds[:, 1].copy()
    dep = np.zeros((n, 2), dtype=np.float64)
    out_bounds = np.zeros((n, 2), dtype=np.float64)
    left_fraction = np.zeros(n, dtype=np.float32)
    right_fraction = np.zeros(n, dtype=np.float32)
    for i, boundary in enumerate(split):
        if mode == "centered":
            left_start, left_end = boundary - block, boundary
            right_start, right_end = boundary, boundary + block
            # A split is an instantaneous local observation. Its full source
            # union is retained in computation_context below.
            out_bounds[i] = [output_times[i], output_times[i]]
        else:
            left_start, left_end = i - 2 * block + 1, i - block + 1
            right_start, right_end = i - block + 1, i + 1
            out_bounds[i] = frame_bounds[i]
        if left_start < 0 or right_end > n:
            dep[i] = frame_bounds[i]
            continue
        dep[i] = [frame_bounds[left_start, 0], frame_bounds[right_end - 1, 1]]
        left_fraction[i] = active[left_start:left_end].mean()
        right_fraction[i] = active[right_start:right_end].mean()
        if left_fraction[i] != 1.0 or right_fraction[i] != 1.0:
            continue
        values[i, 0] = _cosine_novelty(chroma[left_start:left_end], chroma[right_start:right_end])
        valid[i] = True
    policy = "past_only" if mode == "past_only" else "bidirectional"
    params = {"hop_s": hop_s, "win_s": win_s, "window_s": window_s, "mode": mode,
              "activity_rms": activity_rms, "window_frames": block}
    realized = h / float(stimulus.sr_hz)
    pair_id = _output_pair_id(stimulus, "audio.music.tonal_novelty", params)
    diagnostics = np.column_stack((valid, left_fraction, right_fraction,
                                   np.full(n, block))).astype(np.float32)
    return {
        "default": _series(stimulus, values, output_times, out_bounds, extractor="audio.music.tonal_novelty",
                           params=params, names=["tonal_novelty"], policy=policy, dependencies=dep,
                           hop_s=realized, extra={"validity_output": "diagnostics", "validity_column": "valid",
                                                   "output_pair_id": pair_id}),
        "diagnostics": _series(stimulus, diagnostics, output_times, out_bounds,
                               extractor="audio.music.tonal_novelty.diagnostics", params=params,
                               names=["valid", "left_activity_fraction", "right_activity_fraction", "window_frames"],
                               policy=policy, dependencies=dep, hop_s=realized,
                               extra={"output_pair_id": pair_id}),
    }


def music_sequence_recurrence(
    stimulus: AudioStimulus, *, hop_s: float = 0.1, win_s: float = 0.25,
    embedding_frames: int = 4, delay_frames: int = 1, history_s: float = 30.0,
    exclusion_s: float = 0.0, activity_rms: float = 1e-6,
) -> dict[str, object]:
    """Best bounded-history similarity of ordered chroma delay stacks."""
    if isinstance(embedding_frames, bool) or embedding_frames < 1:
        raise ValueError("embedding_frames must be an integer >= 1")
    if isinstance(delay_frames, bool) or delay_frames < 1:
        raise ValueError("delay_frames must be an integer >= 1")
    if int(embedding_frames) != embedding_frames or int(delay_frames) != delay_frames:
        raise ValueError("embedding_frames and delay_frames must be integers")
    history_s = _finite_positive(history_s, "history_s")
    if isinstance(exclusion_s, bool) or not np.isfinite(exclusion_s) or exclusion_s < 0:
        raise ValueError("exclusion_s must be finite and >= 0")
    chroma, active, _centers, frame_bounds, h, w = _frame_chroma(
        stimulus, hop_s=hop_s, win_s=win_s, activity_rms=activity_rms
    )
    m, d, n = int(embedding_frames), int(delay_frames), len(chroma)
    first_offset = (m - 1) * d
    duration = (first_offset * h + w) / float(stimulus.sr_hz)
    values = np.zeros((n, 2), dtype=np.float32)
    valid = np.zeros(n, bool)
    count = np.zeros(n, dtype=np.int32)
    oldest = np.zeros(n, dtype=np.float32)
    deps = np.zeros((n, 2), dtype=np.float64)
    bounds = np.zeros((n, 2), dtype=np.float64)
    times = frame_bounds[:, 1].copy()
    sr = int(stimulus.sr_hz)
    origin = float(stimulus.start_offset_s)
    history_samples = int(np.floor(history_s * sr + 1e-10))
    exclusion_samples = int(np.ceil(float(exclusion_s) * sr - 1e-10))
    query_fraction = np.zeros(n, dtype=np.float32)
    for i in range(n):
        earliest = i - first_offset
        if earliest < 0:
            # Partial warmup support follows the eventual query-span anchor so
            # output bounds remain monotonic when the first full stack appears.
            bounds[i] = [frame_bounds[0, 0], frame_bounds[i, 1]]
            deps[i] = bounds[i]
            continue
        query_start_sample, query_end_sample = earliest * h, i * h + w
        query_start = origin + query_start_sample / sr
        query_end = origin + query_end_sample / sr
        bounds[i] = [query_start, query_end]
        history_start_sample = max(0, query_end_sample - history_samples)
        history_start = origin + history_start_sample / sr
        # Context includes both the searched history and the entire query stack.
        deps[i] = [min(query_start, history_start), query_end]
        q_idx = np.arange(earliest, i + 1, d)
        query_fraction[i] = active[q_idx].mean()
        if not np.all(active[q_idx]):
            continue
        q = chroma[q_idx].reshape(-1).astype(np.float64)
        q /= max(np.linalg.norm(q), _EPS)
        candidates: list[tuple[float, int]] = []
        # Bound the scan in integer sample geometry.  This keeps exact boundary
        # eligibility independent of the stimulus clock offset.
        j_lo = max(first_offset, first_offset + int(np.ceil((history_start_sample - 1e-10) / h)))
        j_hi = int(np.floor((query_start_sample - w + 1e-10) / h))
        j_hi = min(j_hi, i - int(np.ceil((exclusion_samples - 1e-10) / h)))
        for j in range(j_lo, j_hi + 1):
            cand_start_sample = (j - first_offset) * h
            cand_end_sample = j * h + w
            lag_samples = (i - j) * h
            if (cand_start_sample < history_start_sample or cand_end_sample > query_start_sample
                    or lag_samples < exclusion_samples):
                continue
            idx = np.arange(j - first_offset, j + 1, d)
            if np.all(active[idx]):
                candidates.append((_delay_stack_similarity(q, chroma[idx]), j))
        count[i] = len(candidates)
        if not candidates:
            continue
        # Stable smallest-lag tie break.
        best, j = max(candidates, key=lambda item: (item[0], item[1]))
        values[i] = [best, (i - j) * h / sr]
        oldest[i] = max((i - jj) * h / sr for _score, jj in candidates)
        valid[i] = True
    params = {"hop_s": hop_s, "win_s": win_s, "embedding_frames": m, "delay_frames": d,
              "history_s": history_s, "exclusion_s": exclusion_s, "activity_rms": activity_rms}
    realized = h / float(stimulus.sr_hz)
    pair_id = _output_pair_id(stimulus, "audio.music.sequence_recurrence", params)
    diagnostics = np.column_stack((valid, count, oldest, query_fraction)).astype(np.float32)
    return {
        "default": _series(stimulus, values, times, bounds, extractor="audio.music.sequence_recurrence",
                           params=params, names=["recurrence", "best_lag_s"], policy="past_only",
                           dependencies=deps, hop_s=realized,
                           extra={"embedding_waveform_span_s": duration, "validity_output": "diagnostics",
                                  "validity_column": "valid", "output_pair_id": pair_id}),
        "diagnostics": _series(stimulus, diagnostics, times, bounds,
                               extractor="audio.music.sequence_recurrence.diagnostics", params=params,
                               names=["valid", "candidate_count", "oldest_candidate_lag_s", "query_activity_fraction"],
                               policy="past_only", dependencies=deps, hop_s=realized,
                               extra={"embedding_waveform_span_s": duration, "output_pair_id": pair_id}),
    }
