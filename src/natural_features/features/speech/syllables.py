"""Syllable onset/nucleus/coda structure from phone events.

Syllabification uses the maximal-onset heuristic: vowels are nuclei, and each
intervocalic consonant cluster is split so that up to ``max_onset`` consonants
attach to the following nucleus and the remainder close the preceding
syllable. Temporal gaps longer than ``boundary_gap_s`` (pauses, word breaks in
sparse alignments) act as hard syllable boundaries. This is a heuristic over
phone labels, not a lexicon-based syllabifier.
"""

from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.phonology import (
    _features_for_label,
    _normalize_phone_label,
)

ONC_CHANNELS = ["syll_onset", "syll_nucleus", "syll_coda"]

_ONSET, _NUCLEUS, _CODA = 0, 1, 2


def _classify_chunk(kinds: list[str], *, max_onset: int) -> list[int]:
    """Assign onset/nucleus/coda roles within one boundary-free phone chunk."""

    roles = [_ONSET] * len(kinds)
    nuclei = [i for i, kind in enumerate(kinds) if kind == "nucleus"]
    if not nuclei:
        return roles
    for i in nuclei:
        roles[i] = _NUCLEUS
    # Consonants before the first nucleus stay onsets; after the last, codas.
    for i in range(nuclei[-1] + 1, len(kinds)):
        roles[i] = _CODA
    # Intervocalic clusters: maximal onset for the following syllable.
    for prev_n, next_n in zip(nuclei[:-1], nuclei[1:]):
        cluster = list(range(prev_n + 1, next_n))
        n_onset = min(len(cluster), max_onset)
        for j in cluster[: len(cluster) - n_onset]:
            roles[j] = _CODA
        for j in cluster[len(cluster) - n_onset :]:
            roles[j] = _ONSET
    return roles


def syllable_onc(
    phones: EventSeries,
    *,
    hop_s: float = 0.01,
    max_onset: int = 3,
    boundary_gap_s: float = 0.15,
    duration_s: float | None = None,
) -> FeatureSeries:
    """Rasterize syllable onset/nucleus/coda occupancy onto a hop grid."""

    if hop_s <= 0:
        raise ValueError("hop_s must be > 0")
    labels = phones.label if phones.label is not None else np.array([], dtype=object)
    keep: list[int] = []
    kinds: list[str] = []
    for i in range(len(phones)):
        lab = _normalize_phone_label(str(labels[i]) if i < len(labels) else "")
        feats = _features_for_label(lab)
        if not feats or "silence" in feats:
            continue
        keep.append(i)
        kinds.append("nucleus" if "vowel" in feats else "consonant")

    if keep:
        stop = float(phones.offset_s[keep[-1]])
        if duration_s is not None:
            stop = max(stop, float(duration_s))
        n = max(1, int(np.round(stop / hop_s)))
    else:
        n = max(1, int(np.round((duration_s or hop_s) / hop_s)))
    times = (np.arange(n, dtype=np.float64) + 0.5) * hop_s
    values = np.zeros((n, 3), dtype=np.float32)

    # Split into chunks at temporal gaps, then classify each chunk.
    chunk: list[int] = []
    chunks: list[list[int]] = []
    for pos, i in enumerate(keep):
        if chunk:
            prev = keep[pos - 1]
            gap = float(phones.onset_s[i]) - float(phones.offset_s[prev])
            if gap > boundary_gap_s:
                chunks.append(chunk)
                chunk = []
        chunk.append(pos)
    if chunk:
        chunks.append(chunk)

    for members in chunks:
        roles = _classify_chunk([kinds[p] for p in members], max_onset=max_onset)
        for p, role in zip(members, roles):
            i = keep[p]
            on = float(phones.onset_s[i])
            off = float(phones.offset_s[i])
            mask = (times >= on) & (times < max(off, on + hop_s))
            values[mask, role] = 1.0

    md = extractor_metadata(
        "speech.syllables.onc",
        params={
            "hop_s": hop_s,
            "max_onset": max_onset,
            "boundary_gap_s": boundary_gap_s,
        },
        extra={"backend": "maximal_onset_heuristic"},
    )
    return FeatureSeries(
        values=values,
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": list(ONC_CHANNELS)},
        metadata=md,
        timebase=TimebaseSpec(
            kind="audio_hop", hop_s=hop_s, sampling_rate_hz=1.0 / hop_s
        ),
    )
