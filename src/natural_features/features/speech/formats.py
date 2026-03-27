"""Interchange formats for aligned speech events (CTM, TextGrid)."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from natural_features.core.feature_types import EventSeries
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.contracts import ensure_word_event_metadata
from natural_features.util.io import atomic_write_text


def _base_word_metadata(extractor_name: str, params: dict[str, object]) -> dict[str, object]:
    md = extractor_metadata(extractor_name, params=params)
    return ensure_word_event_metadata(
        md,
        asr_model_name="unknown",
        aligner_backend="imported",
        aligner_version="n/a",
    )


def write_ctm(
    words: EventSeries,
    path: str | Path,
    *,
    utterance_id: str = "utt",
    channel: str = "1",
) -> Path:
    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    conf = words.confidence if words.confidence is not None else np.ones(len(words), dtype=np.float32)
    lines: list[str] = []
    for i in range(len(words)):
        onset = float(words.onset_s[i])
        dur = float(max(0.0, words.offset_s[i] - words.onset_s[i]))
        tok = str(labels[i]).replace(" ", "_")
        score = float(conf[i])
        lines.append(f"{utterance_id} {channel} {onset:.6f} {dur:.6f} {tok} {score:.6f}")
    out = Path(path)
    return atomic_write_text(out, "\n".join(lines) + ("\n" if lines else ""))


def read_ctm(
    path: str | Path,
    *,
    default_confidence: float = 1.0,
) -> EventSeries:
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    on: list[float] = []
    off: list[float] = []
    lab: list[str] = []
    conf: list[float] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid CTM line (expected >=5 fields): {ln}")
        onset = float(parts[2])
        dur = float(parts[3])
        token = str(parts[4]).replace("_", " ")
        score = float(parts[5]) if len(parts) >= 6 else float(default_confidence)
        on.append(onset)
        off.append(onset + max(0.0, dur))
        lab.append(token)
        conf.append(score)
    md = _base_word_metadata(
        "speech.format.read_ctm",
        params={"path": str(p.name)},
    )
    return EventSeries(
        onset_s=np.asarray(on, dtype=np.float64),
        offset_s=np.asarray(off, dtype=np.float64),
        label=np.asarray(lab, dtype=object),
        confidence=np.asarray(conf, dtype=np.float32),
        metadata=md,
    )


def write_textgrid(
    words: EventSeries,
    path: str | Path,
    *,
    tier_name: str = "words",
) -> Path:
    labels = words.label if words.label is not None else np.array([""] * len(words), dtype=object)
    xmin = float(words.onset_s[0]) if len(words) else 0.0
    xmax = float(words.offset_s[-1]) if len(words) else 0.0
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {xmin:.6f}",
        f"xmax = {xmax:.6f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        f'        name = "{tier_name}"',
        f"        xmin = {xmin:.6f}",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(words)}",
    ]
    for i in range(len(words)):
        tok = str(labels[i]).replace('"', '""')
        lines.extend(
            [
                f"        intervals [{i + 1}]:",
                f"            xmin = {float(words.onset_s[i]):.6f}",
                f"            xmax = {float(words.offset_s[i]):.6f}",
                f'            text = "{tok}"',
            ]
        )
    out = Path(path)
    return atomic_write_text(out, "\n".join(lines) + "\n")


_RE_XMIN = re.compile(r"^\s*xmin\s*=\s*([0-9eE+\-.]+)\s*$")
_RE_XMAX = re.compile(r"^\s*xmax\s*=\s*([0-9eE+\-.]+)\s*$")
_RE_TEXT = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')


def read_textgrid(path: str | Path) -> EventSeries:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    on: list[float] = []
    off: list[float] = []
    lab: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "intervals [" not in line:
            i += 1
            continue
        if i + 3 >= len(lines):
            raise ValueError("Malformed TextGrid: truncated interval block")
        m_on = _RE_XMIN.match(lines[i + 1])
        m_off = _RE_XMAX.match(lines[i + 2])
        m_txt = _RE_TEXT.match(lines[i + 3])
        if not (m_on and m_off and m_txt):
            raise ValueError(f"Malformed TextGrid interval block near line {i + 1}")
        on.append(float(m_on.group(1)))
        off.append(float(m_off.group(1)))
        lab.append(m_txt.group(1).replace('""', '"'))
        i += 4
    md = _base_word_metadata(
        "speech.format.read_textgrid",
        params={"path": str(p.name)},
    )
    return EventSeries(
        onset_s=np.asarray(on, dtype=np.float64),
        offset_s=np.asarray(off, dtype=np.float64),
        label=np.asarray(lab, dtype=object),
        confidence=np.ones(len(on), dtype=np.float32),
        metadata=md,
    )
