from __future__ import annotations

import csv
import os
from pathlib import Path
import wave

import numpy as np
import pytest

from natural_features.workflows.audio_batch import extract_audio_files
from natural_features.workflows.multiscale_language import extract_multiscale_language


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNL_ROOT = REPO_ROOT / "data" / "snl_2023_task"
ENABLE_FLAG = "NF_ENABLE_EXTERNAL_DATA"
ROOT_ENV = "NF_SNL_DATA_ROOT"


def _should_run() -> bool:
    return os.environ.get(ENABLE_FLAG, "0").strip() == "1"


def _root() -> Path:
    return Path(os.environ.get(ROOT_ENV, str(DEFAULT_SNL_ROOT)))


def _load_sentence_stimuli(root: Path) -> list[str]:
    p = root / "sentence_stimuli.csv"
    rows: list[str] = []
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "stimulus" not in (r.fieldnames or []):
            raise ValueError("sentence_stimuli.csv missing 'stimulus' column")
        for row in r:
            stim = str(row["stimulus"]).strip()
            if stim:
                rows.append(stim)
    return rows


def _load_sentence_mapping(root: Path) -> dict[str, dict[str, object]]:
    canonical = root / "sentence_metadata.csv"
    if canonical.exists():
        out: dict[str, dict[str, object]] = {}
        with canonical.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            required = {"stimulus", "sentence", "length_tag"}
            if not required.issubset(set(r.fieldnames or [])):
                raise ValueError(f"{canonical} missing required columns: {sorted(required)}")
            for row in r:
                stim = str(row.get("stimulus", "")).strip()
                sent = str(row.get("sentence", "")).strip()
                length_val = int(str(row.get("length_tag", "0")).strip() or 0)
                if stim and sent:
                    out[stim] = {"sentence": sent, "length": length_val}
        return out

    mapping: dict[str, dict[str, object]] = {}
    for p in sorted(root.glob("memory_run_*_version_*.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            required = {"stimulus", "sentence", "length"}
            if not required.issubset(set(r.fieldnames or [])):
                continue
            for row in r:
                stim = str(row.get("stimulus", "")).strip()
                sent = str(row.get("sentence", "")).strip()
                if not stim or not sent:
                    continue
                length_val = int(str(row.get("length", "0")).strip() or 0)
                prev = mapping.get(stim)
                if prev is None:
                    mapping[stim] = {"sentence": sent, "length": length_val}
                else:
                    # Contract: same stimulus should map to the same sentence + length.
                    assert prev["sentence"] == sent, f"Conflicting sentence mapping for {stim}"
                    assert int(prev["length"]) == length_val, f"Conflicting length mapping for {stim}"
    return mapping


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        n_frames = w.getnframes()
        sr = w.getframerate()
    return float(n_frames / sr)


def _length_tag(stim_rel_path: str) -> int:
    stem = Path(stim_rel_path).stem
    last = stem.rsplit("_", 1)[-1]
    return int(last)


if not _should_run():
    pytest.skip(
        f"External dataset tests disabled. Set {ENABLE_FLAG}=1 to run.",
        allow_module_level=True,
    )

ROOT = _root()
if not ROOT.exists():
    pytest.skip(
        f"SNL dataset root not found at {ROOT}. Set {ROOT_ENV} to override.",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.external, pytest.mark.media]


def test_snl_contracts_stimuli_mapping_and_duration() -> None:
    stimuli = _load_sentence_stimuli(ROOT)
    mapping = _load_sentence_mapping(ROOT)

    assert len(stimuli) == 27
    assert len(set(stimuli)) == 27
    assert len(mapping) >= len(stimuli)

    missing_mapping = [s for s in stimuli if s not in mapping]
    assert not missing_mapping, f"Missing sentence mapping for stimuli: {missing_mapping[:5]}"

    durations_by_length: dict[int, list[float]] = {8: [], 12: [], 16: []}
    for stim in stimuli:
        wav_path = ROOT / stim
        assert wav_path.exists(), f"Missing wav file: {wav_path}"
        sentence = str(mapping[stim]["sentence"])
        mapped_length = int(mapping[stim]["length"])
        assert len(sentence.split()) >= 4
        assert mapped_length in {8, 12, 16}
        assert _length_tag(stim) == mapped_length
        dur = _wav_duration_seconds(wav_path)
        assert dur > 1.5
        durations_by_length[mapped_length].append(dur)

    # Contract: longer length tags should correspond to longer utterances on average.
    med8 = float(np.median(durations_by_length[8]))
    med12 = float(np.median(durations_by_length[12]))
    med16 = float(np.median(durations_by_length[16]))
    assert med8 < med12 < med16


def test_snl_multiscale_language_real_audio_subset(tmp_path) -> None:
    stimuli = _load_sentence_stimuli(ROOT)
    mapping = _load_sentence_mapping(ROOT)
    selected = [ROOT / s for s in stimuli[:2]]
    transcript = str(mapping[stimuli[0]]["sentence"])

    result = extract_multiscale_language(
        selected[0],
        transcript_text=transcript,
        scales_s=[2.0, 4.0, 16.0],
        provider_config={"provider": "local_hash", "dim": 64},
        cache_dir=tmp_path / "cache",
        feature_families=["sentence_embeddings", "surprisal", "lexical_controls"],
        as_dataframe=True,
    )

    assert sorted(result.by_scale.keys()) == [2.0, 4.0, 16.0]
    assert result.words is not None
    assert len(result.words) == len(transcript.split())
    assert result.by_scale_dataframe is not None
    for scale, fs in result.by_scale.items():
        assert fs.values.ndim == 2
        assert fs.values.shape[0] == len(fs.times_s)
        assert fs.values.shape[1] > 0
        assert scale in result.by_scale_dataframe
        assert len(result.by_scale_dataframe[scale]) == fs.values.shape[0]


def test_snl_audio_batch_subset_real_audio() -> None:
    stimuli = _load_sentence_stimuli(ROOT)
    selected = [ROOT / s for s in stimuli[:3]]
    batch = extract_audio_files(
        selected,
        resolution_s=1.0,
        selected_features=["rms", "mfcc"],
        collapse="mean+sd",
        as_dataframe=True,
    )

    assert len(batch.files) == 3
    assert batch.long_dataframe is not None
    assert batch.collapsed_dataframe is not None
    assert len(batch.collapsed_dataframe) == 3
    # Contract: collapsed stats should be finite.
    numeric = batch.collapsed_dataframe.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
