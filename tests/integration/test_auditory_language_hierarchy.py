"""One speech wav -> acoustics, prosody, phonology, semantics on one grid."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import mfcc, rms, spectral_stats
from natural_features.features.audio.prosody import prosody_features
from natural_features.fmri.design import concat_feature_series
from natural_features.fmri.resample import build_tr_grid, resample_feature_series
from natural_features.workflows import (
    extract_acoustic_phonetics,
    extract_multiscale_language,
)

ROOT = Path(__file__).resolve().parents[2]
TIER_A_AUDIO = ROOT / "tests" / "stimuli" / "tier_a" / "audio_speechlike.wav"
TIER_A_TRANSCRIPT = ROOT / "tests" / "stimuli" / "tier_a" / "transcript_reference.txt"
EXAMPLE = ROOT / "examples" / "auditory_language_hierarchy.py"

# Offline semantics: embeddings and lexical controls need no model weights.
# Language-model surprisal is a strict backend and is exercised separately.
OFFLINE_SEMANTIC_FAMILIES = [
    "sentence_embeddings",
    "paragraph_embeddings",
    "lexical_controls",
]


def _require_lm_backend() -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
    except ImportError:
        pytest.skip("transformers+torch are not installed")


@pytest.mark.media
def test_hierarchy_levels_align_on_one_grid(tmp_path) -> None:  # noqa: ANN001
    audio = AudioStimulus.from_wav(TIER_A_AUDIO)
    duration_s = audio.start_offset_s + audio.samples.shape[0] / audio.sr_hz
    tr_s = 1.0
    grid = build_tr_grid(duration_s=duration_s, tr_s=tr_s, start_s=audio.start_offset_s)

    def on_grid(fs):  # noqa: ANN001, ANN202
        return resample_feature_series(fs, tr_s=tr_s, method="mean", time_grid_s=grid)

    acoustic = [
        on_grid(rms(audio)),
        on_grid(mfcc(audio, n_mfcc=13, n_mels=40, include_deltas=True)),
        on_grid(spectral_stats(audio)),
    ]
    prosody = on_grid(prosody_features(audio))

    phon = extract_acoustic_phonetics(audio, posterior_backend="acoustic")
    phonology = [on_grid(phon.posteriorgrams), on_grid(phon.articulatory)]

    language = extract_multiscale_language(
        audio,
        transcript_text=TIER_A_TRANSCRIPT.read_text(encoding="utf-8").strip(),
        scales_s=[tr_s],
        feature_families=OFFLINE_SEMANTIC_FAMILIES,
        provider_config={"provider": "local_bow", "dim": 64},
        standardize=False,
        add_intercept=False,
        cache_dir=tmp_path / "emb_cache",
    )
    semantic = language.by_scale[tr_s]

    levels = [*acoustic, prosody, *phonology, semantic]
    for fs in levels:
        assert np.allclose(fs.times_s, grid)

    design = concat_feature_series(levels, standardize=True, add_intercept=False)
    names = [str(n) for n in design.coords.get("feature", [])]
    assert design.values.shape == (len(grid), len(names))
    assert np.isfinite(design.values).all()

    # Every level of the hierarchy contributes named columns.
    assert any(n.startswith("rms") for n in names)
    assert "f0_hz" in names
    assert any(n.startswith("art.") or "posterior_entropy" in n for n in names)
    assert any(n.startswith("sem.sent.") for n in names)
    assert any(n.startswith("sem.par.") for n in names)

    # Semantics segmented into units above the word level.
    assert language.qc["n_words"] > 0
    assert language.qc["n_sentences"] >= 1
    assert language.qc["n_paragraphs"] >= 1


@pytest.mark.media
def test_surprisal_adds_a_predictability_column(tmp_path) -> None:  # noqa: ANN001
    _require_lm_backend()
    audio = AudioStimulus.from_wav(TIER_A_AUDIO)
    tr_s = 1.0
    language = extract_multiscale_language(
        audio,
        transcript_text=TIER_A_TRANSCRIPT.read_text(encoding="utf-8").strip(),
        scales_s=[tr_s],
        feature_families=[*OFFLINE_SEMANTIC_FAMILIES, "surprisal"],
        provider_config={"provider": "local_bow", "dim": 64},
        standardize=False,
        add_intercept=False,
        cache_dir=tmp_path / "emb_cache",
    )
    names = [str(n) for n in language.by_scale[tr_s].coords.get("feature", [])]
    assert any("surprisal" in n for n in names)


@pytest.mark.media
def test_hierarchy_example_script_runs_offline(tmp_path) -> None:  # noqa: ANN001
    out_prefix = tmp_path / "hierarchy_demo"
    proc = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--audio-wav",
            str(TIER_A_AUDIO),
            "--transcript",
            str(TIER_A_TRANSCRIPT),
            "--resolution-s",
            "1.0",
            "--out-prefix",
            str(out_prefix),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    npz = np.load(out_prefix.with_suffix(".npz"))
    meta = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))

    x = npz["X"]
    times = npz["times_s"]
    assert x.shape[0] == len(times) > 0
    assert x.shape[1] == meta["n_columns"] == len(meta["feature_names"])
    assert np.isfinite(x).all()

    blocks = meta["block_slices"]
    assert set(blocks) == {"acoustic", "prosody", "phonology", "semantic"}
    cursor = 0
    for level, (lo, hi) in sorted(blocks.items(), key=lambda kv: kv[1][0]):
        assert lo == cursor and hi > lo, level
        cursor = hi
    assert cursor == x.shape[1]

    names = meta["feature_names"]
    for level, (lo, hi) in blocks.items():
        assert all(n.startswith(f"{level}.") for n in names[lo:hi]), level
