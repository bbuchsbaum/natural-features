#!/usr/bin/env python3
"""Hierarchical auditory-language features from one speech wav.

Builds four feature blocks on one shared time grid, from low-level acoustics
up to sentence/paragraph semantics:

1. ``acoustic.*``   envelope, MFCC, and spectral shape (cochlea-adjacent)
2. ``prosody.*``    F0, voicing, energy, and spectral prosody controls
3. ``phonology.*``  phone-class posteriors mapped to articulatory features
4. ``semantic.*``   sentence/paragraph embeddings, surprisal, lexical controls

The blocks are exported individually and as one standardized design matrix
with recorded column ranges, ready for banded/hierarchical regression or
variance partitioning.

Deterministic offline run against the bundled Tier-A fixture:

    python examples/auditory_language_hierarchy.py \
        --audio-wav tests/stimuli/tier_a/audio_speechlike.wav \
        --transcript tests/stimuli/tier_a/transcript_reference.txt \
        --resolution-s 1.0 \
        --out-prefix /tmp/hierarchy_demo

Swap in the strict neural backends on real data:

    python examples/auditory_language_hierarchy.py \
        --audio-wav story.wav \
        --posterior-backend ctc \
        --semantic-provider openai
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.lowlevel import mfcc, rms, spectral_stats
from natural_features.features.audio.prosody import prosody_features
from natural_features.fmri.design import concat_feature_series
from natural_features.fmri.resample import build_tr_grid, resample_feature_series
from natural_features.util.io import atomic_numpy_savez, atomic_write_json
from natural_features.workflows import (
    extract_acoustic_phonetics,
    extract_multiscale_language,
)


def _prefixed(fs: FeatureSeries, prefix: str) -> FeatureSeries:
    names = [str(n) for n in fs.coords.get("feature", [f"f{i}" for i in range(fs.values.shape[1])])]
    return FeatureSeries(
        values=fs.values,
        times_s=fs.times_s,
        dims=fs.dims,
        coords={"feature": [f"{prefix}{n}" for n in names]},
        metadata=fs.metadata,
        timebase=fs.timebase,
        time_bounds_s=fs.time_bounds_s,
        temporal_context=fs.temporal_context,
    )


def _on_grid(fs: FeatureSeries, *, tr_s: float, grid: np.ndarray, method: str = "mean") -> FeatureSeries:
    return resample_feature_series(fs, tr_s=tr_s, method=method, time_grid_s=grid)


def _read_transcript(raw: str | None) -> str | None:
    if raw is None:
        return None
    p = Path(raw)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-wav", required=True, help="Mono speech wav")
    ap.add_argument(
        "--transcript",
        default=None,
        help="Transcript text or path to a .txt file. Omit to run strict ASR (requires faster-whisper).",
    )
    ap.add_argument("--resolution-s", type=float, default=1.0, help="Shared output grid step in seconds")
    ap.add_argument(
        "--posterior-backend",
        choices=["acoustic", "ctc"],
        default="acoustic",
        help="'ctc' uses the strict wav2vec2 phoneme model (requires transformers+torch and local weights)",
    )
    ap.add_argument("--ctc-device", default="auto", help="CTC device: auto, cuda, mps, or cpu")
    ap.add_argument(
        "--semantic-provider",
        default="local_bow",
        help="Embedding provider: local_bow (offline), local_hash, or openai",
    )
    ap.add_argument("--out-prefix", default="hierarchy_design", help="Output file prefix")
    args = ap.parse_args()

    audio = AudioStimulus.from_wav(args.audio_wav)
    duration_s = float(audio.start_offset_s + audio.samples.shape[0] / audio.sr_hz)
    tr_s = float(args.resolution_s)
    grid = build_tr_grid(duration_s=duration_s, tr_s=tr_s, start_s=audio.start_offset_s)
    transcript = _read_transcript(args.transcript)

    # Level 1: low-level acoustics.
    acoustic = [
        _prefixed(_on_grid(rms(audio), tr_s=tr_s, grid=grid), "acoustic.rms."),
        _prefixed(
            _on_grid(
                mfcc(audio, n_mfcc=13, n_mels=40, include_deltas=True),
                tr_s=tr_s,
                grid=grid,
            ),
            "acoustic.mfcc.",
        ),
        _prefixed(_on_grid(spectral_stats(audio), tr_s=tr_s, grid=grid), "acoustic.spectral."),
    ]

    # Level 2: prosody (F0, voicing, energy, spectral shape controls).
    prosody = [_prefixed(_on_grid(prosody_features(audio), tr_s=tr_s, grid=grid), "prosody.")]

    # Level 3: phonetics/phonology (phone posteriors -> articulatory features).
    phonetics = extract_acoustic_phonetics(
        audio,
        posterior_backend=args.posterior_backend,
        ctc_device=args.ctc_device,
    )
    phonology = [
        _prefixed(_on_grid(phonetics.posteriorgrams, tr_s=tr_s, grid=grid), "phonology.post."),
        _prefixed(_on_grid(phonetics.articulatory, tr_s=tr_s, grid=grid), "phonology.artic."),
    ]

    # Level 4: sentence/paragraph semantics plus word-level predictability.
    language = extract_multiscale_language(
        audio,
        transcript_text=transcript,
        scales_s=[tr_s],
        feature_families=[
            "sentence_embeddings",
            "paragraph_embeddings",
            "surprisal",
            "lexical_controls",
        ],
        provider_config={"provider": args.semantic_provider},
        standardize=False,
        add_intercept=False,
    )
    semantic_dm = language.by_scale[tr_s]
    if len(semantic_dm.times_s) != len(grid) or not np.allclose(semantic_dm.times_s, grid):
        # ASR word onsets can start after t=0; re-render onto the canonical grid.
        semantic_dm = _on_grid(semantic_dm, tr_s=tr_s, grid=grid, method="nearest")
    semantics = [_prefixed(semantic_dm, "semantic.")]

    blocks = {
        "acoustic": acoustic,
        "prosody": prosody,
        "phonology": phonology,
        "semantic": semantics,
    }
    ordered = [fs for level in blocks.values() for fs in level]
    design = concat_feature_series(ordered, standardize=True, add_intercept=False)
    names = [str(n) for n in design.coords.get("feature", [])]

    block_slices: dict[str, list[int]] = {}
    cursor = 0
    for level, feats in blocks.items():
        width = sum(f.values.shape[1] for f in feats)
        block_slices[level] = [cursor, cursor + width]
        cursor += width

    out_prefix = Path(args.out_prefix)
    npz_path = out_prefix.with_suffix(".npz")
    meta_path = out_prefix.with_suffix(".json")
    atomic_numpy_savez(
        npz_path,
        X=design.values.astype(np.float32),
        times_s=design.times_s.astype(np.float64),
    )
    atomic_write_json(
        meta_path,
        {
            "audio_wav": str(args.audio_wav),
            "duration_s": duration_s,
            "resolution_s": tr_s,
            "n_rows": int(design.values.shape[0]),
            "n_columns": int(design.values.shape[1]),
            "feature_names": names,
            "block_slices": block_slices,
            "posterior_backend": args.posterior_backend,
            "semantic_provider": args.semantic_provider,
            "language_qc": language.qc,
        },
    )

    print(f"design matrix: {design.values.shape[0]} rows x {design.values.shape[1]} columns")
    for level, (lo, hi) in block_slices.items():
        print(f"  {level:<10} columns [{lo}:{hi}]")
    print(f"wrote {npz_path} and {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
