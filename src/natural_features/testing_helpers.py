"""Testing-only helper functions used in unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.audio.lowlevel import mfcc
from natural_features.features.vision.lowlevel import visual_energy
from natural_features.util.hashing import stable_hash
from natural_features.workflows.acoustic_phonetics import extract_acoustic_phonetics
from natural_features.workflows.audio_batch import extract_audio_files
from natural_features.workflows.multiscale_language import extract_multiscale_language


def pass_through_feature_series(x: FeatureSeries, *, scale: float = 1.0) -> FeatureSeries:
    return FeatureSeries(
        values=x.values * scale,
        times_s=x.times_s,
        metadata={"extractor_id": "test.pass_through", "params_hash": "test"},
    )


def wrong_typed_output(x: FeatureSeries) -> dict[str, str]:
    return {"default": "not_a_feature_series"}


def _array_fingerprint(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=np.float64)
    rounded = np.round(arr, 6)
    payload = {
        "shape": list(rounded.shape),
        "sum": float(np.sum(rounded)),
        "mean": float(np.mean(rounded)),
        "std": float(np.std(rounded)),
        "head": rounded.reshape(-1)[:64].tolist(),
    }
    return stable_hash(payload, length=20)


def _feature_summary(values: np.ndarray, times_s: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "time_start_s": float(times_s[0]) if len(times_s) else 0.0,
        "time_end_s": float(times_s[-1]) if len(times_s) else 0.0,
        "fingerprint": _array_fingerprint(values),
    }


def build_tier_a_golden_reference(base_dir: str | Path) -> dict[str, Any]:
    base = Path(base_dir)
    tier_a = base / "tests" / "stimuli" / "tier_a"
    wav = tier_a / "audio_speechlike.wav"
    vid_npy = tier_a / "video_scene_cut.npy"
    transcript = (tier_a / "transcript_reference.txt").read_text(encoding="utf-8").strip()

    audio = AudioStimulus.from_wav(wav)
    frames = np.load(vid_npy)
    video = VideoStimulus.from_array(frames, fps=10.0)

    ve = visual_energy(video, include_deltas=True)
    m = mfcc(audio, hop_s=0.01, win_s=0.025, n_mfcc=13, n_mels=40, include_deltas=True)

    ap = extract_acoustic_phonetics(
        audio,
        posterior_backend="acoustic",
        hop_s=0.02,
        resolution_s=0.5,
        execution_mode="fallback",
    )

    ml = extract_multiscale_language(
        transcript,
        scales_s=[2.0, 4.0, 16.0],
        provider_config={"provider": "local_bow", "dim": 256},
        execution_mode="fallback",
        as_dataframe=False,
    )

    batch = extract_audio_files(
        [wav],
        resolution_s=1.0,
        selected_features=["rms", "mfcc", "spectral_stats", "vad"],
        as_dataframe=False,
        collapse="mean+sd",
    )
    one = batch.files[wav.stem]

    return {
        "reference_version": 1,
        "tier": "A",
        "stimulus": {
            "audio": str(wav.relative_to(base)),
            "video": str(vid_npy.relative_to(base)),
        },
        "visual_energy": _feature_summary(ve.values, ve.times_s),
        "mfcc": _feature_summary(m.values, m.times_s),
        "acoustic_phonetics": {
            "posteriorgrams": _feature_summary(ap.posteriorgrams.values, ap.posteriorgrams.times_s),
            "articulatory": _feature_summary(ap.articulatory.values, ap.articulatory.times_s),
        },
        "multiscale_language": {
            str(scale): _feature_summary(fs.values, fs.times_s) for scale, fs in ml.by_scale.items()
        },
        "audio_batch": {
            "matrix": _feature_summary(one.matrix, one.times_s),
            "collapsed": {
                "shape": list(one.collapsed_vector.shape) if one.collapsed_vector is not None else [0],
                "fingerprint": _array_fingerprint(one.collapsed_vector) if one.collapsed_vector is not None else "",
            },
        },
    }
