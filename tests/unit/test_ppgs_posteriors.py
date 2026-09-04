from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from natural_features.core.backend_errors import (
    BackendDependencyError,
    BackendLoadError,
)
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.phonology import articulatory_from_posteriors
from natural_features.features.speech.ppgs import (
    PPGS_HOP_S,
    PPGS_PHONE_LABELS,
    PPGS_SILENCE_INDEX,
    leading_trailing_silence_frames,
    pad_posteriors_with_silence,
    ppg_phone_posteriors,
)
from natural_features.workflows.acoustic_phonetics import extract_acoustic_phonetics


def _tone(duration_s: float, sr: int = 16000, freq: float = 220.0) -> np.ndarray:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _padded_clip() -> AudioStimulus:
    sr = 16000
    wav = np.concatenate(
        [
            np.zeros(sr // 2, dtype=np.float32),
            _tone(0.5, sr=sr),
            np.zeros(sr // 2, dtype=np.float32),
        ]
    )
    return AudioStimulus.from_array(wav, sr_hz=sr)


def _install_optional_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    if "ppgs" not in sys.modules:
        monkeypatch.setitem(sys.modules, "ppgs", types.ModuleType("ppgs"))
    try:
        import torch  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))


def _patch_ppgs_inference(
    monkeypatch: pytest.MonkeyPatch,
    infer,
    *,
    checkpoint: str = "/tmp/fake-ppgs.pt",
) -> None:
    _install_optional_imports(monkeypatch)
    monkeypatch.setattr(
        "natural_features.features.speech.ppgs._resolve_ppgs_checkpoint",
        lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "natural_features.features.speech.ppgs._infer_ppg_matrix", infer
    )


def test_ppg_phone_inventory_is_cmu_40() -> None:
    assert len(PPGS_PHONE_LABELS) == 40
    assert PPGS_PHONE_LABELS[PPGS_SILENCE_INDEX] == "silence"
    assert "EY" in PPGS_PHONE_LABELS
    assert "AY" in PPGS_PHONE_LABELS
    assert "AW" in PPGS_PHONE_LABELS
    assert "OY" in PPGS_PHONE_LABELS


def test_leading_trailing_silence_frames_on_padded_tone() -> None:
    wav = _padded_clip().samples
    lead, trail = leading_trailing_silence_frames(wav)
    assert lead == 50
    assert trail == 50


def test_pad_posteriors_with_silence_restores_edges() -> None:
    interior = np.zeros((3, 40), dtype=np.float32)
    interior[:, 0] = 1.0
    out = pad_posteriors_with_silence(interior, lead_frames=2, trail_frames=1)
    assert out.shape == (6, 40)
    assert np.allclose(out[:2, PPGS_SILENCE_INDEX], 1.0)
    assert np.allclose(out[-1, PPGS_SILENCE_INDEX], 1.0)
    assert np.allclose(out[2:5, 0], 1.0)


def test_ppg_posteriors_reject_surrogate_execution() -> None:
    with pytest.raises(ValueError, match="proxy and surrogate"):
        ppg_phone_posteriors(_padded_clip(), execution_mode="fallback")


def test_ppg_posteriors_reject_unknown_representation() -> None:
    with pytest.raises(ValueError, match="representation"):
        ppg_phone_posteriors(_padded_clip(), representation="encodec")


def test_ppg_posteriors_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ppgs", None)
    with pytest.raises(BackendDependencyError) as raised:
        ppg_phone_posteriors(_padded_clip(), checkpoint="/tmp/unused.pt")
    assert raised.value.backend == "ppgs"
    assert raised.value.phase == "dependency"


def test_ppg_checkpoint_load_failure_is_not_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_optional_imports(monkeypatch)

    def boom(*_args: object, **_kwargs: object) -> str:
        raise OSError("not in cache")

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(BackendLoadError) as raised:
        ppg_phone_posteriors(_padded_clip(), local_files_only=True)
    assert raised.value.phase == "load"


def test_ppg_posteriors_with_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_infer(wav: np.ndarray, sample_rate: int, **kwargs: object) -> np.ndarray:
        seen["n"] = int(wav.shape[0])
        seen["sample_rate"] = int(sample_rate)
        seen["kwargs"] = kwargs
        n = max(1, int(wav.shape[0]) // 160)
        probs = np.zeros((n, len(PPGS_PHONE_LABELS)), dtype=np.float32)
        probs[:, 0] = 1.0
        return probs

    _patch_ppgs_inference(monkeypatch, fake_infer)
    out = ppg_phone_posteriors(
        _padded_clip(),
        representation="mel",
        checkpoint="/tmp/fake-ppgs.pt",
        trim_edge_silence=True,
        legacy_mode=True,
    )
    assert out.values.shape[1] == 40
    assert out.values.shape[0] == 150
    assert list(out.coords["feature"]) == PPGS_PHONE_LABELS
    assert out.timebase.hop_s == pytest.approx(PPGS_HOP_S)
    assert out.metadata["backend"] == "ppgs"
    assert out.metadata["fallback_used"] is False
    assert out.metadata["label_namespace"] == "arpabet"
    assert seen["n"] == 8000
    assert seen["sample_rate"] == 16000
    assert np.allclose(out.values[:50, PPGS_SILENCE_INDEX], 1.0)
    assert np.allclose(out.values[50:100, 0], 1.0)
    assert np.allclose(out.values[100:, PPGS_SILENCE_INDEX], 1.0)
    assert np.allclose(out.values.sum(axis=1), 1.0)


def test_ppg_posteriors_can_keep_edge_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}

    def fake_infer(wav: np.ndarray, sample_rate: int, **_kwargs: object) -> np.ndarray:
        seen["n"] = int(wav.shape[0])
        n = max(1, int(wav.shape[0]) // 160)
        probs = np.zeros((n, len(PPGS_PHONE_LABELS)), dtype=np.float32)
        probs[:, 0] = 1.0
        return probs

    _patch_ppgs_inference(monkeypatch, fake_infer)
    out = ppg_phone_posteriors(
        _padded_clip(),
        checkpoint="/tmp/fake-ppgs.pt",
        trim_edge_silence=False,
    )
    assert seen["n"] == 24000
    assert out.values.shape[0] == 150
    assert np.allclose(out.values[:, 0], 1.0)


def test_workflow_ppgs_backend_maps_to_articulatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_infer(wav: np.ndarray, sample_rate: int, **_kwargs: object) -> np.ndarray:
        n = max(1, int(wav.shape[0]) // 160)
        probs = np.zeros((n, len(PPGS_PHONE_LABELS)), dtype=np.float32)
        probs[:, PPGS_PHONE_LABELS.index("EY")] = 1.0
        return probs

    _patch_ppgs_inference(monkeypatch, fake_infer)
    result = extract_acoustic_phonetics(
        _padded_clip(),
        posterior_backend="ppgs",
        ppgs_checkpoint="/tmp/fake-ppgs.pt",
        include_uncertainty=False,
    )
    names = list(result.articulatory.coords.get("feature", []))
    ix = {n: i for i, n in enumerate(names)}
    speech_rows = result.posteriorgrams.values[50:100]
    assert np.allclose(speech_rows[:, PPGS_PHONE_LABELS.index("EY")], 1.0)
    assert result.articulatory.values[60, ix["vowel"]] > 0
    assert result.articulatory.values[60, ix["front"]] > 0


def test_every_ppgs_phone_maps_to_an_articulatory_feature() -> None:
    values = np.eye(len(PPGS_PHONE_LABELS), dtype=np.float32)
    post = FeatureSeries(
        values=values,
        times_s=np.arange(len(PPGS_PHONE_LABELS), dtype=np.float64) * 0.01,
        dims=("time", "feature"),
        coords={"feature": list(PPGS_PHONE_LABELS)},
        metadata=extractor_metadata("test.ppgs_inventory"),
        timebase=TimebaseSpec(kind="audio_hop", hop_s=0.01, sampling_rate_hz=100.0),
    )
    art = articulatory_from_posteriors(
        post, include_uncertainty=False, renormalize_posteriors=False
    )
    assert np.all(art.values.sum(axis=1) > 0)


def test_articulatory_mapping_covers_ppgs_diphthongs_and_space_silence() -> None:
    labels = ["EY", "AY", "AW", "OY", " "]
    values = np.eye(len(labels), dtype=np.float32)
    post = FeatureSeries(
        values=values,
        times_s=np.arange(len(labels), dtype=np.float64) * 0.01,
        dims=("time", "feature"),
        coords={"feature": labels},
        metadata=extractor_metadata("test.ppgs_labels"),
        timebase=TimebaseSpec(kind="audio_hop", hop_s=0.01, sampling_rate_hz=100.0),
    )
    art = articulatory_from_posteriors(
        post, include_uncertainty=False, renormalize_posteriors=False
    )
    names = list(art.coords.get("feature", []))
    ix = {n: i for i, n in enumerate(names)}
    assert art.values[0, ix["front"]] > 0
    assert art.values[1, ix["front"]] > 0
    assert art.values[2, ix["back"]] > 0
    assert art.values[3, ix["front"]] > 0
    assert art.values[3, ix["back"]] > 0
    assert art.values[4, ix["silence"]] > 0
    assert art.values[4, ix["vowel"]] == 0
