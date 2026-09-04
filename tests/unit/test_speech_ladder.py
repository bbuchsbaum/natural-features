from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.backend_errors import BackendDependencyError
from natural_features.core.feature_types import EventSeries, FeatureSeries
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.envelope import audio_envelope
from natural_features.features.audio.modulation import spectrotemporal_modulation
from natural_features.features.audio.phonetic import audio_formants, audio_harmonicity
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.gestures import (
    articulatory_dynamics,
    articulatory_gestures,
)
from natural_features.features.speech.phones import phones_from_textgrid
from natural_features.features.speech.phonetic_cues import phonetic_cues
from natural_features.features.speech.phonology import (
    DEFAULT_ARTICULATORY_FEATURES,
    DEFAULT_DISTINCTIVE_FEATURES,
    acoustic_phone_posteriors,
    articulatory_from_posteriors,
    distinctive_from_phoneme_events,
    distinctive_from_posteriors,
    phoneme_event_series,
)
from natural_features.features.speech.sparc import SPARC_EMA_LABELS, sparc_articulatory
from natural_features.features.stats.residualize import residualize_feature_series
from natural_features.workflows.speech_ladder import extract_speech_ladder


def _tone(sr: int = 8000, seconds: float = 1.0, freq: float = 220.0) -> AudioStimulus:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return AudioStimulus.from_array(
        (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr_hz=sr
    )


def _series(
    values: np.ndarray, hop_s: float = 0.01, name: str = "test.feat"
) -> FeatureSeries:
    times = np.arange(values.shape[0], dtype=np.float64) * hop_s
    return FeatureSeries(
        values=values.astype(np.float32),
        times_s=times,
        dims=("time", "feature"),
        coords={"feature": [f"c{i}" for i in range(values.shape[1])]},
        metadata=extractor_metadata(name, params={}),
    )


def _phones() -> EventSeries:
    return phoneme_event_series(
        onset_s=np.array([0.10, 0.20, 0.35], dtype=np.float64),
        offset_s=np.array([0.20, 0.35, 0.50], dtype=np.float64),
        labels=np.array(["P", "AE", "T"], dtype=object),
        label_namespace="arpabet",
        namespace_version="test",
        source_word_alignment_id="test",
    )


def test_residualize_is_orthogonal_to_predictors() -> None:
    n = 80
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
    pred = _series(x, name="pred")
    target = _series(2.0 * x + 0.5, name="target")
    resid = residualize_feature_series(target, pred, hop_s=0.01)
    design = np.concatenate([np.ones((n, 1)), x], axis=1)
    proj = design.T @ resid.values.astype(np.float64)
    np.testing.assert_allclose(proj, 0.0, atol=1e-4)
    assert resid.metadata["backend"] == "ols"
    assert resid.metadata["analysis_hop_s"] == 0.01


def test_envelope_channels_and_onset() -> None:
    env = audio_envelope(_tone(), hop_s=0.01)
    assert list(env.coords["feature"]) == ["envelope", "rms", "delta", "onset"]
    assert env.values.shape[1] == 4
    assert env.metadata["backend"] == "hilbert_rms"
    assert np.all(env.values[:, 3] >= 0.0)


def test_modulation_am_tone_loads_syllabic_band() -> None:
    sr = 8000
    t = np.arange(int(sr * 2.0), dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * 200.0 * t)
    am = 0.5 * (1.0 + np.sin(2 * np.pi * 5.0 * t))
    audio = AudioStimulus.from_array((0.4 * am * carrier).astype(np.float32), sr_hz=sr)
    mod = spectrotemporal_modulation(audio, hop_s=0.01)
    assert mod.metadata["backend"] == "stft_rate_scale"
    names = list(mod.coords["feature"])
    syllabic = [i for i, name in enumerate(names) if name.startswith("rate_4_8")]
    phonetic = [i for i, name in enumerate(names) if name.startswith("rate_16_32")]
    assert syllabic and phonetic
    assert float(mod.values[:, syllabic].mean()) > float(mod.values[:, phonetic].mean())


def test_formants_and_hnr_shapes() -> None:
    audio = _tone(seconds=0.6)
    formants = audio_formants(audio, hop_s=0.01, voicing_gate=False)
    hnr = audio_harmonicity(audio, hop_s=0.01)
    assert formants.values.shape[1] == 9
    assert formants.metadata["backend"] == "lpc_autocorr"
    assert list(formants.coords["feature"])[:3] == ["f1", "f2", "f3"]
    assert hnr.values.shape[1] == 1
    assert np.all(np.isfinite(formants.values))


def test_distinctive_features_do_not_widen_old_articulatory_set() -> None:
    audio = _tone()
    post = acoustic_phone_posteriors(audio, hop_s=0.02)
    art = articulatory_from_posteriors(post, include_uncertainty=False)
    dist = distinctive_from_posteriors(post, include_uncertainty=False)
    assert list(art.coords["feature"]) == list(DEFAULT_ARTICULATORY_FEATURES)
    assert "continuant" in dist.coords["feature"]
    assert "strident" in dist.coords["feature"]
    assert (
        dist.metadata["extractor_name"]
        == "speech.phonology.distinctive_from_posteriors"
    )
    # The context-free posterior path must not advertise aspiration.
    assert "aspirated" not in DEFAULT_DISTINCTIVE_FEATURES
    assert "aspirated" not in dist.coords["feature"]
    phones = _phones()
    ev = distinctive_from_phoneme_events(phones, include_confidence=False)
    names = list(ev.coords["feature"])
    assert names == list(DEFAULT_DISTINCTIVE_FEATURES) + ["aspirated"]
    ix = {name: i for i, name in enumerate(names)}
    assert ev.values[0, ix["stop"]] == 1.0
    assert ev.values[0, ix["voiced"]] == 0.0
    assert ev.values[1, ix["continuant"]] == 1.0
    assert ev.values[1, ix["low"]] == 1.0


def test_contextual_aspiration_is_allophonic() -> None:
    # "pat": initial P released into a vowel is aspirated; final T is not.
    pat = distinctive_from_phoneme_events(_phones(), include_confidence=False)
    ix = {name: i for i, name in enumerate(pat.coords["feature"])}
    assert pat.values[0, ix["aspirated"]] == 1.0
    assert pat.values[2, ix["aspirated"]] == 0.0
    # "spin": S blocks aspiration of P.
    spin = phoneme_event_series(
        onset_s=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        offset_s=np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        labels=np.array(["S", "P", "IH", "N"], dtype=object),
        label_namespace="arpabet",
        namespace_version="test",
        source_word_alignment_id="test",
    )
    vals = distinctive_from_phoneme_events(spin, include_confidence=False)
    jx = {name: i for i, name in enumerate(vals.coords["feature"])}
    assert vals.values[1, jx["aspirated"]] == 0.0


def test_textgrid_tier_split(tmp_path) -> None:
    text = """
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0.10
            xmax = 0.50
            text = "cat"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 3
        intervals [1]:
            xmin = 0.10
            xmax = 0.20
            text = "K"
        intervals [2]:
            xmin = 0.20
            xmax = 0.35
            text = "AE"
        intervals [3]:
            xmin = 0.35
            xmax = 0.50
            text = "T"
"""
    path = tmp_path / "clip.TextGrid"
    path.write_text(text, encoding="utf-8")
    phones = phones_from_textgrid(path)
    assert list(phones.label) == ["K", "AE", "T"]
    from natural_features.features.speech.formats import read_textgrid

    words = read_textgrid(path, tier="words")
    assert list(words.label) == ["cat"]


def test_gestures_dynamics_and_cues() -> None:
    audio = _tone(seconds=0.7)
    phones = _phones()
    gestures = articulatory_gestures(phones, hop_s=0.01)
    assert list(gestures.coords["feature"]) == [
        "lips",
        "tongue_tip",
        "tongue_body",
        "velum",
        "glottis",
    ]
    assert gestures.values[:, 0].max() > 0.5
    assert gestures.metadata["backend"] == "canonical_raised_cosine"
    dyn = articulatory_dynamics(gestures)
    assert "speed" in dyn.coords["feature"]
    assert "vel_lips" in dyn.coords["feature"]
    assert "effort" in dyn.coords["feature"]
    assert "overlap" in dyn.coords["feature"]
    names = list(dyn.coords["feature"])
    overlap = dyn.values[:, names.index("overlap")]
    assert float(overlap.max()) >= 1.0
    cues = phonetic_cues(audio, phones, hop_s=0.01)
    assert list(cues.coords["feature"]) == [
        "vot",
        "burst_centroid",
        "frication_duration",
        "closure_duration",
    ]
    assert float(cues.values[:, 3].max()) > 0.0


def test_gestures_acoustic_gain_departs_from_canonical() -> None:
    audio = _tone(seconds=0.7)
    phones = _phones()
    canonical = articulatory_gestures(phones, hop_s=0.01)
    gained = articulatory_gestures(phones, hop_s=0.01, stimulus=audio)
    assert gained.metadata["backend"] == "canonical_acoustic_gain"
    assert gained.values.shape == canonical.values.shape
    # Gains only attenuate, and the gained series is not a rescaled copy.
    assert np.all(gained.values <= canonical.values + 1e-6)
    assert float(np.abs(gained.values - canonical.values).max()) > 0.0


def test_syllable_onc_roles_and_boundaries() -> None:
    from natural_features.features.speech.syllables import syllable_onc

    phones = _phones()  # P AE T -> onset nucleus coda
    onc = syllable_onc(phones, hop_s=0.01)
    assert list(onc.coords["feature"]) == ["syll_onset", "syll_nucleus", "syll_coda"]
    times = onc.times_s
    ons, nuc, cod = onc.values[:, 0], onc.values[:, 1], onc.values[:, 2]
    mid_p = (times >= 0.10) & (times < 0.20)
    mid_ae = (times >= 0.20) & (times < 0.35)
    mid_t = (times >= 0.35) & (times < 0.50)
    assert ons[mid_p].max() == 1.0 and nuc[mid_p].max() == 0.0
    assert nuc[mid_ae].max() == 1.0
    assert cod[mid_t].max() == 1.0 and ons[mid_t].max() == 0.0
    # Intervocalic consonant resyllabifies as onset of the next syllable: "AE T AA".
    vcv = phoneme_event_series(
        onset_s=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        offset_s=np.array([0.2, 0.3, 0.4], dtype=np.float64),
        labels=np.array(["AE", "T", "AA"], dtype=object),
        label_namespace="arpabet",
        namespace_version="test",
        source_word_alignment_id="test",
    )
    onc2 = syllable_onc(vcv, hop_s=0.01)
    mid_t2 = (onc2.times_s >= 0.2) & (onc2.times_s < 0.3)
    assert onc2.values[mid_t2, 0].max() == 1.0
    assert onc2.values[mid_t2, 2].max() == 0.0


def test_sparc_monkeypatch_and_missing_dependency(monkeypatch) -> None:
    import builtins
    import sys
    import types

    class _Coder:
        def encode(self, path: str):
            return {"ema": np.zeros((20, 12), dtype=np.float32)}

    fake = types.ModuleType("sparc")
    fake.load_model = lambda *args, **kwargs: _Coder()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sparc", fake)
    out = sparc_articulatory(_tone(seconds=0.4), local_files_only=True)
    assert out.values.shape == (20, 12)
    assert list(out.coords["feature"]) == SPARC_EMA_LABELS
    assert out.metadata["backend"] == "sparc_template_ema"

    orig_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "sparc" or str(name).startswith("sparc."):
            raise ImportError("speech-articulatory-coding is not installed")
        return orig_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sparc", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(BackendDependencyError):
        sparc_articulatory(_tone(seconds=0.2), local_files_only=True)


def test_extract_speech_ladder_cheap_backends() -> None:
    phones = _phones()
    bundle = extract_speech_ladder(
        _tone(seconds=0.6),
        posterior_backend="acoustic",
        include_sparc=False,
        include_residuals=True,
        phones=phones,
    )
    assert {
        "a1",
        "a2",
        "a3",
        "p_posteriors",
        "p_features",
        "g_gestures",
        "m_dynamics",
        "m_syllables",
    } <= set(bundle.features)
    assert bundle.features["g_gestures"].metadata["backend"] == (
        "canonical_acoustic_gain"
    )
    assert "a2|a1" in bundle.features
    assert "p|a" in bundle.features
    assert "g_ema" not in bundle.features
    assert bundle.features["p_features"].values.shape[1] >= len(
        DEFAULT_DISTINCTIVE_FEATURES
    )


def test_ladder_extractors_are_registered() -> None:
    names = {spec.name for spec in Registry.with_builtin_specs().list()}
    assert "audio.envelope" in names
    assert "speech.articulatory.dynamics" in names
    assert "speech.phones.mfa" in names
    assert "speech.syllables.onc" in names
