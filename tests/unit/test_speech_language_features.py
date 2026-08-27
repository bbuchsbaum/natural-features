from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.recipe import execute_recipe
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.timebase import TimebaseSpec
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.align import alignment_qc, whisperx_align
from natural_features.features.speech.asr import whisper_transcribe
from natural_features.features.speech.phonology import (
    acoustic_phone_posteriors,
    articulatory_features,
    articulatory_from_posteriors,
    ctc_phone_posteriors,
    phoneme_posteriorgrams,
)
from natural_features.features.language.embed import bert_word_embeddings
from natural_features.features.speech.ssl import wavlm_hidden_states


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioStimulus.from_array(x, sr_hz=sr)


def test_explicit_transcript_and_passthrough_alignment_qc() -> None:
    a = _audio()
    asr = whisper_transcribe(
        a,
        transcript_text="hello explicit transcript",
        device="cpu",
    )
    assert "words" in asr and "segments" in asr and "qc" in asr
    qc = alignment_qc(asr["words"])
    assert "low_confidence_words" in qc
    aligned = whisperx_align(a, asr["words"], backend="none")
    assert "words" in aligned and "qc" in aligned


def test_named_neural_methods_reject_surrogate_execution() -> None:
    a = _audio()
    asr = whisper_transcribe(a, transcript_text="hello world")
    words = asr["words"]
    post = phoneme_posteriorgrams(a)
    art = articulatory_features(words)
    assert post.values.shape[0] == len(post.times_s)
    assert art.values.shape[1] == 5
    with pytest.raises(ValueError, match="proxy and surrogate"):
        wavlm_hidden_states(a, execution_mode="fallback")
    with pytest.raises(ValueError, match="proxy and surrogate"):
        bert_word_embeddings(words, execution_mode="fallback")


def test_option1_posterior_to_articulatory_pipeline() -> None:
    a = _audio()
    post = acoustic_phone_posteriors(a, hop_s=0.02)
    art = articulatory_from_posteriors(post, include_uncertainty=True)
    names = list(art.coords.get("feature", []))
    assert post.values.ndim == 2
    assert art.values.ndim == 2
    assert art.values.shape[0] == post.values.shape[0]
    assert "bilabial" in names
    assert "alveolar" in names
    assert "posterior_entropy" in names
    assert np.all(np.isfinite(art.values))


def test_ctc_posteriors_reject_fallback_and_fail_loudly() -> None:
    a = _audio()
    with pytest.raises(ValueError, match="proxy and surrogate"):
        ctc_phone_posteriors(
            a,
            execution_mode="fallback",
            local_files_only=True,
        )

    # Force an error path in strict mode with an invalid model id.
    with pytest.raises(RuntimeError):
        ctc_phone_posteriors(
            a,
            model="__missing__/__missing__",
            local_files_only=True,
            strict_dependency=True,
        )


def test_articulatory_mapping_handles_multichar_ipa_tokens() -> None:
    labels = ["aɪ", "dʒ", "n̩", "iː", "t", "silence"]
    values = np.eye(len(labels), dtype=np.float32)
    post = FeatureSeries(
        values=values,
        times_s=np.arange(len(labels), dtype=np.float64) * 0.1,
        dims=("time", "feature"),
        coords={"feature": labels},
        metadata=extractor_metadata("test.ipa_post"),
        timebase=TimebaseSpec(kind="audio_hop", hop_s=0.1, sampling_rate_hz=10.0),
    )
    art = articulatory_from_posteriors(post, include_uncertainty=False, renormalize_posteriors=False)
    names = list(art.coords.get("feature", []))
    ix = {n: i for i, n in enumerate(names)}
    # aɪ should map as vowel + front.
    assert art.values[0, ix["vowel"]] > 0
    assert art.values[0, ix["front"]] > 0
    # dʒ should map as affricate + consonant.
    assert art.values[1, ix["affricate"]] > 0
    assert art.values[1, ix["consonant"]] > 0
    # n̩ should map to nasal + voiced + sonorant.
    assert art.values[2, ix["nasal"]] > 0
    assert art.values[2, ix["voiced"]] > 0
    assert art.values[2, ix["sonorant"]] > 0
    # silence token should map to silence feature.
    assert art.values[5, ix["silence"]] > 0


def test_articulatory_mapping_does_not_mix_alphabets() -> None:
    labels = ["SH", "silence"]
    values = np.eye(len(labels), dtype=np.float32)
    post = FeatureSeries(
        values=values,
        times_s=np.arange(len(labels), dtype=np.float64) * 0.1,
        dims=("time", "feature"),
        coords={"feature": labels},
        metadata=extractor_metadata("test.phone_labels"),
        timebase=TimebaseSpec(kind="audio_hop", hop_s=0.1, sampling_rate_hz=10.0),
    )
    art = articulatory_from_posteriors(post, include_uncertainty=False, renormalize_posteriors=False)
    names = list(art.coords.get("feature", []))
    ix = {n: i for i, n in enumerate(names)}
    # SH should be postalveolar fricative only, not spurious alveolar/glottal composition.
    assert art.values[0, ix["postalveolar"]] > 0
    assert art.values[0, ix["alveolar"]] == 0
    assert art.values[0, ix["glottal"]] == 0
    # "silence" should not decompose into IPA letters.
    assert art.values[1, ix["silence"]] > 0
    assert art.values[1, ix["vowel"]] == 0


def test_recipe_refs_for_speech_language_chain() -> None:
    a = _audio()
    reg = Registry.with_builtin_specs()
    recipe = {
        "features": [
            {
                "id": "asr",
                "use": "speech.asr.whisper",
                "params": {"transcript_text": "hello world"},
            },
            {
                "id": "art",
                "use": "speech.articulatory.features",
                "inputs": {"words": "ref: asr.words"},
            },
            {
                "id": "post",
                "use": "speech.phonology.acoustic_posteriors",
            },
            {
                "id": "art_post",
                "use": "speech.articulatory.from_posteriors",
                "inputs": {"posteriors": "ref: post.default"},
            },
        ]
    }
    out = execute_recipe(recipe, registry=reg, inputs={"audio": a})
    assert "asr" in out.steps and "art" in out.steps and "art_post" in out.steps
