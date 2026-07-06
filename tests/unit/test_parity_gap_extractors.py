from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.registry import Registry
from natural_features.core.stimulus import AudioStimulus, VideoStimulus
from natural_features.features.audio.cochlear import audio_gammatone
from natural_features.features.audio.neural import audio_ast_embeddings, audio_clap_embeddings
from natural_features.features.audio.prosody import audio_pitch, prosody_features
from natural_features.features.language.discourse import discourse_features
from natural_features.features.language.embed import lm_hidden_states
from natural_features.features.language.syntax import syntactic_features
from natural_features.features.preprocess import text_tokenize
from natural_features.features.speech.diarization import speaker_diarization
from natural_features.features.speech.emotion import speech_emotion
from natural_features.features.speech.ssl import hubert_hidden_states
from natural_features.features.speech.vad import neural_vad, speech_vad
from natural_features.features.vision.dct import vision_dct_features
from natural_features.features.vision.motion import optical_flow
from natural_features.features.vision.semantic import vision_semantic_views
from natural_features.workflows.extract_features import available_features, extract_features


GAP_FEATURE_IDS = {
    "audio.ast",
    "audio.clap",
    "audio.gammatone",
    "audio.pitch",
    "audio.prosody",
    "language.discourse",
    "language.syntax",
    "speech.diarization",
    "speech.emotion",
    "speech.hubert",
    "speech.neural_vad",
    "vision.dct",
    "vision.optical_flow",
    "vision.semantic_views",
}

ROOT = Path(__file__).resolve().parents[2]
PARITY_MANIFEST_PATH = ROOT / "tools" / "parity" / "r_public_feature_contracts.yaml"


def _load_parity_features() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(PARITY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    features = payload["features"]
    assert isinstance(features, dict)
    return features


PARITY_FEATURES = _load_parity_features()
R_PUBLIC_FEATURE_IDS = set(PARITY_FEATURES)


def _manifest_set(feature_id: str, key: str) -> set[str]:
    values = PARITY_FEATURES[feature_id].get(key) or []
    return {str(value) for value in values}


def _audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    signal = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 440 * t)
    return AudioStimulus.from_array(signal.astype(np.float32), sr_hz=sr)


def _speechy_audio() -> AudioStimulus:
    sr = 8000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    tone = 0.35 * np.sin(2 * np.pi * 220 * t)
    signal = np.concatenate(
        [
            np.zeros(sr // 4, dtype=np.float32),
            tone.astype(np.float32),
            np.zeros(sr // 4, dtype=np.float32),
        ]
    )
    return AudioStimulus.from_array(signal, sr_hz=sr)


def _video() -> VideoStimulus:
    rng = np.random.default_rng(123)
    frames = rng.integers(0, 255, size=(6, 12, 12, 3), dtype=np.uint8)
    frames[3:] = np.clip(frames[3:] + 30, 0, 255).astype(np.uint8)
    return VideoStimulus.from_array(frames, fps=3.0)


def _words() -> EventSeries:
    return text_tokenize("The quick fox jumps quickly. The fox watches.")


def test_gap_feature_ids_are_registered() -> None:
    registry = Registry.with_builtin_specs()
    registered = {spec.name for spec in registry.list()}

    assert GAP_FEATURE_IDS <= registered


def test_r_public_feature_ids_are_registered() -> None:
    registry = Registry.with_builtin_specs()
    registered = {spec.name for spec in registry.list()}

    assert R_PUBLIC_FEATURE_IDS <= registered


def test_r_public_feature_bundles_match_catalog_contract() -> None:
    entries = {entry.feature_id: entry for entry in available_features(budget="all")}

    for feature_id in R_PUBLIC_FEATURE_IDS:
        assert set(entries[feature_id].bundles) == _manifest_set(feature_id, "bundles")


def test_r_public_defaults_are_present_with_only_allowed_python_extras() -> None:
    entries = {entry.feature_id: entry for entry in available_features(budget="all")}

    for feature_id in R_PUBLIC_FEATURE_IDS:
        actual = set(entries[feature_id].default_params)
        expected = _manifest_set(feature_id, "required_default_keys")
        allowed_extra = _manifest_set(feature_id, "allowed_python_default_extras")
        assert expected <= actual, feature_id
        assert actual - expected <= allowed_extra, feature_id


def test_parity_manifest_audit_passes_against_python_catalog() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "parity" / "check_r_catalog_parity.py"),
            "--no-r-compare",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "parity-check: OK" in result.stdout


def test_asr_and_vad_output_contracts_are_explicit() -> None:
    registry = Registry.with_builtin_specs()
    words_spec = registry.get("speech.words")
    whisper_spec = registry.get("speech.asr.whisper")
    chunked_spec = registry.get("speech.asr.whisper_chunked")
    neural_vad_entry = {entry.feature_id: entry for entry in available_features(budget="all")}["speech.neural_vad"]

    for spec in [words_spec, whisper_spec, chunked_spec]:
        assert set(spec.outputs) == {"segments", "words", "qc"}
        assert spec.outputs["qc"] == {"schema": "dict", "kind": "qc"}
    assert neural_vad_entry.output_schema == "FeatureSeries/v1"


def test_gap_feature_ids_are_discoverable_by_workflow_catalog() -> None:
    assert {"audio.ast", "audio.clap", "audio.gammatone", "audio.pitch", "audio.prosody"} <= {
        entry.feature_id for entry in available_features(modality="audio", budget="allow_python")
    }
    assert {"language.discourse", "language.syntax"} <= {
        entry.feature_id for entry in available_features(modality="words", budget="allow_python")
    }
    assert {"speech.diarization", "speech.emotion", "speech.hubert", "speech.neural_vad"} <= {
        entry.feature_id for entry in available_features(modality="audio", budget="all")
    }
    assert {"vision.dct", "vision.optical_flow", "vision.semantic_views"} <= {
        entry.feature_id for entry in available_features(modality="video", budget="all")
    }


def test_audio_gap_features_return_feature_series() -> None:
    audio = _audio()

    gammatone = audio_gammatone(audio, n_channels=8)
    pitch = audio_pitch(audio)
    prosody = prosody_features(audio)
    clap = audio_clap_embeddings(
        audio,
        model="natural-features-test/missing-clap",
        stride_s=0.25,
        dim=8,
        strict_dependency=False,
    )
    ast = audio_ast_embeddings(
        audio,
        model="natural-features-test/missing-ast",
        stride_s=0.25,
        dim=8,
        strict_dependency=False,
    )

    assert isinstance(gammatone, FeatureSeries)
    assert gammatone.values.shape[1] == 8
    assert pitch.values.shape[1] == 2
    assert prosody.values.shape[1] == 6
    assert clap.values.shape[1] == 8
    assert ast.values.shape[1] == 8
    assert clap.metadata["extractor_name"] == "audio.clap"
    assert ast.metadata["extractor_name"] == "audio.ast"
    assert clap.metadata["fallback_used"] is True
    assert ast.metadata["fallback_used"] is True


def test_language_gap_features_return_word_aligned_features() -> None:
    words = _words()

    discourse = discourse_features(words)
    hidden = lm_hidden_states(
        words,
        model="natural-features-test/missing-causal-lm",
        layers=[1],
        strict_dependency=False,
    )
    syntax = syntactic_features(words, model="natural_features_missing_spacy_model", strict_dependency=False)

    assert discourse.values.shape == (len(words), 5)
    assert hidden.values.shape[:2] == (len(words), 1)
    assert hidden.metadata["extractor_name"] == "language.hidden_states"
    assert hidden.metadata["fallback_used"] is True
    assert syntax.values.shape == (len(words), 8)
    assert discourse.metadata["extractor_name"] == "language.discourse"
    assert syntax.metadata["extractor_name"] == "language.syntax"
    assert syntax.metadata["fallback_used"] is True


def test_speech_gap_features_return_expected_contracts() -> None:
    audio = _audio()
    speech_audio = _speechy_audio()

    hubert = hubert_hidden_states(
        audio,
        model="natural-features-test/missing-hubert",
        layers=[1],
        strict_dependency=False,
    )
    vad = neural_vad(audio, strict_dependency=False)
    vad_events = speech_vad(speech_audio)
    diarization = speaker_diarization(audio, strict_dependency=False)
    emotion = speech_emotion(audio, strict_dependency=False)

    assert isinstance(hubert, FeatureSeries)
    assert hubert.values.ndim == 3
    assert hubert.values.shape[1] == 1
    assert hubert.metadata["extractor_name"] == "speech.hubert"
    assert vad.values.shape[1] == 1
    assert vad.metadata["extractor_name"] == "speech.neural_vad"
    assert isinstance(vad_events, EventSeries)
    assert vad_events.metadata["extractor_name"] == "speech.vad"
    assert len(vad_events) >= 1
    assert isinstance(diarization, TrackSeries)
    assert diarization.values.shape[1:] == (1, 1)
    assert diarization.metadata["extractor_name"] == "speech.diarization"
    assert emotion.values.shape[1] == 4
    assert emotion.metadata["extractor_name"] == "speech.emotion"


def test_vision_gap_features_return_expected_contracts() -> None:
    video = _video()

    dct = vision_dct_features(video, k=6, size=8)
    flow = optical_flow(video, strict_dependency=False)
    semantic = vision_semantic_views(video, strict_dependency=False)

    assert dct.values.shape == (len(video.frame_times_s), 6)
    assert dct.metadata["extractor_name"] == "vision.dct"
    assert flow.values.shape == (len(video.frame_times_s), 4)
    assert flow.metadata["extractor_name"] == "vision.optical_flow"
    assert len(semantic) == len(video.frame_times_s)
    assert semantic.metadata["extractor_name"] == "vision.semantic_views"
    assert set(semantic.label).issubset({"structured_scene", "colorful_scene", "bright_scene", "dark_scene"})


def test_extract_features_can_execute_new_public_gap_ids() -> None:
    audio_result = extract_features(
        _audio(),
        features=["audio.gammatone", "audio.pitch", "audio.prosody"],
        feature_params={"audio.gammatone": {"n_channels": 4}},
    )
    text_result = extract_features(
        "one two two",
        features=["text.tokenize", "language.discourse", "language.syntax"],
        budget="allow_python",
    )
    video_result = extract_features(_video(), features=["vision.dct", "vision.semantic_views"], budget="all")

    assert audio_result.features["audio.gammatone"].values.shape[1] == 4
    assert text_result.features["language.discourse"].values.shape[0] == 3
    assert text_result.features["language.syntax"].values.shape[0] == 3
    assert video_result.features["vision.dct"].values.shape[1] == 64
    assert len(video_result.features["vision.semantic_views"]) == len(_video().frame_times_s)


def test_extract_features_can_execute_r_public_alias_ids() -> None:
    audio = _speechy_audio()
    video = _video()

    audio_result = extract_features(
        audio,
        features=["audio.egemaps", "speech.vad", "speech.wavlm", "speech.ctc"],
        budget="allow_python",
        feature_params={
            "speech.wavlm": {"layers": [1]},
            "speech.ctc": {"model": "natural-features-test/missing-ctc"},
        },
    )
    text_result = extract_features(
        "one two two",
        features=["text.tokenize", "language.hidden_states"],
        budget="allow_python",
        feature_params={
            "language.hidden_states": {
                "model": "natural-features-test/missing-causal-lm",
                "layers": [1],
            }
        },
    )
    video_result = extract_features(
        video,
        features=["vision.motion_energy", "vision.social_proxies"],
        budget="allow_python",
    )

    assert audio_result.features["audio.egemaps"].values.ndim == 2
    assert isinstance(audio_result.features["speech.vad"], EventSeries)
    assert audio_result.features["speech.wavlm"].values.ndim == 3
    assert audio_result.features["speech.ctc"].values.ndim == 2
    assert text_result.features["language.hidden_states"].values.shape[:2] == (3, 1)
    assert video_result.features["vision.motion_energy"].values.ndim == 2
    assert video_result.features["vision.social_proxies"].values.shape[1] == 3
