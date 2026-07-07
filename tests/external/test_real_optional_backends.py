from __future__ import annotations

import os

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries, FeatureSeries, TrackSeries
from natural_features.core.stimulus import AudioStimulus, ImageStimulus, VideoStimulus
from natural_features.features.audio.neural import audio_ast_embeddings, audio_clap_embeddings
from natural_features.features.preprocess import image_ocr, text_tokenize
from natural_features.features.language.syntax import syntactic_features
from natural_features.features.speech.diarization import speaker_diarization
from natural_features.features.speech.emotion import speech_emotion
from natural_features.features.speech.ssl import hubert_hidden_states, wavlm_hidden_states
from natural_features.features.speech.vad import neural_vad
from natural_features.features.vision.neural import vision_clip_embeddings, vision_dino_embeddings
from natural_features.features.vision.semantic import vision_semantic_views


pytestmark = [pytest.mark.external, pytest.mark.nightly]


def _env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"Set {name} to a local model id/path to run this real backend test.")
    return value


def _audio() -> AudioStimulus:
    sr = 16000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    signal = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
    return AudioStimulus.from_array(signal.astype(np.float32), sr_hz=sr)


def _image() -> ImageStimulus:
    h, w = 32, 32
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    rgb = np.stack([np.tile(x, (h, 1)), np.tile(y, (1, w)), np.full((h, w), 0.3, dtype=np.float32)], axis=-1)
    return ImageStimulus.from_array(rgb)


def _video() -> VideoStimulus:
    frames = np.stack([_image().image, np.flipud(_image().image)], axis=0)
    return VideoStimulus.from_array(frames.astype(np.float32), fps=2.0)


def _assert_real_feature_series(out: FeatureSeries, *, feature_id: str) -> None:
    assert isinstance(out, FeatureSeries)
    assert out.metadata["extractor_name"] == feature_id
    assert out.metadata["fallback_used"] is False
    assert out.values.shape[0] == len(out.times_s)
    assert out.values.size > 0
    assert np.isfinite(out.values).all()


def test_real_ast_backend_local_model_contract() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_AST_MODEL")

    out = audio_ast_embeddings(
        _audio(),
        model=model,
        dim=8,
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="audio.ast")
    assert out.values.shape[1] == 8


def test_real_clap_backend_local_model_contract() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_CLAP_MODEL")

    out = audio_clap_embeddings(
        _audio(),
        model=model,
        dim=8,
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="audio.clap")
    assert out.values.shape[1] == 8


def test_real_hubert_backend_local_model_contract() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_HUBERT_MODEL")

    out = hubert_hidden_states(
        _audio(),
        model=model,
        layers=[1],
        pooling="mean",
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="speech.hubert")
    assert out.values.shape[1:] == (1, 1)


def test_real_wavlm_backend_local_model_contract() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_WAVLM_MODEL")

    out = wavlm_hidden_states(
        _audio(),
        model=model,
        layers=[1],
        pooling="mean",
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="speech.ssl.wavlm")
    assert out.values.shape[1:] == (1, 1)


def test_real_speech_emotion_backend_local_model_contract() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_SPEECH_EMOTION_MODEL")

    out = speech_emotion(
        _audio(),
        model=model,
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="speech.emotion")
    assert out.values.shape[0] == 1
    assert out.values.shape[1] >= 2


def test_real_neural_vad_backend_contract() -> None:
    pytest.importorskip("torch")
    model = _env_required("NF_TEST_NEURAL_VAD_MODEL")
    if model in {"silero", "silero_vad", "package"}:
        pytest.importorskip("silero_vad")

    out = neural_vad(
        _audio(),
        model=model,
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="speech.neural_vad")
    assert out.values.shape[1] == 1
    assert out.metadata["backend"].startswith("silero")


def test_real_pyannote_diarization_backend_local_pipeline_contract() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyannote.audio")
    model = _env_required("NF_TEST_PYANNOTE_DIARIZATION_MODEL")

    out = speaker_diarization(
        _audio(),
        model=model,
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    assert isinstance(out, TrackSeries)
    assert out.metadata["extractor_name"] == "speech.diarization"
    assert out.metadata["fallback_used"] is False
    assert out.metadata["backend"] == "pyannote"
    assert out.values.shape[0] == len(out.times_s)
    assert out.values.ndim == 3
    assert out.values.shape[2] == 1


def test_real_clip_backend_local_model_contract() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_CLIP_MODEL")

    out = vision_clip_embeddings(
        _image(),
        model=model,
        dim=8,
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="vision.clip")
    assert out.values.shape == (1, 8)


def test_real_dino_backend_local_model_contract() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_DINO_MODEL")

    out = vision_dino_embeddings(
        _video(),
        model=model,
        layers=[1],
        pooling="mean",
        dim=8,
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    _assert_real_feature_series(out, feature_id="vision.dino")
    assert out.values.shape == (2, 8)


def test_real_semantic_views_backend_local_model_contract() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = _env_required("NF_TEST_SEMANTIC_VIEWS_MODEL")

    out = vision_semantic_views(
        _video(),
        model=model,
        labels=["gradient_frame", "flipped_gradient_frame"],
        local_files_only=True,
        execution_mode="strict",
        strict_dependency=True,
    )

    assert isinstance(out, EventSeries)
    assert out.metadata["extractor_name"] == "vision.semantic_views"
    assert out.metadata["fallback_used"] is False
    assert out.metadata["backend"] == "transformers_clip_zero_shot"
    assert len(out) == 2
    assert out.confidence is not None
    assert np.isfinite(out.confidence).all()


def test_real_spacy_syntax_backend_contract() -> None:
    pytest.importorskip("spacy")
    model = _env_required("NF_TEST_SPACY_MODEL")
    words = text_tokenize("The quiet room opens slowly.")

    out = syntactic_features(words, model=model, execution_mode="strict", strict_dependency=True)

    _assert_real_feature_series(out, feature_id="language.syntax")
    assert out.metadata["backend"] == "spacy"
    assert len(out.metadata["pos_tags"]) == len(words)
    assert out.values.shape == (len(words), 8)


def test_real_tesseract_ocr_backend_contract() -> None:
    if os.environ.get("NF_TEST_ENABLE_TESSERACT_OCR", "").strip() != "1":
        pytest.skip("Set NF_TEST_ENABLE_TESSERACT_OCR=1 to run the real OCR backend test.")
    pil_image = pytest.importorskip("PIL.Image")
    pytest.importorskip("pytesseract")

    image = pil_image.new("RGB", (180, 64), "white")
    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((12, 20), "TEST", fill="black", font=font)
    except Exception as exc:  # pragma: no cover - defensive fixture construction
        pytest.skip(f"PIL text fixture unavailable: {exc}")
    stim = ImageStimulus.from_array(np.asarray(image).astype(np.float32) / 255.0, duration_s=1.0)

    try:
        out = image_ocr(stim, min_confidence=0.0, execution_mode="strict", strict_dependency=True)
    except RuntimeError as exc:
        pytest.skip(f"real OCR backend unavailable: {exc}")

    assert isinstance(out, EventSeries)
    assert out.metadata["extractor_name"] == "image.ocr"
    assert out.metadata["fallback_used"] is False
    assert out.metadata["backend"] == "pytesseract"
    raw_labels = out.label if out.label is not None else []
    labels = {str(label).upper() for label in raw_labels}
    assert "TEST" in labels
