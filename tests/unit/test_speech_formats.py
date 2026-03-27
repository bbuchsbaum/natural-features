from __future__ import annotations

import numpy as np

from natural_features.core.feature_types import EventSeries
from natural_features.features.common import extractor_metadata
from natural_features.features.speech.formats import read_ctm, read_textgrid, write_ctm, write_textgrid


def _words() -> EventSeries:
    return EventSeries(
        onset_s=np.array([0.10, 0.40, 0.80], dtype=np.float64),
        offset_s=np.array([0.30, 0.60, 1.00], dtype=np.float64),
        label=np.array(["hello", "world", "again"], dtype=object),
        confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        metadata=extractor_metadata("test.words"),
    )


def test_ctm_roundtrip_preserves_tokens_and_times(tmp_path) -> None:
    words = _words()
    p = write_ctm(words, tmp_path / "words.ctm", utterance_id="utt1")
    loaded = read_ctm(p)
    assert list(loaded.label) == list(words.label)
    np.testing.assert_allclose(loaded.onset_s, words.onset_s, atol=1e-6)
    np.testing.assert_allclose(loaded.offset_s, words.offset_s, atol=1e-6)


def test_textgrid_roundtrip_preserves_tokens_and_times(tmp_path) -> None:
    words = _words()
    p = write_textgrid(words, tmp_path / "words.TextGrid")
    loaded = read_textgrid(p)
    assert list(loaded.label) == list(words.label)
    np.testing.assert_allclose(loaded.onset_s, words.onset_s, atol=1e-6)
    np.testing.assert_allclose(loaded.offset_s, words.offset_s, atol=1e-6)
