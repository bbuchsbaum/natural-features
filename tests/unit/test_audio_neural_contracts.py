"""Contract tests for the whole-clip audio embedding helpers.

Both cases here were live failures against transformers 5.x and torch 2.13:

* ``get_audio_features`` used to return the projection tensor and now returns a
  ``ModelOutput``, so ``_numpy`` has to unwrap it before casting.
* CLAP's feature extractor requires 48 kHz and AST's requires 16 kHz. Passing a
  stimulus at the wrong rate produced ``"audio projection failed"``, which points at
  the model rather than at the fixable input.

Neither needs the real models, so these run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.neural import (
    _numpy,
    _require_processor_sample_rate,
    _single_clip_embedding,
)
from natural_features.core.backend_errors import BackendInferenceError


class _FakeTensor:
    """Minimal stand-in for a torch tensor: has .detach().cpu().numpy()."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeModelOutput:
    """Stand-in for BaseModelOutputWithPooling."""

    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeFeatureExtractor:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate


class _FakeProcessor:
    def __init__(self, sampling_rate: int) -> None:
        self.feature_extractor = _FakeFeatureExtractor(sampling_rate)


def _stim(sr: int) -> AudioStimulus:
    return AudioStimulus.from_array(np.zeros(sr, dtype=np.float32), sr_hz=sr)


def test_numpy_passes_through_a_plain_array() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(1, 6)
    assert np.array_equal(_numpy(arr), arr)


def test_numpy_unwraps_a_tensor() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(1, 6)
    assert np.array_equal(_numpy(_FakeTensor(arr)), arr)


@pytest.mark.parametrize("attr", ["audio_embeds", "pooler_output", "last_hidden_state"])
def test_numpy_unwraps_a_model_output(attr: str) -> None:
    """transformers >= 5 returns a ModelOutput where earlier versions gave a tensor."""

    arr = np.arange(4, dtype=np.float32).reshape(1, 4)
    out = _FakeModelOutput(**{attr: _FakeTensor(arr)})
    assert np.array_equal(_numpy(out), arr)


def test_numpy_prefers_the_projection_over_the_hidden_state() -> None:
    """When both are present the pooled projection is the embedding we want."""

    pooled = np.ones((1, 3), dtype=np.float32)
    hidden = np.zeros((1, 9), dtype=np.float32)
    out = _FakeModelOutput(pooler_output=_FakeTensor(pooled), last_hidden_state=_FakeTensor(hidden))
    assert _numpy(out).shape == (1, 3)


def test_single_clip_embedding_accepts_a_model_output() -> None:
    arr = np.arange(5, dtype=np.float32).reshape(1, 5)
    out = _FakeModelOutput(pooler_output=_FakeTensor(arr))
    assert _single_clip_embedding(out, backend="CLAP", dim=None).shape == (1, 5)


def test_sample_rate_guard_accepts_a_matching_rate() -> None:
    _require_processor_sample_rate(_FakeProcessor(48000), _stim(48000), "CLAP")


def test_sample_rate_guard_rejects_a_mismatch_with_an_actionable_message() -> None:
    with pytest.raises(BackendInferenceError) as exc:
        _require_processor_sample_rate(_FakeProcessor(48000), _stim(16000), "CLAP")
    msg = str(exc.value)
    assert "48000" in msg and "16000" in msg
    # The point of the guard is that it names the fix, not just the mismatch.
    assert "audio.resample" in msg


def test_sample_rate_guard_is_silent_when_the_rate_is_unknown() -> None:
    """A processor with no declared rate must not block extraction."""

    _require_processor_sample_rate(object(), _stim(16000), "CLAP")


def test_sample_rate_guard_reads_a_bare_feature_extractor() -> None:
    """AST passes an AutoFeatureExtractor directly, not a processor wrapper."""

    with pytest.raises(BackendInferenceError):
        _require_processor_sample_rate(_FakeFeatureExtractor(16000), _stim(44100), "AST")
