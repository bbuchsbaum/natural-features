"""Source identity and clock contracts, independent of optional model weights."""

import sys
import types
from contextlib import nullcontext

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio.vocals import source_pitch_crepe, separate_vocals


def stimulus(x=None, rate=16000, offset=0, clock="audio:reference"):
    return AudioStimulus(
        np.zeros(1600) if x is None else x, rate, start_offset_s=offset, clock=clock
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"source_role": "", "source_provenance": {"ref": "known"}}, "source_role"),
        ({"source_role": "reference", "source_provenance": {}}, "source_role"),
        ({"model": "unknown"}, "full or tiny"),
        ({"hop_s": float("nan")}, "hop_s"),
        ({"fmax": 3000}, "2006"),
    ],
)
def test_pitch_rejects_ambiguous_inputs(kwargs, match):
    opts = dict(source_role="reference", source_provenance={"ref": "known"})
    opts.update(kwargs)
    with pytest.raises(ValueError, match=match):
        source_pitch_crepe(stimulus(), **opts)


def test_pitch_requires_explicit_resampling():
    with pytest.raises(ValueError, match="resample"):
        source_pitch_crepe(
            stimulus(rate=44100),
            source_role="reference",
            source_provenance={"ref": "x"},
        )


def test_pitch_rejects_short_and_stereo():
    for x, match in [(np.zeros(1023), "1024"), (np.zeros((1600, 2)), "downmix")]:
        with pytest.raises(ValueError, match=match):
            source_pitch_crepe(
                stimulus(x), source_role="reference", source_provenance={"ref": "x"}
            )


class Tensor:
    def __init__(self, x):
        self.x = np.asarray(x)

    def __getitem__(self, key):
        return Tensor(self.x[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.x


def test_native_clock_no_padding_offset_and_source_provenance(monkeypatch, tmp_path):
    weights = tmp_path / "assets"
    weights.mkdir()
    (weights / "full.pth").write_bytes(b"fixture-only")
    calls = []

    def predict(audio, rate, **kwargs):
        calls.append(kwargs)
        n = 1 + (audio.x.shape[1] - 1024) // kwargs["hop_length"]
        return Tensor(np.ones((1, n)) * 220), Tensor(np.ones((1, n)) * 0.7)

    backend = types.SimpleNamespace(
        __file__=str(tmp_path / "__init__.py"),
        predict=predict,
        decode=types.SimpleNamespace(weighted_argmax=object()),
    )
    torch = types.SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchcrepe", backend)
    monkeypatch.setattr(
        "natural_features.features.audio.vocals.version", lambda _: "fixture"
    )
    x = np.ones(16000) * 0.2
    a = source_pitch_crepe(
        stimulus(x, offset=3.789),
        source_role="estimated_vocals",
        source_provenance={"separator": "test-reference"},
    )
    np.testing.assert_allclose(
        a.times_s, 3.789 + 0.032 + np.arange(47) * 0.02, atol=1e-12
    )
    np.testing.assert_allclose(a.time_bounds_s[0], [3.789, 3.853], atol=1e-12)
    np.testing.assert_allclose(a.values[:, 2], 0.2)
    assert a.clock == "audio:reference"
    assert a.metadata["voicing_threshold"] is None
    assert a.metadata["source_provenance"] == {"separator": "test-reference"}
    assert calls[0]["pad"] is False
    assert calls[0]["decoder"] is backend.decode.weighted_argmax
    assert len(a.metadata["weight_sha256"]) == 64


def test_separation_requires_local_weights_and_native_rate(tmp_path):
    with pytest.raises(ValueError, match="resample"):
        separate_vocals(stimulus(), model_repo=tmp_path)
    with pytest.raises(ValueError, match="prefetched"):
        separate_vocals(stimulus(rate=44100), model_repo=tmp_path)


def test_nonfinite_audio_rejected(tmp_path):
    for x in [np.full(1600, np.nan), np.full(1600, np.inf)]:
        with pytest.raises(ValueError, match="finite"):
            separate_vocals(stimulus(x, rate=44100), model_repo=tmp_path)
