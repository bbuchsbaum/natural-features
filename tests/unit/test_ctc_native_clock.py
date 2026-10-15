"""Regression checks against the valid-convolution geometry of real checkpoints."""
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech import phonology as ph


@pytest.fixture
def native(monkeypatch):
    runtime = ph.CTCModelRuntime(
        processor=SimpleNamespace(),
        model=SimpleNamespace(config=SimpleNamespace(
            model_type="wav2vec2", conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            conv_stride=[5, 2, 2, 2, 2, 2, 2], add_adapter=False,
        )), torch=None, device="cpu", model_id="test-native", sample_rate_hz=16000,
        local_files_only=True,
    )
    calls = []

    def forward(rt, wav):
        n = len(wav)
        # Independent sequential valid-convolution output-length calculation.
        for kernel, stride in zip(rt.model.config.conv_kernel, rt.model.config.conv_stride):
            n = (n - kernel) // stride + 1
        calls.append(len(wav))
        return np.full((n, 3), 1/3, dtype=np.float32)

    monkeypatch.setattr(ph, "_ctc_forward_chunk", forward)
    return runtime, calls


def extract(native, duration, sr=16000, offset=0, **kwargs):
    runtime, _ = native
    stimulus = AudioStimulus.from_array(
        np.zeros(round(duration * sr), np.float32), sr_hz=sr, start_offset_s=offset,
    )
    return ph.ctc_phone_posteriors(
        stimulus, model="test-native", runtime=runtime, drop_special_tokens=False, **kwargs,
    )


@pytest.mark.parametrize("duration,count", [(30, 1499), (58, 2899), (58.011, 2900)])
def test_real_frontend_grid_has_no_stretch_or_chunk_seam(native, duration, count):
    out = extract(native, duration, offset=2.7)
    np.testing.assert_allclose(out.times_s, 2.7 + .0125 + np.arange(count) * .02, atol=1e-12)
    np.testing.assert_allclose(out.time_bounds_s[:, 0], 2.7 + np.arange(count) * .02, atol=1e-12)
    np.testing.assert_allclose(np.diff(out.time_bounds_s, axis=1), .025, atol=1e-12)
    assert out.timebase.hop_s == .02
    assert out.metadata["regular_native_grid"]
    assert out.metadata["input_context"] == "entire_inference_chunk"
    assert out.time_bounds_s[-1, 1] <= 2.7 + duration


def test_awkward_partial_chunks_use_one_global_sample_grid(native):
    out = extract(native, 3.173, chunk_window_s=1.003, chunk_overlap_s=.137)
    np.testing.assert_allclose(out.times_s, .0125 + np.arange(158) * .02, atol=1e-12)
    chunks = out.metadata["chunks"]
    assert len(chunks) > 2
    assert any(c["sample_start"] != c["planned_sample_start"] for c in chunks)
    ownership = np.zeros(len(out.times_s), dtype=int)
    for c in chunks:
        assert 0 <= c["planned_sample_start"] - c["sample_start"] < 320
        for lo, hi in c["kept_row_ranges"]:
            ownership[lo:hi] += 1
            assert np.all(out.time_bounds_s[lo:hi, 0] >= c["input_start_s"] - 1e-12)
            assert np.all(out.time_bounds_s[lo:hi, 1] <= c["input_end_s"] + 1e-12)
    np.testing.assert_array_equal(ownership, 1)


def test_no_overlap_preserves_real_gaps_instead_of_stretching(native):
    out = extract(native, 3, chunk_window_s=1, chunk_overlap_s=0)
    assert not out.metadata["regular_native_grid"]
    assert out.timebase.kind == "irregular"
    assert out.timebase.hop_s is None
    np.testing.assert_allclose(out.times_s[[0, 48, 49, 97, 98, -1]],
                               [.0125, .9725, 1.0125, 1.9725, 2.0125, 2.9725])


def test_resampling_and_offset_do_not_change_native_geometry(native):
    out = extract(native, 1, sr=44100, offset=-.7)
    np.testing.assert_allclose(out.times_s, -.7 + .0125 + np.arange(49) * .02, atol=1e-12)
    assert out.metadata["model_samples"] == 16000
    assert out.metadata["source_samples"] == 44100
    assert out.metadata["source_sample_rate_hz"] == 44100


@pytest.mark.parametrize("setting", ["unknown", "adapter", "count"])
def test_unknown_frontends_and_output_count_disagreement_fail(native, monkeypatch, setting):
    rt, _ = native
    if setting == "unknown":
        rt.model.config.model_type = "unqualified"
    elif setting == "adapter":
        rt.model.config.add_adapter = True
    else:
        monkeypatch.setattr(ph, "_ctc_forward_chunk", lambda *args: np.ones((50, 3)))
    with pytest.raises((ValueError, RuntimeError), match="frontend|adapter"):
        extract(native, 1)


def test_explicit_stride_is_an_assertion_and_short_input_is_not_padded(native):
    with pytest.raises(ValueError, match="native CTC hop"):
        extract(native, 1, stride_s=.01)
    with pytest.raises(RuntimeError, match="receptive field"):
        extract(native, .01)
    assert len(native[1]) == 0


def test_articulatory_mapping_preserves_support_and_context(native):
    out = extract(native, 1)
    mapped = ph.articulatory_from_posteriors(out)
    np.testing.assert_array_equal(mapped.time_bounds_s, out.time_bounds_s)
    assert mapped.temporal_context == out.temporal_context
    assert mapped.metadata["source_metadata"]["chunks"] == out.metadata["chunks"]


@pytest.mark.parametrize("delimiter", ["|", "<wb>"])
def test_word_boundaries_are_not_blank_phone_or_acoustic_silence(delimiter):
    labels = ["<pad>", delimiter, "a", "sil"]
    processor = SimpleNamespace(tokenizer=SimpleNamespace(
        word_delimiter_token=delimiter, convert_ids_to_tokens=lambda i: labels[i],
    ))
    raw = np.array([[.1, .7, .15, .05]], np.float32)
    probs, names = ph._decode_ctc_labels(processor, raw, drop_special_tokens=False)
    assert names == ["<pad>", "word_boundary", "a", "silence"]
    np.testing.assert_allclose(probs, raw)
    from natural_features.core.feature_types import FeatureSeries
    post = FeatureSeries(values=probs, times_s=np.array([.02]),
                         coords={"feature": names}, metadata={"extractor_id": "test", "params_hash": "test"})
    art = ph.articulatory_from_posteriors(post, include_uncertainty=False)
    silence = list(art.coords["feature"]).index("silence")
    assert art.values[0, silence] == pytest.approx(.05)
    filtered, filtered_names = ph._decode_ctc_labels(processor, raw, drop_special_tokens=True)
    assert filtered_names == ["word_boundary", "a", "silence"]
    np.testing.assert_allclose(filtered, raw[:, 1:] / .9)


def test_bare_word_separator_cannot_become_articulatory_silence():
    assert ph._features_for_label("|") == set()
    assert ph._features_for_label("word_boundary") == set()
    assert ph._features_for_label("sil") == {"silence"}
