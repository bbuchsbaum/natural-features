from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.backend_errors import BackendInferenceError, BackendLoadError
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio import source_separation as ss
from natural_features.features.audio import stem_levels as sl


@pytest.fixture
def runtime(monkeypatch):
    def separate_stereo(x):
        # Canonical sources in deliberately non-canonical backend order.
        values = {"vocals": x, "drums": x * .5, "bass": x * 0, "other": x * .25}
        return np.stack([values[n] for n in ("other", "vocals", "bass", "drums")], axis=1)
    rt = SimpleNamespace(separate_stereo=separate_stereo,
                         sources=["other", "vocals", "bass", "drums"], versions={"test": "fixed"})
    monkeypatch.setattr(sl, "_load_runtime", lambda *a, **k: rt)
    return rt


def _audio(x, *, sr=1000, offset=7.25):
    return AudioStimulus.from_array(np.asarray(x, dtype=np.float32), sr, start_offset_s=offset, clock="song")


def test_known_stem_aggregation_clock_and_full_window_support(runtime):
    x = np.ones((510, 2), dtype=np.float32) * .1
    out = sl.audio_stem_levels(_audio(x), hop_s=.1, window_s=.25)
    value, diagnostic = out["default"], out["diagnostics"]
    assert value.coords["feature"] == list(sl.STEMS)
    # Independent RMS oracle for gains 1, .5, 0, .25.
    expected = np.array([20 * np.log10(.1), 20 * np.log10(.05), -120, 20 * np.log10(.025)])
    # Polyphase resampling has a documented zero-boundary transient; its
    # levels remain within the analytic whole-frame RMS oracle tolerance.
    np.testing.assert_allclose(value.values[0], expected, atol=.01)
    assert len(value.times_s) == 3
    np.testing.assert_allclose(value.times_s, [7.375, 7.475, 7.575])
    np.testing.assert_allclose(value.time_bounds_s, [[7.25, 7.5], [7.35, 7.6], [7.45, 7.7]])
    assert value.clock == "song"
    assert diagnostic.values[0, 7] == 1  # bass_silent
    assert diagnostic.values[0, 8] == 1  # bass_below_floor, exact zero is below floor
    assert value.metadata["computation_context"]["policy"] == "bidirectional"
    assert value.metadata["extractor_name"] == "audio.stem_levels"
    assert diagnostic.metadata["extractor_name"] == "audio.stem_levels.diagnostics"
    assert value.metadata["output_pair_id"] == diagnostic.metadata["output_pair_id"]


def test_gain_stereo_energy_floor_and_ratio_validity(runtime):
    n = 250
    # Antiphase stereo has zero mono mean but nonzero stereo energy.
    x = np.column_stack([np.full(n, .01), np.full(n, -.01)]).astype(np.float32)
    base = sl.audio_stem_levels(_audio(x), hop_s=.25, window_s=.25, units="rms")["default"]
    base_db = sl.audio_stem_levels(_audio(x), hop_s=.25, window_s=.25)["default"]
    doubled = sl.audio_stem_levels(_audio(2 * x), hop_s=.25, window_s=.25)["default"]
    assert base.values[0, 0] == pytest.approx(.01, abs=2e-5)
    assert doubled.values[0, 0] - base_db.values[0, 0] == pytest.approx(6.0206, abs=1e-4)

    runtime.sources = ["vocals", "drums", "bass", "other"]

    def cancellation(stereo):
        z = np.zeros_like(stereo)
        return np.stack([stereo, z, stereo, -stereo], axis=1)  # vocals, drums, bass, other
    runtime.separate_stereo = cancellation
    ratio = sl.audio_stem_levels(_audio(x), hop_s=.25, window_s=.25, include_ratio=True)
    assert ratio["default"].values[0, 0] > -100
    assert np.isnan(ratio["default"].values[0, -1])
    assert ratio["diagnostics"].values[0, -1] == 0
    assert ratio["default"].metadata["column_units"] == {
        "vocals": "dbfs", "drums": "dbfs", "bass": "dbfs", "other": "dbfs",
        "vocals_to_accompaniment_db": "db",
    }

    runtime.separate_stereo = lambda stereo: np.stack([stereo, stereo, stereo, stereo], axis=1)
    summed = sl.audio_stem_levels(_audio(x), hop_s=.25, window_s=.25, include_ratio=True)["default"]
    # Accompaniment is its waveform sum (3x), rather than a sum of powers.
    assert summed.values[0, -1] == pytest.approx(-20 * np.log10(3), abs=1e-5)


def test_quiet_floor_and_backend_failures(runtime):
    quiet = _audio(np.full((250, 2), 1e-8, dtype=np.float32))
    out = sl.audio_stem_levels(quiet, hop_s=.25, window_s=.25, db_floor=-80)
    assert out["default"].values[0, 0] == -80
    assert out["diagnostics"].values[0, 2] == 1
    runtime.separate_stereo = lambda x: np.zeros((len(x), 3, 2))
    with pytest.raises(BackendInferenceError):
        sl.audio_stem_levels(quiet, hop_s=.25, window_s=.25)
    sl._load_runtime = lambda *a, **k: (_ for _ in ()).throw(BackendLoadError("x", "asset"))
    with pytest.raises(BackendLoadError):
        sl.audio_stem_levels(quiet, hop_s=.25, window_s=.25)


def test_batched_window_rms_matches_naive_multichannel_and_empty_frames():
    rng = np.random.default_rng(42)
    stems = rng.normal(size=(43, 5, 3)).astype(np.float32)
    starts, window = np.array([0, 2, 11, 23, 31]), 12
    actual = sl._batched_window_rms(stems, starts, window, batch_size=2)
    expected = np.array([
        np.sqrt(np.mean(stems[start:start + window].astype(np.float64) ** 2, axis=(0, 2)))
        for start in starts
    ])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert sl._batched_window_rms(stems, np.array([], dtype=np.int64), window).shape == (0, 5)


@pytest.mark.parametrize("kwargs", [{"hop_s": True}, {"window_s": np.bool_(True)},
                                     {"db_floor": False}, {"db_floor": 1e100}])
def test_boolean_and_float32_overflow_parameters_rejected_before_backend_loading(monkeypatch, kwargs):
    monkeypatch.setattr(sl, "_load_runtime", lambda *a, **k: pytest.fail("must not load backend"))
    options = {"hop_s": .25, "window_s": .25, **kwargs}
    with pytest.raises(ValueError):
        sl.audio_stem_levels(_audio(np.ones((250, 2))), **options)


def test_short_noninteger_grid_and_malformed_channels(runtime):
    short = sl.audio_stem_levels(_audio(np.ones((7, 2))), hop_s=.0073, window_s=.0251)
    assert short["default"].values.shape == (0, 4)
    regular = sl.audio_stem_levels(_audio(np.ones((100, 2))), hop_s=.0073, window_s=.0251)["default"]
    assert regular.timebase.hop_s == .007
    np.testing.assert_allclose(np.diff(regular.times_s), .007)
    silence = sl.audio_stem_levels(_audio(np.zeros((250, 2))), hop_s=.25, window_s=.25)
    np.testing.assert_array_equal(silence["default"].values, np.full((1, 4), -120, dtype=np.float32))
    np.testing.assert_array_equal(silence["diagnostics"].values[:, [1, 4, 7, 10]], 1)
    with pytest.raises(ValueError, match="mono or stereo"):
        sl.audio_stem_levels(_audio(np.ones((250, 3))), hop_s=.25, window_s=.25)


def test_loader_rejects_wrong_hash_without_crepe_or_model_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / ss.SEPARATOR_FILE
    checkpoint.write_bytes(b"wrong")
    monkeypatch.delitem(__import__("sys").modules, "torchcrepe", raising=False)
    monkeypatch.setattr(ss, "_sha256", lambda path: "not-the-pinned-hash")
    monkeypatch.setattr(ss, "_cached_separator_runtime", lambda *a: pytest.fail("must not load model"))
    with pytest.raises(BackendLoadError, match="SHA256 mismatch"):
        ss.load_separator_runtime(str(checkpoint), local_files_only=True)
    assert "torchcrepe" not in __import__("sys").modules


def test_separator_runtime_stereo_antiphase_energy_differs_from_legacy_mono(monkeypatch):
    torch = pytest.importorskip("torch")
    apply = pytest.importorskip("demucs.apply")

    def fake_apply(model, mix, **kwargs):
        return mix[:, None]  # batch, source, channel, sample

    monkeypatch.setattr(apply, "apply_model", fake_apply)
    runtime = ss.SeparatorRuntime(torch, SimpleNamespace(sources=["vocals"]), "cpu", {})
    phase = .1 * np.sin(2 * np.pi * np.arange(4410) / 31)
    stereo = np.column_stack([phase, -phase]).astype(np.float32)
    separated = runtime.separate_stereo(stereo)
    assert separated.shape == (len(stereo), 1, 2)
    assert np.max(np.abs(separated)) > 0
    assert not np.any(runtime.separate(stereo))
    # The level path must not turn quiet nonzero input into exact silence.
    tiny = runtime.separate_stereo(stereo * 1e-10)
    assert np.max(np.abs(tiny)) > 0
    np.testing.assert_allclose(tiny / 1e-10, separated, atol=2e-8)


def test_nonfinite_resampling_is_an_inference_error(runtime, monkeypatch):
    real_resample = sl._resample

    def invalid_resample(x, source_sr, target_sr):
        result = real_resample(x, source_sr, target_sr)
        if x.ndim == 3:
            result[0, 0, 0] = np.nan
        return result

    monkeypatch.setattr(sl, "_resample", invalid_resample)
    with pytest.raises(BackendInferenceError):
        sl.audio_stem_levels(_audio(np.ones((300, 2), np.float32) * .1))
