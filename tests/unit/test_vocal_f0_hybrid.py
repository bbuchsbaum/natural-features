import importlib
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio import vocal_f0 as vf


def test_internal_gap_support_never_fills_missing_output():
    c = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)
    actual = vf._gap_aware_core(c, np.ones(7, bool), .01, .01, .01)
    np.testing.assert_array_equal(actual, [0, 1, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(vf._gap_aware_core(c, np.ones(7, bool), .01, 0, .01),
                                  [0, 1, 0, 0, 0, 1, 0])
    assert not actual[3]


def test_unavailable_and_endpoint_gaps_are_never_bridged():
    c = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    available = np.ones(9, bool)
    available[4] = False
    np.testing.assert_array_equal(vf._gap_aware_core(c, available, .01, 1., .01),
                                  [0, 0, 1, 0, 0, 0, 1, 0, 0])


def test_exact_gap_limit_and_awkward_hop():
    c = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(vf._gap_aware_core(c, np.ones(8, bool), .01, .02, .01),
                                  [0, 1, 1, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(vf._gap_aware_core(c, np.ones(8, bool), .01, .019, .01),
                                  [0, 1, 0, 0, 0, 0, 1, 0])
    # floor(.02/(117/16000))=2, ceil(.01/(117/16000))=2.
    np.testing.assert_array_equal(vf._gap_aware_core(c, np.ones(8, bool), .01, .02, 117/16000),
                                  [0, 0, 1, 0, 0, 1, 0, 0])


def test_alternating_candidate_is_an_honest_continuity_counterexample():
    c = np.array([1, 0, 1, 0, 1, 0, 1], dtype=bool)
    actual = vf._gap_aware_core(c, np.ones(7, bool), .01, .01, .01)
    np.testing.assert_array_equal(actual, [0, 0, 1, 0, 1, 0, 0])
    # Some isolated candidates survive envelope support: not continuous voicing.
    assert actual.sum() == 2 and not np.any(actual & ~c)


def test_hybrid_matches_independent_explicit_run_oracle():
    rng = np.random.default_rng(21)
    for n in (1, 9, 101):
        for guard, gap in product((1, 2), (2, 4)):
            available = rng.random(n) > .1
            c = (rng.random(n) > .4) & available
            envelope = c.copy()
            for i in range(n):
                if c[i] or not available[i]:
                    continue
                left = i - 1
                while left >= 0 and not c[left]:
                    left -= 1
                right = i + 1
                while right < n and not c[right]:
                    right += 1
                if left >= 0 and right < n and right - left - 1 <= gap and all(available[left:right+1]):
                    envelope[i] = True
            expected = [bool(c[i] and i >= guard and i + guard < n and all(envelope[i-guard:i+guard+1])) for i in range(n)]
            np.testing.assert_array_equal(vf._gap_aware_core(c, available, guard, gap, 1.), expected)


def test_all_frozen_profiles_direct_replay_and_native_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    cal = importlib.import_module('calibrate_vocal_f0')
    def separate(x):
        return np.column_stack([x[:, 0] * .1, x[:, 0] * .9])
    def pitch(x, hop, *args):
        n = 1 + len(x) // hop
        confidence = np.full(n, .9)
        confidence[20:22] = .3
        confidence[50:56] = .3
        return np.full(n, 200.), confidence
    monkeypatch.setattr(vf, '_load_runtime', lambda *a: SimpleNamespace(
        separate=separate, pitch=pitch, sources=['other', 'vocals'], versions={}))
    t = np.arange(16000) / 16000
    x = (.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    audio = AudioStimulus.from_array(x, 16000, start_offset_s=7.25)
    base = vf.audio_vocal_f0(audio)
    for threshold, guard, gap in product((.25, .5, .75), (.01, .02), (.02, .04)):
        params = dict(local_periodicity_threshold=threshold, voicing_guard_s=guard, max_internal_gap_s=gap)
        direct = vf.audio_vocal_f0(audio, **params)
        replayed = cal.replay(base, .65, .03, .01, **params)
        np.testing.assert_array_equal(direct.values, replayed.values)
        np.testing.assert_array_equal(direct.times_s, base.times_s)
        keep = direct.values[:, 4] > 0
        assert not np.any(keep & (base.values[:, 4] == 0))
        assert not keep[20:22].any() and not keep[50:56].any()
        np.testing.assert_array_equal(direct.values[keep, 0], base.values[keep, 0])
        assert np.isnan(direct.values[~keep, 0]).all()


@pytest.mark.parametrize('value', [-.1, np.nan, np.inf])
def test_invalid_gap_rejected(value):
    with pytest.raises(ValueError, match='max_internal_gap_s'):
        vf._gap_aware_core(np.ones(3, bool), np.ones(3, bool), .01, value, .01)
