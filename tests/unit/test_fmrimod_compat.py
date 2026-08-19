from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.feature_types import EventSeries
from natural_features.fmri.compat import (
    event_series_to_fmrimod_event_variable,
    has_fmrimod,
    map_hrf_name,
    render_events_with_fmrimod,
    to_sampling_frame,
)
from natural_features.fmri.hrf import hrf_kernel

pytestmark = [pytest.mark.external]


def _events() -> EventSeries:
    return EventSeries(
        onset_s=np.array([0.5, 1.5, 2.5]),
        offset_s=np.array([0.7, 1.9, 2.8]),
        confidence=np.array([0.7, 0.9, 0.4]),
        metadata={"extractor_id": "ev", "params_hash": "pp"},
    )


@pytest.mark.skipif(not has_fmrimod(), reason="fmrimod not available")
def test_sampling_frame_adapter_and_render() -> None:
    sf = to_sampling_frame(tr_s=1.0, n_scans=5)
    grid = sf.grid()
    assert grid.shape == (5,)
    out = render_events_with_fmrimod(_events(), tr_s=1.0, n_scans=5, hrf="glover")
    assert out.values.shape == (5, 1)


@pytest.mark.skipif(not has_fmrimod(), reason="fmrimod not available")
def test_event_variable_conversion_and_hrf_mapping() -> None:
    ev = event_series_to_fmrimod_event_variable(_events(), value_mode="duration")
    assert ev.name == "event"
    assert map_hrf_name("glover") == "spmg1"
    k = hrf_kernel(1.0, kind="glover", backend="fmrimod")
    assert k.ndim == 1


@pytest.mark.skipif(not has_fmrimod(), reason="fmrimod not available")
def test_hrf_regressor_spmg1_lags_energy() -> None:
    from natural_features.core.stimulus import AudioStimulus
    from natural_features.core.timebase import ClockMap, TemporalContext
    from natural_features.core.feature_bundle import temporal_object_in_clock
    from natural_features.fmri.compat import hrf_regressor
    from natural_features.workflows.extract_features import extract_features

    sr_hz = 8000
    duration_s = 16.0
    t = np.arange(int(duration_s * sr_hz), dtype=np.float32) / sr_hz
    envelope = np.full(t.shape, 0.03, dtype=np.float32)
    envelope[(t >= 4.0) & (t < 8.0)] = 0.4
    samples = envelope * np.sin(2.0 * np.pi * 220.0 * t)
    audio = AudioStimulus.from_array(samples, sr_hz=sr_hz)
    rms = extract_features(audio, features=["audio.rms"]).features["audio.rms"]

    stim_onset_s = 0.67
    tr_s = 2.0
    n_trs = int(np.ceil((stim_onset_s + duration_s) / tr_s))
    rms_scan = temporal_object_in_clock(
        rms,
        "scan:run-01",
        context=TemporalContext(
            (ClockMap("stimulus", "scan:run-01", offset_s=stim_onset_s),)
        ),
    )
    bold = hrf_regressor(
        rms_scan,
        tr_s=tr_s,
        n_scans=n_trs,
        hrf="spmg1",
        start_time=0.0,
    )
    assert bold.values.shape == (n_trs, 1)
    assert np.all(np.isfinite(bold.values))
    np.testing.assert_allclose(bold.times_s, np.arange(n_trs) * tr_s)
    # Loud RMS starts at scan 4.67 s; SPMG1 peaks several seconds later.
    energy = bold.values[:, 0]
    peak_s = float(bold.times_s[int(np.argmax(energy))])
    assert peak_s > 4.67
    assert float(np.max(energy)) > float(energy[0])
