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
