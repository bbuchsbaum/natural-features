from __future__ import annotations

import numpy as np
import pytest

from natural_features import (
    ClockMap,
    EventSeries,
    FeatureBundle,
    TemporalContext,
    TimebaseSpec,
)

pytestmark = pytest.mark.external


def test_fmrimod_consumes_protocol_payload_without_natfeatures_types() -> None:
    pytest.importorskip("fmrimod")
    from fmrimod.events import EventVariable
    from fmrimod.sampling import SamplingFrame

    events = EventSeries(
        onset_s=np.array([30.0, 47.2, 220.4]),
        offset_s=np.array([30.4, 48.1, 221.2]),
        confidence=np.array([0.8, 0.9, 0.7]),
        metadata={"extractor_id": "external.events", "params_hash": "acceptance"},
        timebase=TimebaseSpec(kind="events", reference="stimulus"),
    )
    bundle = FeatureBundle(
        {"events": events},
        temporal_context=TemporalContext(
            (ClockMap("stimulus", "scan:run-01", offset_s=-23.0),)
        ),
    )
    payload = bundle.temporal_payload(
        "events",
        target_clock="scan:run-01",
    ).to_dict()

    # From this point onward only the protocol mapping is used. fmrimod owns
    # event construction and scan sampling.
    model_event = EventVariable(
        name="natural_feature",
        onsets=payload["onset_s"],
        durations=payload["offset_s"] - payload["onset_s"],
        values=payload["confidence"],
        center=False,
    )
    frame = SamplingFrame(
        blocklens=113,
        tr=1.77,
        start_time=0.0,
        precision=0.1,
    )

    np.testing.assert_allclose(model_event.onsets, [7.0, 24.2, 197.4])
    np.testing.assert_allclose(model_event.durations, [0.4, 0.9, 0.8])
    assert frame.n_scans == 113
    assert frame.TR == 1.77
