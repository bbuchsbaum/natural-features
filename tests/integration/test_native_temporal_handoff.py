from __future__ import annotations

import inspect

import numpy as np

from natural_features import (
    ClockMap,
    EventSeries,
    FeatureBundle,
    FeatureSeries,
    SupportSpec,
    TemporalContext,
    TimebaseSpec,
)


def _metadata(name: str) -> dict[str, str]:
    return {"extractor_id": name, "params_hash": "acceptance"}


def _acceptance_bundle() -> FeatureBundle:
    context = TemporalContext(
        (ClockMap("stimulus", "scan:run-01", offset_s=-23.0),)
    )
    fast_times = 23.0 + np.arange(0.0, 200.0, 0.1)
    slow_times = 23.0 + np.arange(0.0, 200.0, 0.5)
    fast = FeatureSeries(
        values=np.arange(len(fast_times), dtype=np.float32)[:, None],
        times_s=fast_times,
        metadata=_metadata("acceptance.fast"),
        timebase=TimebaseSpec(
            kind="audio_hop",
            reference="stimulus",
            hop_s=0.1,
            support=SupportSpec(kind="window", width_s=0.1),
        ),
    )
    slow = FeatureSeries(
        values=np.arange(len(slow_times), dtype=np.float32)[:, None],
        times_s=slow_times,
        metadata=_metadata("acceptance.slow"),
        timebase=TimebaseSpec(
            kind="windows",
            reference="stimulus",
            stride_s=0.5,
            support=SupportSpec(kind="window", width_s=0.5),
        ),
    )
    events = EventSeries(
        onset_s=np.array([30.0, 47.2, 220.4]),
        offset_s=np.array([30.4, 48.1, 221.2]),
        label=np.array(["one", "two", "three"], dtype=object),
        metadata=_metadata("acceptance.events"),
        timebase=TimebaseSpec(kind="events", reference="stimulus"),
    )
    return FeatureBundle(
        {"fast": fast, "slow": slow, "events": events},
        temporal_context=context,
        metadata={"scan_duration_s": 200.0},
    )


def test_native_grid_to_downstream_payload_acceptance_contract() -> None:
    bundle = _acceptance_bundle()
    fast = bundle.features["fast"]
    slow = bundle.features["slow"]

    fast_scan = bundle.in_clock("fast", "scan:run-01")
    slow_scan = bundle.in_clock("slow", "scan:run-01")
    events_payload = bundle.temporal_payload(
        "events",
        target_clock="scan:run-01",
    ).to_dict()

    assert len(fast_scan.times_s) == 2000
    assert len(slow_scan.times_s) == 400
    assert fast_scan.values is fast.values
    assert slow_scan.values is slow.values
    np.testing.assert_allclose(fast_scan.times_s[:3], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(slow_scan.times_s[:3], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(events_payload["onset_s"], [7.0, 24.2, 197.4])
    assert events_payload["clock"] == "scan:run-01"
    assert events_payload["timebase"]["support"]["kind"] == "interval"
    assert events_payload["metadata"]["scan_duration_s"] == 200.0

    # TR is introduced only by the downstream consumer. The native handoff API
    # has no TR or HRF argument and leaves all feature grids unchanged.
    assert "tr_s" not in inspect.signature(bundle.temporal_payload).parameters
    assert "hrf" not in inspect.signature(bundle.temporal_payload).parameters
    downstream_tr_s = 1.77
    downstream_grid = np.arange(
        0.0,
        bundle.metadata["scan_duration_s"],
        downstream_tr_s,
    )
    assert downstream_grid[0] == 0.0
    assert downstream_grid[-1] < 200.0
