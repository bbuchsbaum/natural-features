"""Minimal local flow execution engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import time
from typing import Any

from natural_features.flow.cache import cache_fingerprint, invalidation_reasons
from natural_features.flow.spec import FlowSpec, StageSpec


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageRunRecord:
    stage_id: str
    status: str
    fingerprint: str
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    error: str | None = None
    invalidation_reasons: list[str] = field(default_factory=list)


@dataclass
class FlowRunResult:
    flow_id: str
    run_id: str
    mode: str
    started_at: str
    finished_at: str
    stage_records: list[StageRunRecord]
    outputs: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "run_id": self.run_id,
            "mode": self.mode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage_records": [asdict(r) for r in self.stage_records],
        }


def run_flow(
    flow: FlowSpec,
    *,
    run_id: str = "run-local",
    mode: str = "run",
    prior_fingerprints: dict[str, dict[str, Any]] | None = None,
    code_version: str = "dev",
    model_revision: str = "none",
) -> FlowRunResult:
    if mode not in {"run", "plan"}:
        raise ValueError("mode must be 'run' or 'plan'")
    prior_fingerprints = prior_fingerprints or {}
    started = _utc_now()
    outputs: dict[str, Any] = {}
    records: dict[str, StageRunRecord] = {}

    for stage in flow.topological_order():
        payload = {
            "extractor_name": stage.stage_id,
            "params": stage.params,
            "code_version": code_version,
            "model_revision": model_revision,
            "upstream_ids": stage.deps,
        }
        fp = cache_fingerprint(**payload)
        prev = prior_fingerprints.get(stage.stage_id, {})
        reasons = invalidation_reasons(prev, payload)

        dep_failed = False
        for d in stage.deps:
            dep = records.get(d)
            if dep is None or dep.status in {"failed", "cancelled"}:
                dep_failed = True
                break
        if dep_failed and stage.stop_on_upstream_fail:
            records[stage.stage_id] = StageRunRecord(
                stage_id=stage.stage_id,
                status="cancelled",
                fingerprint=fp,
                invalidation_reasons=["upstream-failed"],
            )
            continue

        if reasons == ["cache-valid"]:
            records[stage.stage_id] = StageRunRecord(
                stage_id=stage.stage_id,
                status="cached",
                fingerprint=fp,
                invalidation_reasons=reasons,
            )
            continue

        if mode == "plan":
            records[stage.stage_id] = StageRunRecord(
                stage_id=stage.stage_id,
                status="pending",
                fingerprint=fp,
                invalidation_reasons=reasons,
            )
            continue

        records[stage.stage_id] = _execute_stage(stage, fp, reasons, outputs)

    finished = _utc_now()
    return FlowRunResult(
        flow_id=flow.flow_id,
        run_id=run_id,
        mode=mode,
        started_at=started,
        finished_at=finished,
        stage_records=[records[s.stage_id] for s in flow.topological_order()],
        outputs=outputs,
    )


def _execute_stage(
    stage: StageSpec,
    fingerprint: str,
    invalidation: list[str],
    outputs: dict[str, Any],
) -> StageRunRecord:
    record = StageRunRecord(
        stage_id=stage.stage_id,
        status="running",
        fingerprint=fingerprint,
        invalidation_reasons=invalidation,
        started_at=_utc_now(),
    )
    t0 = time.perf_counter()
    max_attempts = 1 + max(0, stage.retry_policy.retries)
    err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        record.attempts = attempt
        try:
            kwargs = dict(stage.params)
            dep_outputs = {d: outputs.get(d) for d in stage.deps}
            result = stage.fn(dep_outputs=dep_outputs, **kwargs)
            outputs[stage.stage_id] = result
            record.status = "completed"
            record.error = None
            break
        except stage.retry_policy.retry_on_exceptions as exc:
            err = exc
            if attempt >= max_attempts:
                record.status = "failed"
                record.error = repr(exc)
                break
            if stage.retry_policy.retry_backoff_s > 0:
                time.sleep(stage.retry_policy.retry_backoff_s)
    if err is not None and record.status != "completed" and record.error is None:
        record.status = "failed"
        record.error = repr(err)
    record.finished_at = _utc_now()
    record.duration_s = float(time.perf_counter() - t0)
    return record
