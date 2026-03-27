"""Flow and stage specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


StageCallable = Callable[..., Any]


@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 0
    retry_backoff_s: float = 0.0
    retry_on_exceptions: tuple[type[Exception], ...] = (Exception,)


@dataclass
class StageSpec:
    stage_id: str
    fn: StageCallable
    params: dict[str, Any] = field(default_factory=dict)
    deps: list[str] = field(default_factory=list)
    priority: int = 0
    stop_on_upstream_fail: bool = True
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass
class FlowSpec:
    flow_id: str
    stages: list[StageSpec]

    def by_id(self) -> dict[str, StageSpec]:
        return {s.stage_id: s for s in self.stages}

    def topological_order(self) -> list[StageSpec]:
        by = self.by_id()
        indegree: dict[str, int] = {k: 0 for k in by}
        children: dict[str, list[str]] = {k: [] for k in by}
        for stage in self.stages:
            for d in stage.deps:
                if d not in by:
                    raise ValueError(f"Unknown dependency '{d}' for stage '{stage.stage_id}'")
                indegree[stage.stage_id] += 1
                children[d].append(stage.stage_id)
        ready = sorted([k for k, v in indegree.items() if v == 0], key=lambda k: by[k].priority)
        order: list[str] = []
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            for child in children[cur]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort(key=lambda k: by[k].priority)
        if len(order) != len(self.stages):
            raise ValueError("Cycle detected in flow dependencies")
        return [by[k] for k in order]

