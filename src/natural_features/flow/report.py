"""Static HTML reporting."""

from __future__ import annotations

from html import escape
from pathlib import Path

from natural_features.flow.engine import FlowRunResult


def write_html_report(result: FlowRunResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in result.stage_records:
        rows.append(
            "<tr>"
            f"<td>{escape(str(r.stage_id))}</td>"
            f"<td>{escape(str(r.status))}</td>"
            f"<td>{r.attempts}</td>"
            f"<td>{escape(', '.join(r.invalidation_reasons))}</td>"
            f"<td>{r.duration_s if r.duration_s is not None else ''}</td>"
            f"<td>{escape(str(r.error or ''))}</td>"
            "</tr>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>natural_features run report</title>"
        "<style>body{font-family:ui-sans-serif,system-ui;padding:20px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left}th{background:#f5f5f5}</style></head><body>"
        f"<h1>Run Report: {escape(str(result.run_id))}</h1>"
        f"<p>Flow: {escape(str(result.flow_id))}</p>"
        f"<p>Mode: {escape(str(result.mode))}</p>"
        "<table><thead><tr><th>Stage</th><th>Status</th><th>Attempts</th><th>Invalidation</th><th>Duration (s)</th><th>Error</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )
    p.write_text(html, encoding="utf-8")
    return p
