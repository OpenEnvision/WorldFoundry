"""Generate a self-contained HTML run browser from a run index payload.

The browser is a single-file, dependency-free page that renders a filterable
table of runs with a detail panel showing the selected run's JSON.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.utils import write_text


RUN_BROWSER_SCHEMA_VERSION = "worldfoundry-run-browser"


def _json_for_script(payload: Mapping[str, Any]) -> str:
    """Serialize *payload* for embedding in an HTML ``<script>`` tag, escaping ``<``, ``>``, and ``&``."""
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _metric_columns(index: Mapping[str, Any]) -> list[str]:
    """Extract up to 8 metric IDs from a run index to use as table columns."""
    values = index.get("metric_ids") or []
    if not isinstance(values, list):
        return []
    return [str(item) for item in values[:8]]


def build_run_browser_html(index: Mapping[str, Any]) -> str:
    """Build a dependency-free HTML run browser from a run index payload."""

    payload = dict(index)
    payload["browser_schema_version"] = RUN_BROWSER_SCHEMA_VERSION
    metrics = _metric_columns(payload)
    metric_headers = "\n".join(f"<th>{html.escape(metric)}</th>" for metric in metrics)
    metric_list_json = _json_for_script({"metrics": metrics})
    index_json = _json_for_script(payload)
    title = "WorldFoundry Runs"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5d6978;
      --line: #d8dee8;
      --accent: #0f766e;
      --bad: #b42318;
      --warn: #9a6700;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111418;
        --panel: #171c22;
        --text: #edf1f5;
        --muted: #aab4c0;
        --line: #2e3742;
        --accent: #2dd4bf;
        --bad: #ff8a80;
        --warn: #f5c451;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0; font-size: 22px; font-weight: 650; letter-spacing: 0; }}
    .summary {{ color: var(--muted); white-space: nowrap; }}
    main {{ padding: 18px 28px 28px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 360px) repeat(3, minmax(140px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    input, select {{
      width: 100%;
      min-height: 34px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--panel);
      color: var(--muted);
      font-weight: 600;
    }}
    tr:hover td {{ background: color-mix(in srgb, var(--accent) 8%, transparent); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .status-succeeded {{ color: var(--accent); font-weight: 600; }}
    .status-failed, .status-invalid {{ color: var(--bad); font-weight: 600; }}
    .status-completed_with_failures {{ color: var(--warn); font-weight: 600; }}
    .detail {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(260px, 380px) 1fr;
      gap: 14px;
    }}
    .panel {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
    }}
    .panel h2 {{ margin: 0 0 10px; font-size: 14px; }}
    pre {{
      margin: 0;
      overflow: auto;
      max-height: 460px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    @media (max-width: 820px) {{
      header {{ display: block; padding: 18px; }}
      main {{ padding: 14px 18px 22px; }}
      .toolbar {{ grid-template-columns: 1fr; }}
      .detail {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>WorldFoundry Runs</h1>
      <div class="summary" id="summary"></div>
    </div>
    <div class="summary" id="issues"></div>
  </header>
  <main>
    <div class="toolbar">
      <input id="query" placeholder="Filter model, benchmark, run id, status">
      <select id="benchmark"><option value="">All benchmarks</option></select>
      <select id="model"><option value="">All models</option></select>
      <select id="status"><option value="">All statuses</option></select>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Status</th>
            <th>Model</th>
            <th>Benchmark</th>
            <th>Dataset</th>
            <th>Backend</th>
            <th>Samples</th>
            {metric_headers}
            <th>Artifacts</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <section class="detail">
      <div class="panel">
        <h2>Selected Run</h2>
        <div id="selected">Select a row.</div>
      </div>
      <div class="panel">
        <h2>JSON</h2>
        <pre id="json"></pre>
      </div>
    </section>
  </main>
  <script id="worldfoundry-index" type="application/json">{index_json}</script>
  <script id="worldfoundry-metrics" type="application/json">{metric_list_json}</script>
  <script>
    const index = JSON.parse(document.getElementById('worldfoundry-index').textContent);
    const metrics = JSON.parse(document.getElementById('worldfoundry-metrics').textContent).metrics;
    const rows = Array.isArray(index.rows) ? index.rows : [];
    const query = document.getElementById('query');
    const benchmark = document.getElementById('benchmark');
    const model = document.getElementById('model');
    const status = document.getElementById('status');
    const tbody = document.getElementById('rows');
    const selected = document.getElementById('selected');
    const json = document.getElementById('json');

    function text(value) {{ return value === null || value === undefined ? '' : String(value); }}
    function formatValue(value) {{
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
      return text(value);
    }}
    function optionList(select, values) {{
      [...new Set(values.filter(Boolean).map(text))].sort().forEach((value) => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }});
    }}
    function link(path, label) {{
      if (!path) return '';
      return `<a href="${{path}}" target="_blank" rel="noreferrer">${{label}}</a>`;
    }}
    function rowMatches(row) {{
      const q = query.value.trim().toLowerCase();
      const haystack = [
        row.run_id, row.label, row.model_id, row.model_name,
        row.benchmark, row.dataset_id, row.status, row.backend
      ].map(text).join(' ').toLowerCase();
      return (!q || haystack.includes(q))
        && (!benchmark.value || text(row.benchmark) === benchmark.value)
        && (!model.value || text(row.model_id || row.model_name) === model.value)
        && (!status.value || text(row.status) === status.value);
    }}
    function renderRows() {{
      const filtered = rows.filter(rowMatches);
      document.getElementById('summary').textContent =
        `${{filtered.length}} shown / ${{rows.length}} indexed`;
      tbody.innerHTML = '';
      filtered.forEach((row, visibleIndex) => {{
        const tr = document.createElement('tr');
        tr.tabIndex = 0;
        tr.dataset.index = String(rows.indexOf(row));
        const artifacts = row.artifacts || {{}};
        const cells = [
          `<td>${{text(row.run_id || row.label)}}</td>`,
          `<td class="status-${{text(row.status)}}">${{text(row.status)}}</td>`,
          `<td>${{text(row.model_id || row.model_name)}}</td>`,
          `<td>${{text(row.benchmark)}}</td>`,
          `<td>${{text(row.dataset_id)}}</td>`,
          `<td class="${{text(row.backend) === 'mock' ? 'status-failed' : ''}}">${{text(row.backend)}}</td>`,
          `<td>${{formatValue(row.sample_count)}}</td>`
        ];
        metrics.forEach((metric) => {{
          const metricValue = row.metrics && Object.prototype.hasOwnProperty.call(row.metrics, metric)
            ? row.metrics[metric] : '';
          cells.push(`<td>${{formatValue(metricValue)}}</td>`);
        }});
        cells.push(`<td>${{link(row.source_path, 'summary')}} ${{link(artifacts.scorecard, 'scorecard')}} ${{link(artifacts.report, 'report')}}</td>`);
        tr.innerHTML = cells.join('');
        tr.addEventListener('click', () => selectRow(row));
        tr.addEventListener('keydown', (event) => {{
          if (event.key === 'Enter' || event.key === ' ') selectRow(row);
        }});
        tbody.appendChild(tr);
        if (visibleIndex === 0 && !json.textContent) selectRow(row);
      }});
    }}
    function selectRow(row) {{
      const artifacts = row.artifacts || {{}};
      selected.innerHTML = [
        `<div><b>${{text(row.run_id || row.label)}}</b></div>`,
        `<div>${{text(row.model_id || row.model_name)}} / ${{text(row.benchmark)}}</div>`,
        `<div>Status: ${{text(row.status)}}; backend: ${{text(row.backend) || 'n/a'}}; samples: ${{formatValue(row.sample_count)}}</div>`,
        `<div>${{link(row.source_path, 'summary')}} ${{link(artifacts.scorecard, 'scorecard')}} ${{link(artifacts.report, 'report')}}</div>`
      ].join('');
      json.textContent = JSON.stringify(row, null, 2);
    }}
    optionList(benchmark, rows.map((row) => row.benchmark));
    optionList(model, rows.map((row) => row.model_id || row.model_name));
    optionList(status, rows.map((row) => row.status));
    document.getElementById('issues').textContent = Array.isArray(index.issues) && index.issues.length
      ? `${{index.issues.length}} issue(s)` : '';
    [query, benchmark, model, status].forEach((control) => control.addEventListener('input', renderRows));
    renderRows();
  </script>
</body>
</html>
"""


def write_run_browser(index: Mapping[str, Any], output_html: str | Path) -> Path:
    """Build the run browser HTML and write it to *output_html*.

    Returns:
        The resolved path of the written file.
    """
    return write_text(output_html, build_run_browser_html(index))


__all__ = [
    "RUN_BROWSER_SCHEMA_VERSION",
    "build_run_browser_html",
    "write_run_browser",
]
