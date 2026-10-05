#!/usr/bin/env python3
# eval/human_eval_server.py
"""
Web UI for human evaluation of world-model benchmark runs.

Supports multiple independent annotators, each with a unique URL:
    http://<host>:<port>/?annotator=<name>   — annotator mode (blind: no LLM answers shown)
    http://<host>:<port>/                    — master view (read-only: LLM + all annotators)

Dependencies:
    pip install flask

Usage:
    python -m eval.human_eval_server --host 0.0.0.0 --port 7860

Each task page shows:
  - Initial frame and output video
  - Video WM prompt
  - Verifier rows (occlusion / trigger / state evolution / physical inaccuracy)
  - Annotator mode: T/F buttons; answers auto-saved to human_<name>_report.json + summary.json
  - Master mode: read-only badges for LLM and every annotator side by side
"""

import argparse
import base64
import json
import mimetypes
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, abort, jsonify, request

try:
    from benchmark.runners.utils import load_yaml
except ImportError:
    import yaml  # type: ignore

    def load_yaml(path):  # type: ignore
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

try:
    from eval.control_judge import (
        _load_task_fields as _ctrl_load_task_fields,
        _compute_requested_fields,
    )
    _HAS_CONTROL_JUDGE = True
except ImportError:
    _HAS_CONTROL_JUDGE = False

try:
    from eval.utils import _path_from_rel, occlusion_base
except ImportError:
    def occlusion_base(task_id):  # type: ignore
        for s in ("_cardboard", "_curtain", "_lightoff"):
            if task_id.endswith(s):
                return task_id[: -len(s)]
        return task_id
    def _path_from_rel(s, base=None):  # type: ignore
        from pathlib import Path as _P
        p = _P(s)
        return p if p.is_absolute() else ((_P.cwd() if base is None else base) / p).resolve()

# Run-name substrings that identify camera-controlled models (same logic as eval_cli.py)
_CAMERA_CONTROLLED_NAMES = {"hy", "genie", "lingbot"}

# Matches per-task annotator report filenames: human_<annotator>_report.json
_ANNOTATOR_RE = re.compile(r"^human_(.+)_report\.json$")


app = Flask(__name__)

# Set at startup via main()
RUNS_DIR: Path = Path("runs")
TASKS_ROOT: Path = Path("benchmark/tasks")
EXCLUDE_BASELINE: bool = False  # when True, tasks ending in _00 are hidden

# Built once at startup: task_id -> yaml path
_YAML_MAP: Dict[str, Path] = {}


def _build_yaml_map() -> None:
    """Scan TASKS_ROOT once and index every <task_id>.yaml file.

    Also adds base-name aliases (e.g. "ice_on_burner" -> ice_on_burner_cardboard.yaml)
    so that camera-controlled runs whose task IDs use the base name can still
    resolve the correct yaml and init frame.  sorted() ensures _cardboard wins.
    """
    global _YAML_MAP
    print(f"Scanning tasks directory: {TASKS_ROOT} …", flush=True)
    _YAML_MAP = {}
    for p in sorted(TASKS_ROOT.rglob("*.yaml")):
        _YAML_MAP[p.stem] = p
        base = occlusion_base(p.stem)
        if base != p.stem and base not in _YAML_MAP:
            _YAML_MAP[base] = p
    print(f"  Found {len(_YAML_MAP)} task YAML files.", flush=True)


# ---------------------------------------------------------------------------
# Path encoding helpers
# ---------------------------------------------------------------------------

def _enc(p: str) -> str:
    return base64.urlsafe_b64encode(p.encode("utf-8")).decode("ascii")


def _dec(s: str) -> Path:
    return Path(base64.urlsafe_b64decode(s.encode("ascii")).decode("utf-8"))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _list_runs() -> List[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "per_task").exists()
    )


def _list_task_ids(run_name: str) -> List[str]:
    pt = RUNS_DIR / run_name / "per_task"
    if not pt.exists():
        return []
    return sorted(
        d.name for d in pt.iterdir()
        if d.is_dir()
        and (
            any(d.glob("control_report*.json")) or
            any(d.glob("se_report*.json"))
        )
        and not (EXCLUDE_BASELINE and d.name.endswith("_00"))
    )


def _find_task_yaml(task_id: str) -> Optional[Path]:
    return _YAML_MAP.get(task_id)


def _list_annotators(run_name: str) -> List[str]:
    """Return sorted list of annotator IDs who have annotated any task in this run."""
    pt = RUNS_DIR / run_name / "per_task"
    if not pt.exists():
        return []
    annotators: set = set()
    for task_dir in pt.iterdir():
        if not task_dir.is_dir():
            continue
        for f in task_dir.iterdir():
            m = _ANNOTATOR_RE.match(f.name)
            if m:
                annotators.add(m.group(1))
    return sorted(annotators)


def _load_annotator_report(pt_dir: Path, annotator: str) -> Dict[str, Any]:
    """Load human_<annotator>_report.json; returns {} if missing or unreadable."""
    p = pt_dir / f"human_{annotator}_report.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_llm_report(pt_dir: Path, prefix: str) -> Dict[str, Any]:
    """Load a per-task LLM judge report.

    Prefers judge-tagged files (``prefix__<judge>.json``, sorted — last = most
    recent alphabetically) so new runs with multi-VLM naming are handled
    automatically. Falls back to the legacy untagged file (``prefix.json``)
    for old runs and human-eval init stubs.
    """
    tagged = sorted(pt_dir.glob(f"{prefix}__*.json"))
    if tagged:
        try:
            return json.loads(tagged[-1].read_text(encoding="utf-8"))
        except Exception:
            pass
    legacy = pt_dir / f"{prefix}.json"
    if legacy.exists():
        try:
            return json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_task_data(run_name: str, task_id: str, annotator: Optional[str] = None) -> Dict[str, Any]:
    pt_dir = RUNS_DIR / run_name / "per_task" / task_id

    # LLM reports — prefer judge-tagged files, fall back to legacy/stub names.
    cr   = _load_llm_report(pt_dir, "control_report")
    se   = _load_llm_report(pt_dir, "se_report")
    phys = _load_llm_report(pt_dir, "physics_report")

    if not cr and not se:
        raise FileNotFoundError(f"No control or se report found in: {pt_dir}")

    camera_controlled = any(n in run_name.lower() for n in _CAMERA_CONTROLLED_NAMES)

    # YAML: task_level and init_frame
    task_level: Optional[Any] = None
    init_frame_path: str = ""
    yaml_path = _find_task_yaml(task_id)
    if yaml_path:
        td = load_yaml(yaml_path)
        task_level = td.get("level")
        # Use yaml_path.stem (the actual variant name) so that base-name aliases
        # (e.g. task_id="ice_on_burner" -> yaml_path.stem="ice_on_burner_cardboard")
        # correctly locate the init frame on disk.
        init_frame_candidate = yaml_path.parent / f"{yaml_path.stem}_init_frame.png"
        if init_frame_candidate.exists():
            init_frame_path = str(init_frame_candidate)

    # WM prompts and requested fields from YAML
    video_wm_prompt: str = ""
    requested_trigger   = cr.get("requested_trigger",   "")
    requested_occlusion = cr.get("requested_occlusion", "")
    if _HAS_CONTROL_JUDGE and yaml_path:
        try:
            video_wm, camera_wm, camera_pose = _ctrl_load_task_fields(yaml_path)
            requested_trigger, requested_occlusion = _compute_requested_fields(
                video_wm, camera_wm, camera_pose, camera_controlled=camera_controlled
            )
            video_wm_prompt = (camera_wm if camera_controlled else video_wm) or ""
        except Exception:
            pass

    def furl(raw: str) -> Optional[str]:
        return f"/file?p={_enc(raw)}" if raw else None

    final_frame_path: str = ""
    ff_candidate = pt_dir / "final_frame.png"
    if ff_candidate.exists():
        final_frame_path = str(ff_candidate)

    llm_control = {
        "requested_occlusion": requested_occlusion,
        "requested_trigger":   requested_trigger,
        "occlusion_done":      cr.get("occlusion_done"),
        "trigger_applied":     cr.get("trigger_applied"),
        "state_evol":          se.get("llm_state_evol"),
        "physical_inaccuracy": phys.get("physical_inaccuracy"),
        "notes":               cr.get("notes", ""),
        "occlusion_prompt":    cr.get("occlusion_prompt", ""),
        "trigger_prompt":      cr.get("trigger_prompt", ""),
        "se_prompt":           se.get("filled_prompt1", ""),
        "physics_prompt":      phys.get("prompt", ""),
    }

    base: Dict[str, Any] = {
        "task_id":         task_id,
        "task_level":      task_level,
        "run_name":        run_name,
        "init_frame_url":  furl(init_frame_path),
        "final_frame_url": furl(final_frame_path),
        "video_url":       furl(str(_path_from_rel(cr["wm_video"])) if cr.get("wm_video") else ""),
        "video_wm_prompt": video_wm_prompt,
        "llm_control":     llm_control,
    }

    if annotator:
        # Annotator mode: load only this annotator's saved answers from their report file.
        # This is the primary persistence mechanism — the file is written by api_answer
        # and re-read every time the annotator loads a task, restoring their prior choices.
        rep = _load_annotator_report(pt_dir, annotator)
        human_control = {
            k: rep[k]
            for k in ("occlusion_done", "trigger_applied", "physical_inaccuracy")
            if k in rep
        }
        base["mode"] = "annotator"
        base["annotator"] = annotator
        base["human_control"] = human_control
        base["human_state_evol"] = rep.get("state_evol")  # None if not yet answered
    else:
        # Master mode: collect every annotator's answers from all human_*_report.json files.
        annotations: Dict[str, Any] = {}
        for f in sorted(pt_dir.iterdir()):
            m = _ANNOTATOR_RE.match(f.name)
            if m:
                ann_name = m.group(1)
                try:
                    rep = json.loads(f.read_text(encoding="utf-8"))
                    annotations[ann_name] = {
                        k: rep.get(k)
                        for k in ("occlusion_done", "trigger_applied", "state_evol", "physical_inaccuracy")
                    }
                except Exception:
                    pass
        base["mode"] = "master"
        base["annotator"] = None
        base["annotations"] = annotations

    return base


# ---------------------------------------------------------------------------
# Range-aware file serving (required for video seeking in the browser)
# ---------------------------------------------------------------------------

def _serve_range(path: Path) -> Response:
    file_size = path.stat().st_size
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    rng = request.headers.get("Range", "")

    if not rng or not rng.startswith("bytes="):
        with open(path, "rb") as f:
            data = f.read()
        r = Response(data, status=200, mimetype=mime)
        r.headers["Accept-Ranges"] = "bytes"
        r.headers["Content-Length"] = str(file_size)
        return r

    spec = rng[6:].strip()
    start_s, _, end_s = spec.partition("-")
    start  = int(start_s) if start_s else 0
    end    = int(end_s)   if end_s   else file_size - 1
    end    = min(end, file_size - 1)
    length = end - start + 1

    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(length)

    r = Response(data, status=206, mimetype=mime)
    r.headers["Content-Range"]  = f"bytes {start}-{end}/{file_size}"
    r.headers["Content-Length"] = str(length)
    r.headers["Accept-Ranges"]  = "bytes"
    return r


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(_HTML, status=200, mimetype="text/html; charset=utf-8")


@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())


@app.route("/api/runs/<run_name>/tasks")
def api_tasks(run_name: str):
    return jsonify(_list_task_ids(run_name))


@app.route("/api/runs/<run_name>/annotators")
def api_annotators(run_name: str):
    return jsonify(_list_annotators(run_name))


@app.route("/api/runs/<run_name>/random")
def api_random(run_name: str):
    ids = _list_task_ids(run_name)
    if not ids:
        abort(404, "No tasks found in run")
    return jsonify({"task_id": random.choice(ids), "total": len(ids)})


@app.route("/api/runs/<run_name>/task/<task_id>")
def api_task(run_name: str, task_id: str):
    annotator = request.args.get("annotator") or None
    try:
        return jsonify(_load_task_data(run_name, task_id, annotator=annotator))
    except FileNotFoundError as e:
        abort(404, str(e))
    except Exception as e:
        abort(500, str(e))


@app.route("/api/runs/<run_name>/task/<task_id>/answer", methods=["POST"])
def api_answer(run_name: str, task_id: str):
    annotator = request.args.get("annotator") or None
    if not annotator:
        abort(400, "Missing ?annotator= query parameter — master view is read-only")

    body = request.get_json(force=True) or {}
    pt_dir = RUNS_DIR / run_name / "per_task" / task_id
    pt_dir.mkdir(parents=True, exist_ok=True)

    # "MISSING" sentinel distinguishes absent key from explicitly null value.
    _se_raw = body.get("human_state_evol", "MISSING")
    has_se = _se_raw != "MISSING" and _se_raw is not None
    state_evol: Optional[bool] = bool(_se_raw) if has_se else None

    ctrl = body.get("control", {})

    # ── Write human_<annotator>_report.json ─────────────────────────────────
    report_path = pt_dir / f"human_{annotator}_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"annotator": annotator, "task_id": task_id}

    for field in ("occlusion_done", "trigger_applied", "physical_inaccuracy"):
        if field in ctrl:
            report[field] = ctrl[field]
    if has_se:
        report["state_evol"] = state_evol

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Update summary.json annotations.<annotator> ──────────────────────────
    if ctrl or has_se:
        summary_path = RUNS_DIR / run_name / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                task_entry = next(
                    (t for t in summary.get("tasks", []) if t.get("task_id") == task_id),
                    None,
                )
                if task_entry is not None:
                    ann_entry = task_entry.setdefault("annotations", {}).setdefault(annotator, {})
                    for field in ("occlusion_done", "trigger_applied", "physical_inaccuracy"):
                        if field in ctrl:
                            ann_entry[field] = ctrl[field]
                    if has_se:
                        ann_entry["state_evol"] = state_evol
                    summary_path.write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
            except Exception:
                pass  # summary update is best-effort

    return jsonify({"status": "ok"})


@app.route("/file")
def serve_file():
    p = request.args.get("p", "")
    if not p:
        abort(400, "Missing p parameter")
    try:
        path = _dec(p)
    except Exception:
        abort(400, "Bad path encoding")
    if not path.exists():
        abort(404, "File not found")
    try:
        return _serve_range(path)
    except Exception as e:
        abort(500, str(e))


# ---------------------------------------------------------------------------
# Inline HTML / CSS / JS
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>STEVO-Bench Human Eval</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --bg:          #0d1117;
      --card-bg:     #161b22;
      --card-head:   #21262d;
      --border:      #30363d;
      --text:        #c9d1d9;
      --muted:       #8b949e;
      --accent:      #58a6ff;
      --green:       #3fb950;
      --red:         #f85149;
    }
    * { box-sizing: border-box; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; min-height: 100vh; }

    /* ── Top bar ── */
    #topbar {
      background: var(--card-bg); border-bottom: 1px solid var(--border);
      padding: 8px 18px; display: flex; align-items: center; gap: 10px;
      position: sticky; top: 0; z-index: 100; flex-wrap: wrap;
    }
    #topbar .brand { font-weight: 700; font-size: 15px; color: var(--accent); white-space: nowrap; }
    #topbar select, #topbar input[type=text] {
      background: var(--card-head); color: var(--text);
      border: 1px solid var(--border); border-radius: 6px;
      padding: 5px 10px; font-size: 13px;
    }
    #topbar input[type=text] { width: 240px; }
    #task-search-wrap { position: relative; }
    #search-dropdown {
      position: absolute; top: calc(100% + 4px); left: 0; min-width: 260px;
      background: var(--card-head); border: 1px solid var(--border);
      border-radius: 6px; max-height: 220px; overflow-y: auto; z-index: 200; display: none;
    }
    .search-item { padding: 7px 12px; cursor: pointer; font-size: 13px; }
    .search-item:hover { background: var(--border); }
    #task-info { color: var(--muted); font-size: 12px; white-space: nowrap; }
    #task-info .tid { color: var(--accent); font-weight: 600; }
    .spacer { flex: 1; }
    .save-chip {
      font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 10px;
      display: none;
    }
    .save-chip.saving { background: #d29922; color: #000; display: inline-block; }
    .save-chip.saved  { background: var(--green); color: #000; display: inline-block; }
    .save-chip.error  { background: var(--red);   color: #fff; display: inline-block; }

    /* ── Buttons ── */
    .btn-primary-sm {
      background: var(--accent); color: #0d1117; border: none;
      padding: 6px 16px; border-radius: 6px; font-weight: 600; font-size: 13px; cursor: pointer;
    }
    .btn-primary-sm:hover { background: #79c0ff; }
    .btn-primary-sm:disabled { opacity: 0.45; cursor: default; }

    /* ── Card ── */
    .ev-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 14px; overflow: hidden; }
    .ev-card-header { background: var(--card-head); padding: 7px 14px; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--accent); border-bottom: 1px solid var(--border); }
    .ev-card-body { padding: 10px; }

    /* ── Verdict row ── */
    .verdict-row { display: flex; gap: 10px; margin-bottom: 14px; }
    .v-box {
      flex: 1; text-align: center; padding: 12px 8px; border-radius: 8px;
      font-size: 15px; font-weight: 700; border: 2px solid var(--border);
      background: var(--card-head); color: var(--muted);
    }
    .v-pass { background: rgba(63,185,80,0.12); border-color: var(--green); color: var(--green); }
    .v-fail { background: rgba(248,81,73,0.12);  border-color: var(--red);   color: var(--red); }
    .v-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 400; margin-bottom: 2px; }

    /* ── Answer badge ── */
    .abadge {
      display: inline-block; padding: 2px 9px; border-radius: 10px;
      font-size: 11px; font-weight: 600; text-transform: uppercase;
    }
    .ab-yes { background: rgba(63,185,80,0.18); color: var(--green); border: 1px solid var(--green); }
    .ab-no  { background: rgba(248,81,73,0.18); color: var(--red);   border: 1px solid var(--red); }
    .ab-unk { background: rgba(139,148,158,0.15); color: var(--muted); border: 1px solid var(--border); }

    /* ── T / F toggle buttons ── */
    .tf-wrap { display: flex; gap: 4px; justify-content: center; }
    .btn-tf {
      padding: 3px 13px; font-size: 12px; font-weight: 700; border-radius: 5px;
      border: 1.5px solid var(--border); background: transparent;
      color: var(--muted); cursor: pointer; transition: all 0.12s;
    }
    .btn-tf:hover { background: var(--card-head); }
    .btn-tf.on-t  { background: rgba(63,185,80,0.2); border-color: var(--green); color: var(--green); }
    .btn-tf.on-f  { background: rgba(248,81,73,0.2); border-color: var(--red);   color: var(--red); }

    /* ── Question table ── */
    .q-table { width: 100%; border-collapse: separate; border-spacing: 0 5px; }
    .q-table th {
      color: var(--muted); font-size: 11px; text-transform: uppercase;
      letter-spacing: 0.5px; padding: 4px 8px; font-weight: 600;
    }
    .q-row td {
      background: var(--card-head); padding: 9px 10px;
      border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }
    .q-row td:first-child { border-left: 1px solid var(--border); border-radius: 6px 0 0 6px; }
    .q-row td:last-child  { border-right: 1px solid var(--border); border-radius: 0 6px 6px 0; }
    .q-text  { font-size: 13px; line-height: 1.4; }
    .q-notes { font-size: 11px; color: var(--muted); margin-top: 3px; font-style: italic; }
    .conf-pct { font-size: 10px; color: var(--muted); margin-top: 2px; }

    /* ── Control rows ── */
    .ctrl-row {
      display: flex; align-items: flex-start; gap: 10px; padding: 10px;
      background: var(--card-head); border-radius: 6px; margin-bottom: 8px;
      border: 1px solid var(--border); flex-wrap: wrap;
    }
    .ctrl-body { flex: 1; }
    .ctrl-lbl  { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 3px; }
    .ctrl-val  { font-size: 13px; }
    .ctrl-llm  { display: flex; flex-direction: column; align-items: center; gap: 3px; min-width: 54px; }
    .ctrl-human { display: flex; flex-direction: column; align-items: center; gap: 3px; min-width: 66px; }
    .ctrl-notes {
      font-size: 12px; color: var(--muted); font-style: italic;
      padding: 7px 10px; background: var(--bg); border-radius: 5px;
      border-left: 3px solid var(--border); margin-top: 4px;
    }

    /* ── Media ── */
    #init-frame   { width: 100%; max-height: 340px; object-fit: contain; border-radius: 5px; background: #000; display: block; }
    #output-video { width: 100%; border-radius: 5px; background: #000; display: block; }
    .media-ph {
      width: 100%; height: 180px; background: var(--card-head); border-radius: 5px;
      border: 1px dashed var(--border); display: flex; align-items: center;
      justify-content: center; color: var(--muted); font-size: 13px;
    }
    .prompt-box {
      background: var(--card-head); border: 1px solid var(--border); border-radius: 6px;
      padding: 11px; font-size: 13px; line-height: 1.6; white-space: pre-wrap;
      color: var(--text); max-height: 200px; overflow-y: auto;
    }

    /* ── Prompt dropdown ── */
    .prompt-details {
      flex-basis: 100%; margin-top: 4px; padding-top: 6px;
      border-top: 1px solid var(--border);
    }
    .prompt-details summary {
      font-size: 13px; color: var(--muted); cursor: pointer;
      user-select: none; list-style: none; display: inline-flex; align-items: center; gap: 4px;
    }
    .prompt-details summary::before { content: '▶'; font-size: 10px; transition: transform 0.15s; }
    .prompt-details[open] summary::before { transform: rotate(90deg); }
    .prompt-details summary:hover { color: var(--accent); }
    .prompt-details .prompt-text {
      font-size: 13px; white-space: pre-wrap; color: var(--muted);
      background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
      padding: 10px 12px; margin-top: 6px; max-height: 260px; overflow-y: auto; line-height: 1.6;
      width: 100%;
    }

    /* ── Mode chip ── */
    .mode-chip {
      font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 10px;
      white-space: nowrap;
    }
    .mode-chip.annotator { background: rgba(88,166,255,0.15); color: var(--accent); border: 1px solid var(--accent); }
    .mode-chip.master    { background: rgba(139,148,158,0.15); color: var(--muted); border: 1px solid var(--border); }

    /* ── Annotator badge columns ── */
    .ctrl-ann {
      display: flex; flex-direction: column; align-items: center; gap: 3px;
      min-width: 54px;
    }

    /* ── Misc ── */
    #empty-state { text-align: center; padding: 80px 20px; color: var(--muted); }
    #empty-state .big { font-size: 52px; }
    #empty-state .msg { font-size: 20px; margin-top: 10px; }
    #loading-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,0.45);
      display: none; align-items: center; justify-content: center; z-index: 999;
    }
    .spinner {
      width: 42px; height: 42px; border: 4px solid var(--border);
      border-top-color: var(--accent); border-radius: 50%;
      animation: spin 0.75s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ── Nav buttons ── */
    .btn-nav {
      background: var(--card-head); color: var(--text); border: 1px solid var(--border);
      padding: 6px 13px; border-radius: 6px; font-weight: 600; font-size: 13px; cursor: pointer;
      white-space: nowrap;
    }
    .btn-nav:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
    .btn-nav:disabled { opacity: 0.35; cursor: default; }

    /* ── Progress bar ── */
    #progress-wrap { display: flex; align-items: center; gap: 8px; }
    #progress-label { font-size: 12px; color: var(--muted); white-space: nowrap; }
    #progress-bar-bg {
      width: 100px; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
    }
    #progress-bar-fill {
      height: 100%; background: var(--accent); border-radius: 3px;
      width: 0%; transition: width 0.2s;
    }
  </style>
</head>
<body>

<!-- Loading overlay -->
<div id="loading-overlay"><div class="spinner"></div></div>

<!-- Top bar -->
<div id="topbar">
  <span class="brand">STEVO-Bench Eval</span>

  <select id="run-select">
    <option value="">— Select run —</option>
  </select>

  <div id="task-search-wrap">
    <input type="text" id="task-search" placeholder="Search task ID…" autocomplete="off" disabled>
    <div id="search-dropdown"></div>
  </div>

  <button class="btn-nav" id="btn-prev" disabled>← Prev</button>
  <button class="btn-nav" id="btn-next" disabled>Next →</button>

  <span class="spacer"></span>

  <span id="mode-chip" class="mode-chip" style="display:none"></span>

  <span id="task-info" style="display:none">
    <span class="tid" id="info-tid"></span>
    <span id="info-level" style="color:var(--muted); font-size:12px;"></span>
  </span>

  <div id="progress-wrap" style="display:none">
    <span id="progress-label"></span>
    <div id="progress-bar-bg"><div id="progress-bar-fill"></div></div>
  </div>

  <span id="save-chip" class="save-chip"></span>
</div>

<!-- Empty state -->
<div id="empty-state">
  <div class="big">🔍</div>
  <div class="msg">Select a run to begin</div>
</div>

<!-- Task view -->
<div id="task-view" style="display:none; padding: 16px 20px; max-width: 1400px; margin: 0 auto;">
  <div class="row g-3">

    <!-- Left column: media -->
    <div class="col-lg-5">

      <div class="ev-card">
        <div class="ev-card-header">Initial Frame</div>
        <div class="ev-card-body">
          <img id="init-frame" alt="Initial frame" style="display:none">
          <div id="init-ph" class="media-ph">No image available</div>
        </div>
      </div>

      <div class="ev-card">
        <div class="ev-card-header">Output Video</div>
        <div class="ev-card-body">
          <video id="output-video" controls style="display:none"></video>
          <div id="video-ph" class="media-ph">No video available</div>
        </div>
      </div>

      <div class="ev-card">
        <div class="ev-card-header">Video WM Prompt</div>
        <div class="ev-card-body">
          <div id="video-prompt" class="prompt-box"></div>
        </div>
      </div>

    </div><!-- /left -->

    <!-- Right column: evaluation -->
    <div class="col-lg-7">

      <!-- Verdicts (rendered dynamically by refreshVerdicts()) -->
      <div id="verdict-row" class="verdict-row"></div>

      <!-- Verifier -->
      <div class="ev-card">
        <div class="ev-card-header">Verifier</div>
        <div class="ev-card-body" id="ctrl-body">
          <div style="color:var(--muted); font-size:13px;">No control data</div>
        </div>
      </div>

    </div><!-- /right -->
  </div>
</div>

<script>
// =============================================================================
// Mode detection — read from URL query param once at page load.
// Annotator URL : http://host:port/?annotator=alice
// Master URL    : http://host:port/
// =============================================================================
const ANNOTATOR = new URLSearchParams(location.search).get('annotator') || null;
const IS_MASTER = !ANNOTATOR;

// =============================================================================
// State
// =============================================================================
let S = {
  run:            null,   // string
  taskId:         null,   // string
  taskIndex:      -1,     // current position in allTaskIds
  llmControl:     {},     // LLM judge answers
  humanControl:   {},     // this annotator's answers (annotator mode only)
  humanStateEvol: null,   // this annotator's state-evol answer (annotator mode only)
  annotations:    {},     // all annotators' answers keyed by name (master mode only)
  allTaskIds:     [],     // ordered list for sequential navigation
};

let _saveTimer   = null;
let _saving      = false;  // true while a save attempt (or retry loop) is in flight
let _pendingSave = false;  // true if a new change arrived while _saving was true

// =============================================================================
// Initialise
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
  // Show mode chip
  const chip = document.getElementById('mode-chip');
  if (IS_MASTER) {
    chip.textContent = 'Master View';
    chip.classList.add('master');
  } else {
    chip.textContent = 'Annotator: ' + ANNOTATOR;
    chip.classList.add('annotator');
  }
  chip.style.display = '';

  loadRuns();

  document.getElementById('run-select').addEventListener('change', e => {
    if (e.target.value) selectRun(e.target.value);
  });
  document.getElementById('btn-prev').addEventListener('click', loadPrevTask);
  document.getElementById('btn-next').addEventListener('click', loadNextTask);

  // Task search
  const inp = document.getElementById('task-search');
  const dd  = document.getElementById('search-dropdown');
  inp.addEventListener('input', () => {
    const q = inp.value.trim().toLowerCase();
    if (!q || !S.allTaskIds.length) { dd.style.display = 'none'; return; }
    const hits = S.allTaskIds.filter(id => id.toLowerCase().includes(q)).slice(0, 25);
    if (!hits.length) { dd.style.display = 'none'; return; }
    dd.innerHTML = hits.map(id =>
      `<div class="search-item" onclick="pickTask('${esc(id)}')">${esc(id)}</div>`
    ).join('');
    dd.style.display = 'block';
  });
  document.addEventListener('click', e => {
    if (!document.getElementById('task-search-wrap').contains(e.target))
      dd.style.display = 'none';
  });
});

// =============================================================================
// API helpers
// =============================================================================
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

// Build task API URL — include ?annotator= in annotator mode
function taskUrl(runName, taskId) {
  const base = `/api/runs/${enc(runName)}/task/${enc(taskId)}`;
  return ANNOTATOR ? `${base}?annotator=${enc(ANNOTATOR)}` : base;
}

// =============================================================================
// Run selection
// =============================================================================
async function loadRuns() {
  const runs = await api('/api/runs');
  const sel = document.getElementById('run-select');
  sel.innerHTML = '<option value="">— Select run —</option>' +
    runs.map(r => `<option value="${esc(r)}">${esc(r)}</option>`).join('');
}

async function selectRun(runName) {
  S.run = runName;
  setLoading(true);
  try {
    S.allTaskIds = await api(`/api/runs/${enc(runName)}/tasks`);
    document.getElementById('task-search').disabled = false;
    if (S.allTaskIds.length > 0) {
      await navigateToIndex(0);
    }
  } catch(e) {
    alert('Failed to load run: ' + e.message);
  } finally {
    setLoading(false);
  }
}

// =============================================================================
// Task navigation
// =============================================================================
async function navigateToIndex(idx) {
  S.taskIndex = idx;
  setLoading(true);
  try {
    await loadTask(S.allTaskIds[idx]);
  } catch(e) {
    alert('Failed to load task: ' + e.message);
  } finally {
    setLoading(false);
  }
}

async function loadPrevTask() {
  if (!S.run || S.taskIndex <= 0) return;
  await navigateToIndex(S.taskIndex - 1);
}

async function loadNextTask() {
  if (!S.run || S.taskIndex >= S.allTaskIds.length - 1) return;
  await navigateToIndex(S.taskIndex + 1);
}

async function pickTask(taskId) {
  document.getElementById('search-dropdown').style.display = 'none';
  document.getElementById('task-search').value = taskId;
  const idx = S.allTaskIds.indexOf(taskId);
  if (idx !== -1) S.taskIndex = idx;
  setLoading(true);
  try {
    await loadTask(taskId);
  } finally {
    setLoading(false);
  }
}

async function loadTask(taskId) {
  // Fetching the task loads the annotator's saved answers from disk, ensuring
  // session persistence across browser reloads and return visits.
  const data = await api(taskUrl(S.run, taskId));
  S.taskId     = taskId;
  S.llmControl = data.llm_control || {};

  if (IS_MASTER) {
    S.annotations    = data.annotations || {};
    S.humanControl   = {};
    S.humanStateEvol = null;
  } else {
    // Restore this annotator's previously saved choices (may all be null on first visit).
    S.humanControl   = { ...data.human_control };
    S.humanStateEvol = data.human_state_evol ?? null;
    S.annotations    = {};
  }

  updateInfoBar(taskId, data.task_level);
  render(data);
  hideSave();
}

function updateInfoBar(taskId, level) {
  document.getElementById('info-tid').textContent = taskId;
  document.getElementById('info-level').textContent = level != null ? ` · Level ${level}` : '';
  document.getElementById('task-info').style.display = '';
  updateNav();
}

function updateNav() {
  const n = S.allTaskIds.length;
  const i = S.taskIndex;
  document.getElementById('btn-prev').disabled = (i <= 0);
  document.getElementById('btn-next').disabled = (i >= n - 1);
  if (n > 0) {
    document.getElementById('progress-label').textContent = `${i + 1} / ${n}`;
    document.getElementById('progress-bar-fill').style.width = ((i + 1) / n * 100).toFixed(1) + '%';
    document.getElementById('progress-wrap').style.display = '';
  }
}

// =============================================================================
// Rendering
// =============================================================================
function render(data) {
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('task-view').style.display   = '';

  // Init frame
  const img = document.getElementById('init-frame');
  const iph = document.getElementById('init-ph');
  if (data.init_frame_url) {
    img.src = data.init_frame_url; img.style.display = ''; iph.style.display = 'none';
  } else {
    img.style.display = 'none'; iph.style.display = '';
  }

  // Video
  const vid = document.getElementById('output-video');
  const vph = document.getElementById('video-ph');
  if (data.video_url) {
    vid.src = data.video_url; vid.load(); vid.style.display = ''; vph.style.display = 'none';
  } else {
    vid.style.display = 'none'; vph.style.display = '';
  }

  // Prompt
  document.getElementById('video-prompt').textContent = data.video_wm_prompt || '(none)';

  renderControl();
  refreshVerdicts();
}

// Build one ctrl-row. `humanCol` is the HTML string for the rightmost column(s).
function ctrlRow(labelText, questionHtml, promptHtml, llmVal, humanCol) {
  const llmColHtml = IS_MASTER
    ? `<div class="ctrl-llm"><div class="ctrl-lbl">LLM</div>${boolBadge(llmVal)}</div>`
    : '';
  return `
    <div class="ctrl-row">
      <div class="ctrl-body">
        <div class="ctrl-lbl">${labelText}</div>
        <div class="ctrl-val">${questionHtml}</div>
        ${promptHtml}
      </div>
      ${llmColHtml}
      ${humanCol}
    </div>`;
}

// Build annotator column(s) for a given field.
// Annotator mode: T/F buttons for the current annotator.
// Master mode: one read-only badge per annotator.
function humanCols(field, isStateEvol) {
  if (IS_MASTER) {
    const names = Object.keys(S.annotations).sort();
    if (!names.length) return '';
    return names.map(name => {
      const val = isStateEvol
        ? S.annotations[name]?.state_evol
        : S.annotations[name]?.[field];
      const initials = name.slice(0, 6);
      return `<div class="ctrl-ann"><div class="ctrl-lbl">${esc(initials)}</div>${boolBadge(val)}</div>`;
    }).join('');
  }

  // Annotator mode
  const cur = isStateEvol ? S.humanStateEvol : S.humanControl[field];
  const onT  = isStateEvol ? `setStateEvol(true)`        : `setControl('${field}',true)`;
  const onF  = isStateEvol ? `setStateEvol(false)`       : `setControl('${field}',false)`;
  return `
    <div class="ctrl-human">
      <div class="ctrl-lbl">Your answer</div>
      <div class="tf-wrap">
        <button class="btn-tf ${cur===true ?'on-t':''}" onclick="${onT}">T</button>
        <button class="btn-tf ${cur===false?'on-f':''}" onclick="${onF}">F</button>
      </div>
    </div>`;
}

function renderControl() {
  const body = document.getElementById('ctrl-body');
  const c = S.llmControl;
  const hasData = c.occlusion_done != null || c.trigger_applied != null ||
                  c.state_evol != null || c.physical_inaccuracy != null ||
                  c.requested_occlusion || c.requested_trigger;
  if (!hasData) {
    body.innerHTML = '<div style="color:var(--muted);font-size:13px;">No control data</div>';
    return;
  }

  const pd = (txt) => txt
    ? `<details class="prompt-details"><summary>View full prompt</summary><div class="prompt-text">${esc(txt)}</div></details>`
    : '';

  body.innerHTML =
    ctrlRow(
      'Occlusion method',
      `Does the video correctly apply the following occlusion method:<br><strong style="color:var(--red)">${esc(c.requested_occlusion||'—')}</strong><br>so the main object(s) become <strong>COMPLETELY</strong> invisible <strong>DURING</strong> the process for a <strong>PROLONGED</strong> time, then <strong>REMOVED</strong> to reveal the objects again? Missing anyone of the four means False. If the process ends <strong>BEFORE</strong> occlusion takes place, that's a False.`,
      '',
      c.occlusion_done,
      humanCols('occlusion_done', false)
    ) +
    ctrlRow(
      'Trigger / action',
      `Does the following trigger action correctly follow in the video:<br><strong style="color:var(--red)">${esc(c.requested_trigger||'—')}</strong>?`,
      '',
      c.trigger_applied,
      humanCols('trigger_applied', false)
    ) +
    ctrlRow(
      'State Evolution',
      'Does the scene state evolve as expected? Click on the dropdown to see the full definition of successful state evolution and follow all guidance to make your decision.',
      pd(c.se_prompt),
      c.state_evol,
      humanCols('state_evol', true)
    ) +
    ctrlRow(
      'Physical inaccuracy',
      'Does the video contain a physically impossible event in the visible portions? This includes: (1) instantaneous violations — object deforms/teleports/changes material with no cause; (2) dynamic violations — cause is shown but wrong effect follows (e.g. water poured in but level drops); (3) continuity violations — background changes instantaneously or scene resets after a blackout. Do NOT flag the intended change not happening — that is a separate metric. Click the prompt dropdown for full guidance.',
      pd(c.physics_prompt),
      c.physical_inaccuracy,
      humanCols('physical_inaccuracy', false)
    ) +
    (c.notes ? `<div class="ctrl-notes">${esc(c.notes)}</div>` : '');
}

// =============================================================================
// Human answer handlers  (annotator mode only)
// =============================================================================
function setControl(field, value) {
  if (IS_MASTER) return;
  S.humanControl[field] = value;
  // Re-render just the human column buttons (cheapest: full re-render of ctrl-body)
  renderControl();
  refreshVerdicts();
  scheduleSave();
}

function setStateEvol(value) {
  if (IS_MASTER) return;
  // Toggle off if tapping the same value again
  S.humanStateEvol = (S.humanStateEvol === value) ? null : value;
  renderControl();
  refreshVerdicts();
  scheduleSave();
}

// =============================================================================
// Verdicts
// =============================================================================
function computeVerdict(occlusion_done, trigger_applied, state_evol, physical_inaccuracy) {
  const vals = [occlusion_done, trigger_applied, state_evol, physical_inaccuracy];
  if (vals.every(v => v === null || v === undefined)) return 'unk';
  if (vals.some(v => v === null || v === undefined)) return 'partial';
  if (occlusion_done === false || trigger_applied === false || state_evol === false ||
      physical_inaccuracy === true) return 'fail';
  if (occlusion_done === true && trigger_applied === true && state_evol === true &&
      physical_inaccuracy === false) return 'pass';
  return 'partial';
}

function verdictBox(label, state) {
  const cls = state === 'pass' ? ' v-pass' : state === 'fail' ? ' v-fail' : '';
  const txt = { pass: 'PASS ✓', fail: 'FAIL ✗', partial: 'PARTIAL', unk: '—' }[state] || '—';
  return `<div class="v-box${cls}"><div class="v-label">${esc(label)}</div><span>${txt}</span></div>`;
}

function refreshVerdicts() {
  const row = document.getElementById('verdict-row');
  const c = S.llmControl;

  if (IS_MASTER) {
    // LLM verdict + one box per annotator
    const llmState = computeVerdict(c.occlusion_done, c.trigger_applied, c.state_evol, c.physical_inaccuracy);
    let html = verdictBox('LLM Verdict', llmState);
    for (const [name, ann] of Object.entries(S.annotations).sort()) {
      const state = computeVerdict(ann.occlusion_done, ann.trigger_applied, ann.state_evol, ann.physical_inaccuracy);
      html += verdictBox(name, state);
    }
    row.innerHTML = html;
  } else {
    // Single "Your Verdict" box
    const state = computeVerdict(
      S.humanControl.occlusion_done, S.humanControl.trigger_applied,
      S.humanStateEvol, S.humanControl.physical_inaccuracy
    );
    row.innerHTML = verdictBox('Your Verdict', state);
  }
}

// =============================================================================
// Save (debounced) — annotator mode only
// =============================================================================
function scheduleSave() {
  if (IS_MASTER) return;
  clearTimeout(_saveTimer);
  showSave('saving');
  _saveTimer = setTimeout(doSave, 450);
}

async function doSave() {
  if (IS_MASTER || !S.run || !S.taskId) return;
  if (_saving) { _pendingSave = true; return; }
  _saving = true;
  _pendingSave = false;

  const savedRun    = S.run;
  const savedTaskId = S.taskId;
  let delay = 2000;  // ms before first retry; doubles each time, capped at 30 s

  try {
    while (true) {
      // Annotator navigated away — stop retrying (their new task will save separately)
      if (S.run !== savedRun || S.taskId !== savedTaskId) return;
      try {
        await api(
          `/api/runs/${enc(savedRun)}/task/${enc(savedTaskId)}/answer?annotator=${enc(ANNOTATOR)}`,
          {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({
              control:          S.humanControl,
              human_state_evol: S.humanStateEvol,
            }),
          }
        );
        showSave('saved');
        setTimeout(hideSave, 2500);
        return;
      } catch(e) {
        console.error(`Save failed, retrying in ${delay / 1000}s:`, e);
        showSave('retrying');
        await new Promise(r => setTimeout(r, delay));
        delay = Math.min(delay * 2, 30000);
        showSave('saving');
      }
    }
  } finally {
    _saving = false;
    if (_pendingSave) { _pendingSave = false; doSave(); }
  }
}

// =============================================================================
// Badge helpers
// =============================================================================
function boolBadge(val) {
  if (val === true)  return '<span class="abadge ab-yes">T</span>';
  if (val === false) return '<span class="abadge ab-no">F</span>';
  return '<span class="abadge ab-unk">?</span>';
}

// =============================================================================
// UI utilities
// =============================================================================
function showSave(state) {
  const el = document.getElementById('save-chip');
  el.className = 'save-chip ' + (state === 'retrying' ? 'saving' : state);
  el.textContent = { saving: 'Saving\u2026', saved: 'Saved \u2713', retrying: 'Retrying\u2026', error: 'Save failed!' }[state] || '';
}
function hideSave() {
  document.getElementById('save-chip').className = 'save-chip';
}
function setLoading(on) {
  document.getElementById('loading-overlay').style.display = on ? 'flex' : 'none';
}
function enc(s) { return encodeURIComponent(s); }
function esc(s) {
  return String(s || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global RUNS_DIR, TASKS_ROOT, EXCLUDE_BASELINE

    parser = argparse.ArgumentParser(description="STEVO-Bench Human Eval UI")
    parser.add_argument("--runs_dir",         default="runs",            help="Path to runs directory (default: runs/)")
    parser.add_argument("--tasks_dir",        default="benchmark/tasks_v4/", help="Path to benchmark tasks root (default: benchmark/tasks/)")
    parser.add_argument("--host",             default="0.0.0.0",         help="Host to bind (default: 0.0.0.0 — all interfaces)")
    parser.add_argument("--port",             default=7860, type=int,    help="Port to listen on (default: 7860)")
    parser.add_argument("--exclude_baseline", action="store_true",       help="Hide tasks ending in _00 (baseline variants) from all views")
    args = parser.parse_args()

    RUNS_DIR          = Path(args.runs_dir).expanduser().resolve()
    TASKS_ROOT        = Path(args.tasks_dir).expanduser().resolve()
    EXCLUDE_BASELINE  = args.exclude_baseline

    print("STEVO-Bench Human Eval UI")
    print(f"  Runs dir  : {RUNS_DIR}")
    print(f"  Tasks dir : {TASKS_ROOT}")
    print()

    _build_yaml_map()

    display_host = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"  Master URL     : http://{display_host}:{args.port}/")
    print(f"  Annotator URLs : http://{display_host}:{args.port}/?annotator=<name>")
    print()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
