# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed Studio replay with isolated UDP impairment and timing gates."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.observability.realtime_timing import TimingDistribution
from worldfoundry.runtime.realtime_regression import (
    RealtimeRegressionCase,
    RealtimeRegressionThresholds,
    RegressionCheck,
    evaluate_realtime_case,
    read_realtime_trace,
)


def validate_profile(profile: Mapping[str, Any]) -> None:
    """Reject invalid replay budgets and link settings before loading weights."""
    for name in ("steps", "repeats", "warmup_steps", "timeout_ms"):
        value = profile[name]
        if type(value) is not int or value < (0 if name == "warmup_steps" else 1):
            raise ValueError(f"Invalid replay {name}: {value}")
    names = [item["name"] for item in profile["networks"]]
    if not names or len(set(names)) != len(names) or not all(names):
        raise ValueError("Network names must be nonempty and unique")
    for network in profile["networks"]:
        for direction in ("uplink", "downlink"):
            link = network[direction]
            for name in ("delay_ms", "jitter_ms", "rate_bps", "loss", "queue_ms"):
                value = link[name]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {network['name']}.{direction}.{name}")
            if link["loss"] > 1:
                raise ValueError("Link loss must be a fraction between zero and one")
        if "outage" in network:
            outage = network["outage"]
            if type(outage["after_steps"]) is not int or not 0 <= outage["after_steps"] < profile["steps"]:
                raise ValueError("Outage must begin during measured STEP calls")
            if not math.isfinite(outage["duration_ms"]) or outage["duration_ms"] <= 0:
                raise ValueError("Outage duration_ms must be positive and finite")
    allowed = {"max_e2e_p90_ms", "max_stall_fraction", "max_reset_ms"}
    for name, value in profile.get("delivery_thresholds", {}).items():
        if name not in allowed or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid delivery threshold: {name}")
        if name == "max_stall_fraction" and value > 1:
            raise ValueError("max_stall_fraction must be between zero and one")
    RealtimeRegressionThresholds.from_dict(profile["thresholds"])


def _distribution(values: list[float]) -> dict[str, Any] | None:
    return TimingDistribution.from_values(values).to_payload() if values else None


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_delivery(session: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize first correlated control presentations, excluding warmup STEP calls."""
    observed = session.get("observations") or {}
    sent = observed.get("sent", [])
    commands = [item for item in sent if item.get("type") == "action" and item.get("action", {}).get("event") == "step"]
    ids = [item["control_id"] for item in commands[profile["warmup_steps"] :]]
    presented = {item["control_id"]: item for item in sent if item.get("type") == "frame_presented"}
    latencies = [float(presented[key]["control_to_display_ms"]) for key in ids if key in presented]
    start, end = observed.get("measuredStart"), observed.get("measuredEnd")
    callbacks = observed.get("callbacks", [])
    fps = (session.get("spec") or {}).get("fps", 0)
    stall_ms = 0.0
    stalls = 0
    if start is not None and end is not None and fps > 0:
        points = [start, *(value for value in callbacks if start <= value <= end), end]
        interval = 1000 / fps
        for left, right in zip(points, points[1:]):
            if right - left > max(500, 3 * interval):
                stalls += 1
                stall_ms += right - left - interval
    chunks = [
        item
        for item in observed.get("received", [])
        if item["type"] == "chunk_done" and item["chunk_index"] > profile["warmup_steps"]
    ]
    displayed = {
        (item["chunk_index"], item["frame_index"])
        for item in observed.get("frames", [])
        if item["chunk_index"] > profile["warmup_steps"]
    }
    return {
        "control_ids": ids,
        "submitted_controls": len(ids),
        "presented_controls": len(latencies),
        "missing_controls": len(ids) - len(latencies),
        "control_to_display_ms": _distribution(latencies),
        "generated_frames": sum(item["frames"] for item in chunks),
        "unique_generated_frames_presented": len(displayed),
        "presentation_stalls": stalls,
        "presentation_stall_ms": stall_ms,
        "presentation_stall_fraction": stall_ms / (end - start)
        if start is not None and end is not None and end > start
        else None,
    }


def _resources(samples: list[dict[str, Any]], session: Mapping[str, Any]) -> dict[str, Any]:
    selected = [
        item
        for item in samples
        if session.get("measured_start_wall_ms", float("inf"))
        <= item["wall_ms"]
        <= session.get("measured_end_wall_ms", 0)
    ]
    cpu = None
    if len(selected) >= 2:
        cpu = (
            100_000
            * (selected[-1]["cpu_s"] - selected[0]["cpu_s"])
            / (selected[-1]["wall_ms"] - selected[0]["wall_ms"])
        )
    return {
        "cpu_percent_one_core": cpu,
        "peak_rss_bytes": max((item["rss_bytes"] for item in selected), default=None),
        "peak_gpu_process_bytes": max(
            (item["gpu_bytes"] for item in selected if item.get("gpu_bytes") is not None), default=None
        ),
    }


def build_report(
    profile: Mapping[str, Any], browser: Mapping[str, Any], trace: Path, resources: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reuse chunk gates and require matching browser/server delivery evidence."""
    events = _events(trace)
    samples = read_realtime_trace(trace)
    presentations = {
        event["fields"]["control_id"]: event["fields"]
        for event in events
        if event.get("event") == "realtime.frame_presented"
    }
    results = []
    limits = profile.get("delivery_thresholds", {})
    for session in browser["sessions"]:
        delivery = summarize_delivery(session, profile)
        receipts = [presentations[key] for key in delivery["control_ids"] if key in presentations]
        sessions = {item["session_id"] for item in receipts}
        session_id = next(iter(sessions)) if len(sessions) == 1 else "missing-session"
        case = RealtimeRegressionCase(
            name=session["name"],
            model_id=profile["model_id"],
            transport="webrtc",
            session_id=session_id,
            thresholds=RealtimeRegressionThresholds.from_dict(profile["thresholds"]),
        )
        server = evaluate_realtime_case(case, samples)
        checks = [
            RegressionCheck("submitted_controls", ">=", profile["steps"], delivery["submitted_controls"]),
            RegressionCheck("presented_controls", ">=", profile["steps"], delivery["presented_controls"]),
            RegressionCheck("missing_controls", "<=", 0, delivery["missing_controls"]),
            RegressionCheck("server_delivery_receipts", ">=", profile["steps"], len(receipts)),
            RegressionCheck("session_count", "<=", 1, len(sessions) if sessions else None),
            RegressionCheck("reset_clean", ">=", 1, int(session.get("reset_clean", False))),
            RegressionCheck("errors", "<=", 0, len(session.get("errors", [])) + len(session["link"]["errors"])),
        ]
        distribution = delivery["control_to_display_ms"]
        for key, actual in (
            ("max_e2e_p90_ms", None if distribution is None else distribution["p90_ms"]),
            ("max_stall_fraction", delivery["presentation_stall_fraction"]),
            ("max_reset_ms", session.get("reset_ms")),
        ):
            if key in limits:
                checks.append(RegressionCheck(key, "<=", limits[key], actual))
        results.append(
            {
                "name": session["name"],
                "network": session["network"],
                "repeat": session["repeat"],
                "passed": server.passed and all(check.passed for check in checks),
                "server": server.to_dict(),
                "delivery": delivery,
                "checks": [check.to_dict() for check in checks],
                "resources": _resources(resources, session),
                "rtc": session.get("rtc"),
                "link": session["link"],
                "errors": session.get("errors", []),
            }
        )
    expected = {
        f"{network['name']}-{repeat + 1}" for network in profile["networks"] for repeat in range(profile["repeats"])
    }
    actual = [item["name"] for item in results]
    complete = set(actual) == expected and len(actual) == len(expected)
    return {
        "passed": complete and all(item["passed"] for item in results),
        "complete": complete,
        "profile": dict(profile),
        "browser_version": browser["browser_version"],
        "results": results,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    """Render per-run values; keep failed runs and missing measurements visible."""
    lines = [
        "# Realtime delivery replay",
        "",
        f"Model: `{report['profile']['model_id']}`",
        "",
        "| Run | Pass | E2E p50 ms | E2E p90 ms | Missing controls | Stall % | CPU % (one core) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(value: Any) -> str:
        return "—" if value is None else f"{value:.1f}"

    for item in report["results"]:
        value = item["delivery"]
        dist = value["control_to_display_ms"] or {}
        fraction = value["presentation_stall_fraction"]
        lines.append(
            f"| {item['name']} | {item['passed']} | {number(dist.get('p50_ms'))} | "
            f"{number(dist.get('p90_ms'))} | {value['missing_controls']} | "
            f"{number(None if fraction is None else fraction * 100)} | "
            f"{number(item['resources']['cpu_percent_one_core'])} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "STEP commands advance native chunks with fixed seed and initial image. "
            "Presentation gaps include generation and playback, not just network delay. "
            "The report is transport/lifecycle evidence, not an accepted numerical replay or an isolated-GPU speedup. "
            "GPU co-tenancy, source state and environment are recorded in environment.json. "
            "Codec negotiation alone does not certify NVENC execution.",
            "",
        ]
    )
    return "\n".join(lines)


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    return result.stdout if result.returncode == 0 else result.stderr


def sample_process(pid: int, stop: threading.Event, rows: list[dict[str, Any]], gpu: int) -> None:
    """Sample the owned Studio process; GPU memory stays null without NVML."""
    nvml, handle = None, None
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml, handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(gpu)
    except ImportError:
        pass
    try:
        while not stop.is_set():
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                break
            memory = None
            if nvml is not None:
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                memory = next((item.usedGpuMemory for item in processes if item.pid == pid), None)
            rows.append(
                {
                    "wall_ms": time.time() * 1000,
                    "cpu_s": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
                    "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                    "gpu_bytes": memory,
                }
            )
            stop.wait(1)
    finally:
        if nvml is not None:
            nvml.nvmlShutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = json.loads(args.profile.read_text())
    for key in ("model_ref", "image"):
        if not Path(profile[key]).exists():
            parser.error(f"{key} does not exist: {profile[key]}")
    try:
        validate_profile(profile)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.update(profile.get("environment", {}))
    env.update(
        CUDA_VISIBLE_DEVICES=str(profile["gpu"]),
        WORLDFOUNDRY_REALTIME_PERF_JSONL=str(output / "timing.jsonl"),
        WORLDFOUNDRY_REALTIME_PERF_WARMUP_CHUNKS=str(profile["warmup_steps"]),
        WORLDFOUNDRY_STUDIO_WORKSPACE_DIR=str(output / "workspace"),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "worldfoundry.studio.cli",
        profile["model_id"],
        "--model-ref",
        profile["model_ref"],
        "--frontend",
        "world",
        "--device",
        "cuda:0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if profile.get("variant"):
        command.extend(["--variant", profile["variant"]])
    metadata = {
        "command": command,
        "python": sys.version,
        "revision": _command_output(["git", "rev-parse", "HEAD"]).strip(),
        "source_changes": _command_output(["git", "status", "--short", "--untracked-files=no"]),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "aiortc", "av")},
        "gpus": _command_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,utilization.gpu", "--format=csv"]
        ),
        "gpu_processes_before": _command_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"]
        ),
        "environment": {
            key: value
            for key, value in env.items()
            if key.startswith(
                ("WORLDFOUNDRY_REALTIME_", "WORLDFOUNDRY_MATRIX_", "CUDA_", "OMP_", "MKL_", "TORCH_", "PYTORCH_")
            )
            or key in ("LD_LIBRARY_PATH", "CHROMIUM_EXECUTABLE")
        },
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stop = threading.Event()
    resources: list[dict[str, Any]] = []
    with (output / "server.log").open("w") as log:
        load_started = time.monotonic()
        server = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = threading.Thread(
            target=sample_process, args=(server.pid, stop, resources, int(profile["gpu"])), daemon=True
        )
        monitor.start()
        try:
            deadline = time.monotonic() + profile["timeout_ms"] / 1000
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Studio exited with {server.returncode}; see {output / 'server.log'}")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/api/runtime/status", timeout=3) as response:
                        status = json.load(response)
                    if status.get("error"):
                        raise RuntimeError(status["error"])
                    if status.get("ready"):
                        metadata["preload_wall_ms"] = (time.monotonic() - load_started) * 1000
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Studio model preload timed out")
            completed = subprocess.run(
                [
                    "node",
                    str(root / "tests/manual/realtime_browser_replay.mjs"),
                    str(output / "profile.json"),
                    f"http://127.0.0.1:{port}",
                    str(output),
                ],
                cwd=root,
                env=env,
                check=False,
            )
            browser = json.loads((output / "browser.json").read_text())
            report = build_report(profile, browser, output / "timing.jsonl", list(resources))
            report["passed"] = report["passed"] and completed.returncode == 0
        finally:
            shutdown_started = time.monotonic()
            forced_kill = False
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                forced_kill = True
                server.kill()
                server.wait()
            stop.set()
            monitor.join(timeout=5)
            (output / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")
            after = _command_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"])
            metadata.update(
                server_pid=server.pid,
                shutdown_ms=(time.monotonic() - shutdown_started) * 1000,
                forced_kill=forced_kill,
                gpu_processes_after=after,
            )
            (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report["shutdown"] = {
        "forced_kill": forced_kill,
        "elapsed_ms": metadata["shutdown_ms"],
        "gpu_process_released": after.startswith("pid,")
        and not any(line.split(",")[0].strip() == str(server.pid) for line in after.splitlines()),
    }
    report["passed"] = report["passed"] and not forced_kill and report["shutdown"]["gpu_process_released"]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(markdown_report(report))
    print(f"Report: {output / 'report.md'} (passed={report['passed']})", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
