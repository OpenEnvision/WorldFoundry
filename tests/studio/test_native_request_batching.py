# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Studio preparation, artifact mapping and resident batching transport contracts."""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from worldfoundry.studio.inference import dispatch, execution
from worldfoundry.studio.inference.batching import native_batch_key
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.resident_batching import ResidentRequest, run_queued_resident_request

pytestmark = pytest.mark.unit
_CONTROL: Path | None = None


def call(prompt: str = "boat", seed: int = 11) -> dict[str, Any]:
    return dict(
        model_id="wan2.1-t2v-1.3b",
        action="run",
        prompt=prompt,
        input_path="",
        image=None,
        video=None,
        last_frame=None,
        reference_files=None,
        interactions_text="",
        camera_view_text="",
        task_type="",
        intrinsics_text="",
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=8,
        num_frames=1,
        call_kwargs_text=json.dumps({"height": 8, "width": 8, "num_inference_steps": 2, "seed": seed}),
        load_kwargs_text=json.dumps(
            {
                "offload_mode": "resident",
                "attention_backend": "torch",
                "rope_precision": "fp64",
                "rms_norm_precision": "fp32",
            }
        ),
        model_ref="unused-test-checkpoint",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )


def entry() -> CatalogEntry:
    return CatalogEntry(
        model_id="wan2.1-t2v-1.3b",
        display_name="Wan",
        module_path="unused",
        class_name="Pipeline",
        family="Wan",
        category="Video",
        summary="",
        supports_from_pretrained=True,
    )


class Pipeline:
    loads = 0
    batches: list[Sequence[Mapping[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> Pipeline:
        cls.loads += 1
        return cls()

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if _CONTROL is not None and kwargs["prompt"] == "single":
            (_CONTROL / "selected.json").write_text(json.dumps(["single"]))
            while (_CONTROL / "hold").exists():
                time.sleep(0.01)
        path = Path(kwargs["output_dir"]) / "generated.png"
        Image.new("RGB", (8, 8), (kwargs["seed"], 0, 0)).save(path)
        return {"generated_image_path": str(path), "metadata": {"seed": kwargs["seed"], "prompt": kwargs["prompt"]}}

    def batch_call(self, calls: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        self.batches.append(calls)
        if _CONTROL is not None:
            (_CONTROL / "selected.json").write_text(json.dumps([item["prompt"] for item in calls]))
            while (_CONTROL / "hold").exists():
                time.sleep(0.01)
        if any(item["prompt"] == "fail" for item in calls):
            raise RuntimeError("native batch failed")
        return tuple(self(**item) for item in calls)


def worker_main(workspace: str, control: str) -> None:
    """Run the real JSON worker with a checkpoint-free artifact producer."""
    from argparse import Namespace

    from worldfoundry.studio.runtime_job import _run_manager_worker

    global _CONTROL
    _CONTROL = Path(control)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, "find_entry", lambda _model_id: entry())
        patch.setattr(execution.StudioManager, "import_pipeline_class", lambda _self, _entry: Pipeline)
        _run_manager_worker(Namespace(workspace_root=workspace, max_cached_pipelines=1))


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[execution.StudioManager]:
    Pipeline.loads = 0
    Pipeline.batches = []
    monkeypatch.setattr(execution, "find_entry", lambda _model_id: entry())
    manager = execution.StudioManager(workspace_root=str(tmp_path))
    monkeypatch.setattr(manager, "import_pipeline_class", lambda _entry: Pipeline)
    try:
        yield manager
    finally:
        manager.close()


@pytest.mark.parametrize("model_id", ["wan2.1-t2v-1.3b", "wan2.2-t2v-a14b", "wan2.2-ti2v-5b"])
def test_workspace_payloads_use_compatible_catalog_ids(model_id: str, tmp_path: Path) -> None:
    from worldfoundry.studio.inference.requests import JobCreateRequest, build_inference_run_kwargs

    keys = []
    for prompt, seed in (("boat", 11), ("dog", 29)):
        image_path = ""
        if model_id == "wan2.2-ti2v-5b":
            image_path = str(tmp_path / f"{prompt}.png")
            Image.new("RGB", (8, 8), (seed, 0, 0)).save(image_path)
        _, kwargs = build_inference_run_kwargs(
            JobCreateRequest(
                model_id=model_id,
                prompt=prompt,
                input_path=image_path,
                params={"seed": seed, "height": 256, "width": 256, "num_frames": 17, "num_inference_steps": 8},
                load_kwargs={
                    "offload_mode": "resident",
                    "attention_backend": "torch",
                    "rope_precision": "fp64",
                    "rms_norm_precision": "fp32",
                },
            )
        )
        keys.append(native_batch_key(kwargs))
    assert keys[0] is not None and keys[0] == keys[1]


def test_manager_batch_preserves_records_artifacts_and_cached_owner(manager: execution.StudioManager) -> None:
    records = manager.run_batch([call(), call("dog", 29)])
    assert Pipeline.loads == 1 and len(Pipeline.batches) == 1
    assert len({record.run_id for record in records}) == 2
    assert len({record.output_dir for record in records}) == 2
    for index, (record, seed) in enumerate(zip(records, (11, 29))):
        assert record.status == "succeeded"
        assert record.metadata["result"]["metadata"]["seed"] == seed
        assert record.metadata["studio_request_batching"]["request_index"] == index
        assert record.metadata["studio_performance"]["batch_execute_ms"] >= 0
        assert Path(record.manifest_path).is_file()
        assert Image.open(Path(record.output_dir) / "generated.png").getpixel((0, 0)) == (seed, 0, 0)
    contexts = tuple(manager.pipeline_cache.values())
    assert len(contexts) == 1 and contexts[0].active_leases == 0
    assert manager.run_batch([call("dog", 29), call()])[1].metadata["result"]["metadata"]["seed"] == 11
    assert Pipeline.loads == 1


def test_manager_image_batch_keeps_independent_condition_paths(
    manager: execution.StudioManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "wan2.2-ti2v-5b"
    monkeypatch.setattr(execution, "find_entry", lambda _model_id: replace(entry(), model_id=model_id))
    calls = []
    for prompt, seed in (("boat", 11), ("dog", 29)):
        path = tmp_path / f"{prompt}.png"
        Image.new("RGB", (8, 8), (seed, 0, 0)).save(path)
        calls.append({**call(prompt, seed), "model_id": model_id, "input_path": str(path)})
    records = manager.run_batch(calls)
    assert Pipeline.loads == 1 and len(Pipeline.batches) == 1
    images = [item["images"] for item in Pipeline.batches[0]]
    assert len(set(images)) == 2
    for image, record, seed in zip(images, records, (11, 29)):
        assert Path(image).is_relative_to(record.output_dir)
        with Image.open(image) as pixels:
            assert pixels.getpixel((0, 0)) == (seed, 0, 0)


def test_image_warmup_paths_keep_resident_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PWD", str(tmp_path))
    kwargs = {**call(), "model_id": "wan2.2-ti2v-5b"}
    options = json.loads(kwargs["load_kwargs_text"])
    options["warmup_requests"] = [{"inputs": {"images": str(tmp_path / "frame.png")}}]
    kwargs["load_kwargs_text"] = json.dumps(options)
    payload, secrets = dispatch._payload_run_kwargs_with_secret_refs(kwargs)
    assert not secrets
    assert payload["load_kwargs_text"] == kwargs["load_kwargs_text"]
    assert native_batch_key(kwargs) is not None


@pytest.mark.parametrize("omitted", ["rope_precision", "rms_norm_precision"])
def test_image_batch_requires_explicit_precision_before_loading(
    omitted: str,
    manager: execution.StudioManager,
) -> None:
    kwargs = {**call(), "model_id": "wan2.2-ti2v-5b"}
    options = json.loads(kwargs["load_kwargs_text"])
    options.pop(omitted)
    kwargs["load_kwargs_text"] = json.dumps(options)
    assert native_batch_key(kwargs) is None
    with pytest.raises(ValueError, match="compatible resident"):
        manager.run_batch([kwargs, kwargs])
    assert Pipeline.loads == 0


def test_image_admission_separates_image_and_text_modes() -> None:
    base = {**call(), "model_id": "wan2.2-ti2v-5b"}
    first, second = {**base, "input_path": "a.png"}, {**base, "input_path": "b.jpg"}
    assert native_batch_key(first) is not None and native_batch_key(first) == native_batch_key(second)
    assert native_batch_key(first) != native_batch_key(base)
    assert native_batch_key({**base, "input_path": "a.mp4"}) is None
    assert native_batch_key({**base, "call_kwargs_text": '{"images":"a.png"}'}) is None


def test_manager_batch_failure_releases_lease_and_can_recover(manager: execution.StudioManager) -> None:
    with pytest.raises(RuntimeError, match="native batch failed"):
        manager.run_batch([call("fail"), call("dog", 29)])
    assert tuple(manager.pipeline_cache.values())[0].active_leases == 0
    assert manager.run_batch([call(), call("dog", 29)])[0].status == "succeeded"
    assert len(Pipeline.batches) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"action": "stream"},
        {"image": "image.png"},
        {"model_id": "wan2.2-i2v-a14b"},
        {"load_kwargs_text": '{"offload_mode":"block","attention_backend":"torch"}'},
        {"load_kwargs_text": '{"offload_mode":"resident","attention_backend":"torch","rope_precision":"fp32"}'},
        {"load_kwargs_text": '{"offload_mode":"resident","attention_backend":"torch","rms_norm_precision":"input"}'},
        {"load_kwargs_text": '{"offload_mode":"resident","attention_backend":"torch","fuse_qkv":true}'},
        {"load_kwargs_text": '{"offload_mode":"resident","attention_backend":"torch","fused_rope":true}'},
        {"load_kwargs_text": '{"offload_mode":"resident","attention_backend":"torch","sequence_parallel":2}'},
        {"call_kwargs_text": '{"prompt":["a","b"]}'},
        {"backend": "api_init"},
    ],
)
def test_unsupported_modes_are_singletons(change: dict[str, Any]) -> None:
    assert native_batch_key({**call(), **change}) is None


def test_admission_ignores_only_request_local_values_and_rejects_mixed_policy(manager: execution.StudioManager) -> None:
    assert native_batch_key(call()) == native_batch_key(call("dog", 29))
    other = {**call(), "num_frames": 17}
    assert native_batch_key(other) != native_batch_key(call())
    with pytest.raises(ValueError, match="compatible resident"):
        manager.run_batch([call(), other])
    assert Pipeline.loads == 0


class Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.control = root / "control"
        self.control.mkdir()
        self.control.joinpath("hold").touch()
        command = [
            sys.executable,
            "-u",
            "-c",
            (
                f"import runpy; ns=runpy.run_path({str(Path(__file__).resolve())!r}); "
                f"ns['worker_main']({str(root / 'workspace')!r}, {str(self.control)!r})"
            ),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=0,
        )
        assert process.stdin is not None and process.stdout is not None
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        self.worker = dispatch._ResidentWorker(
            key=("batch-test",),
            base_key=("batch-test",),
            model_id="wan2.1-t2v-1.3b",
            process=process,
            lock=threading.RLock(),
            decoder=codecs.getincrementaldecoder("utf-8")("replace"),
            command=command,
            created_at=time.monotonic(),
            last_used_at=time.monotonic(),
        )
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.max_delay_seconds = 1.0
        self.requests: list[ResidentRequest] = []

    def submit(self, prompt: str, seed: int, *, deadline: float | None = None) -> Any:
        directory = self.root / str(len(self.requests))
        directory.mkdir()
        request = ResidentRequest(
            payload={
                "request_id": str(len(self.requests)),
                "run_kwargs": call(prompt, seed),
                "result_path": str(directory / "result.json"),
                "error_path": str(directory / "error.json"),
            },
            result_path=directory / "result.json",
            error_path=directory / "error.json",
            deadline=deadline,
            log_callback=None,
            cancel_requested=None,
        )
        self.requests.append(request)
        with dispatch._RESIDENT_WORKERS_LOCK:
            self.worker.in_use += 1
        return self.pool.submit(
            run_queued_resident_request,
            self.worker,
            request,
            max_batch_size=2,
            max_delay_seconds=self.max_delay_seconds,
        )

    def selected(self) -> list[str]:
        deadline = time.monotonic() + 60
        path = self.control / "selected.json"
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert path.exists(), "resident worker did not select a batch"
        return json.loads(path.read_text())

    def release(self) -> None:
        self.control.joinpath("hold").unlink(missing_ok=True)

    def close(self) -> None:
        self.release()
        dispatch._shutdown_resident_worker(self.worker, force=True)
        self.pool.shutdown(wait=True)
        if self.worker.batch_executor is not None:
            self.worker.batch_executor.close()


@pytest.fixture
def resident(tmp_path: Path) -> Iterator[Harness]:
    harness = Harness(tmp_path)
    try:
        yield harness
    finally:
        harness.close()


@pytest.mark.parametrize("fail_send", [False, True])
def test_public_resident_dispatch_uses_queue_without_failure_retries(
    resident: Harness,
    monkeypatch: pytest.MonkeyPatch,
    fail_send: bool,
) -> None:
    from worldfoundry.runtime.conda import RuntimeCondaEnvSpec

    worker = resident.worker
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_BATCH_MAX_SIZE", "2")
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_BATCH_DELAY_MS", "1000")
    monkeypatch.setenv(dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "0")
    monkeypatch.setattr(dispatch, "_RESIDENT_WORKERS", {worker.key: worker})
    monkeypatch.setattr(dispatch, "_ensure_resident_worker_reaper", lambda: None)
    monkeypatch.setattr(
        dispatch,
        "_prepare_resident_run_context",
        lambda **kwargs: dispatch._ResidentRunContext(
            child_run_kwargs=kwargs["run_kwargs"],
            payload_run_kwargs=kwargs["run_kwargs"],
            env={},
            key=worker.key,
        ),
    )
    monkeypatch.setattr(dispatch, "_run_manager_payload_one_shot", lambda **_kwargs: pytest.fail("must not retry"))
    if fail_send:

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise dispatch._ResidentWorkerUnavailable("write failed")

        monkeypatch.setattr(dispatch, "_send_resident_worker_request", fail)

    def run(prompt: str, seed: int) -> execution.RunRecord:
        return dispatch._run_manager_payload_in_conda_impl(
            model_id=worker.model_id,
            spec=RuntimeCondaEnvSpec(model_id=worker.model_id, env_name="test", env_root=resident.root),
            workspace_root=str(resident.root / "workspace"),
            run_kwargs=call(prompt, seed),
            dispatch_root=resident.root / f"public-{seed}",
            request_timeout=0,
        )

    a, b = resident.pool.submit(run, "boat", 11), resident.pool.submit(run, "dog", 29)
    if fail_send:
        for future in (a, b):
            with pytest.raises(RuntimeError, match="write failed"):
                future.result(20)
    else:
        resident.selected()
        resident.release()
        assert [future.result(20).metadata["result"]["metadata"]["seed"] for future in (a, b)] == [11, 29]
    assert worker.in_use == 0


def test_real_resident_protocol_returns_separate_results(resident: Harness) -> None:
    a, b = resident.submit("boat", 11), resident.submit("dog", 29)
    assert set(resident.selected()) == {"boat", "dog"}
    resident.release()
    records = [a.result(20), b.result(20)]
    assert [record.metadata["result"]["metadata"]["seed"] for record in records] == [11, 29]
    assert records[0].run_id != records[1].run_id
    assert all(record.metadata["studio_performance"]["resident_queue_wait_ms"] >= 0 for record in records)
    assert resident.worker.in_use == 0 and resident.worker.process.poll() is None


@pytest.mark.parametrize("timeout", [False, True])
def test_abandoning_one_active_member_keeps_peer_and_worker_alive(resident: Harness, timeout: bool) -> None:
    a, b = resident.submit("boat", 11), resident.submit("dog", 29)
    resident.selected()
    if timeout:
        resident.requests[0].deadline = time.monotonic() - 1
    else:
        resident.requests[0].cancel_requested = lambda: True
    with pytest.raises((RuntimeError, TimeoutError), match="cancel|timed out"):
        a.result(5)
    assert resident.worker.in_use == 2 and resident.worker.process.poll() is None
    resident.release()
    assert b.result(20).metadata["result"]["metadata"]["seed"] == 29
    assert resident.worker.process.poll() is None
    raw = json.loads(resident.requests[0].result_path.read_text())
    manifest = json.loads(Path(raw["manifest_path"]).read_text())
    assert manifest["status"] == "cancelled" and manifest["metadata"]["resident_result_discarded"]
    assert resident.worker.in_use == 0


def test_queued_cancel_does_not_reach_worker(resident: Harness) -> None:
    a, b = resident.submit("boat", 11), resident.submit("dog", 29)
    resident.selected()
    c = resident.submit("queued", 31)
    resident.requests[2].cancel_requested = lambda: True
    with pytest.raises(RuntimeError, match="cancel"):
        c.result(5)
    resident.release()
    a.result(20)
    b.result(20)
    assert set(resident.selected()) == {"boat", "dog"}
    assert not resident.requests[2].result_path.exists() and resident.worker.in_use == 0


def test_failed_batch_fails_all_without_singleton_retry(resident: Harness) -> None:
    a, b = resident.submit("fail", 11), resident.submit("dog", 29)
    resident.selected()
    resident.release()
    for future in (a, b):
        with pytest.raises(RuntimeError, match="native batch failed"):
            future.result(20)
    assert all(request.error_path.is_file() and not request.result_path.exists() for request in resident.requests)
    assert resident.worker.in_use == 0 and resident.worker.process.poll() is not None


@pytest.mark.parametrize("cancel_all", [False, True])
def test_worker_death_or_whole_batch_cancel_finishes_all_waiters(resident: Harness, cancel_all: bool) -> None:
    a, b = resident.submit("boat", 11), resident.submit("dog", 29)
    resident.selected()
    pending = None if cancel_all else resident.submit("queued", 31)
    if cancel_all:
        for request in resident.requests[:2]:
            request.cancel_requested = lambda: True
    else:
        resident.worker.process.kill()
    for future in (a, b, *(() if pending is None else (pending,))):
        with pytest.raises(RuntimeError):
            future.result(20)
    assert resident.worker.batch_executor is not None
    resident.worker.batch_executor.close()
    assert resident.worker.process.poll() is not None and resident.worker.in_use == 0


def test_active_singleton_deadline_preserves_queued_peer(resident: Harness) -> None:
    resident.max_delay_seconds = 0
    resident.submit("prelude", 7).result(60)
    active = resident.submit("single", 11, deadline=time.monotonic() + 3)
    assert resident.selected() == ["single"]
    pending = resident.submit("queued", 29)
    with pytest.raises(TimeoutError, match="timed out"):
        active.result(5)
    resident.release()
    assert pending.result(20).metadata["result"]["metadata"]["seed"] == 29
    assert resident.worker.process.poll() is None and resident.worker.in_use == 0


def test_whole_active_batch_cancel_preserves_queued_peer(resident: Harness) -> None:
    a, b = resident.submit("boat", 11), resident.submit("dog", 29)
    resident.selected()
    pending = resident.submit("queued", 31)
    for request in resident.requests[:2]:
        request.cancel_requested = lambda: True
    for future in (a, b):
        with pytest.raises(RuntimeError, match="cancel"):
            future.result(5)
    assert resident.worker.process.poll() is None
    resident.release()
    assert pending.result(20).metadata["result"]["metadata"]["seed"] == 31
    assert resident.worker.process.poll() is None and resident.worker.in_use == 0
