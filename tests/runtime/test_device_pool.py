from __future__ import annotations

import threading
import time
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from worldfoundry.core.execution.process import terminate_owned_process_group, terminate_process_group
from worldfoundry.runtime import device_pool
from worldfoundry.runtime.device_pool import CudaDeviceLeasePool


def test_container_local_indices_override_host_nvidia_visible_devices(monkeypatch):
    monkeypatch.setattr(device_pool.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        device_pool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0\n1\n2\n"),
    )

    assert device_pool.discover_cuda_device_tokens(
        {"NVIDIA_VISIBLE_DEVICES": "7,0,1"}
    ) == ("0", "1", "2")


def test_explicit_cuda_visibility_remains_authoritative(monkeypatch):
    monkeypatch.setattr(
        device_pool.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("nvidia-smi should not run")),
    )

    assert device_pool.discover_cuda_device_tokens(
        {"CUDA_VISIBLE_DEVICES": "4,2", "NVIDIA_VISIBLE_DEVICES": "7,0"}
    ) == ("4", "2")


def test_cuda_device_leases_are_non_overlapping_and_reusable():
    pool = CudaDeviceLeasePool(("0", "1"))

    first = pool.acquire()
    second = pool.acquire()
    assert first.tokens == ("0",)
    assert second.tokens == ("1",)
    assert pool.available_count == 0

    first.release()
    replacement = pool.acquire()
    assert replacement.tokens == ("0",)

    second.release()
    replacement.release()
    assert pool.available_count == 2


def test_cuda_device_lease_waiter_wakes_after_release():
    pool = CudaDeviceLeasePool(("0",))
    first = pool.acquire()
    acquired: list[str] = []
    ready = threading.Event()

    def wait_for_device() -> None:
        ready.set()
        with pool.acquire() as lease:
            acquired.append(lease.visible_devices)

    thread = threading.Thread(target=wait_for_device)
    thread.start()
    assert ready.wait(timeout=1)
    deadline = time.monotonic() + 1
    while pool.waiting_count == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.waiting_count == 1
    first.release()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert acquired == ["0"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file locks")
def test_shared_pools_choose_another_gpu_and_respect_pinned_deadlines(tmp_path):
    first_pool = CudaDeviceLeasePool(("0", "1"), lock_dir=tmp_path)
    second_pool = CudaDeviceLeasePool(("0", "1"), lock_dir=tmp_path)
    with first_pool.acquire(tokens=("0",)):
        with second_pool.acquire(deadline=time.monotonic() + 1) as other:
            assert other.tokens == ("1",)
        with pytest.raises(TimeoutError):
            second_pool.acquire(tokens=("0",), deadline=time.monotonic() + 0.1)
    with second_pool.acquire(tokens=("0",), deadline=time.monotonic() + 1):
        pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX file locks")
def test_gpu_uuid_and_ordinal_alias_share_a_lock(tmp_path):
    ordinal = CudaDeviceLeasePool(("0",), lock_dir=tmp_path, lock_keys={"0": "GPU-test"})
    uuid = CudaDeviceLeasePool(("GPU-test",), lock_dir=tmp_path)
    with ordinal.acquire():
        with pytest.raises(TimeoutError):
            uuid.acquire(deadline=time.monotonic() + 0.1)
    with pytest.raises(ValueError, match="aliases"):
        CudaDeviceLeasePool(("0", "GPU-test"), lock_keys={"0": "GPU-test"})


def test_free_pool_still_honors_cancel_and_expired_deadline():
    pool = CudaDeviceLeasePool(("0",))
    with pytest.raises(RuntimeError, match="cancelled"):
        pool.acquire(cancel_requested=lambda: True)
    with pytest.raises(TimeoutError):
        pool.acquire(deadline=time.monotonic() - 1)
    assert pool.available_count == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_a_live_worker_prevents_lease_release(tmp_path):
    pool = CudaDeviceLeasePool(("0",), lock_dir=tmp_path)
    lease = pool.acquire()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                               start_new_session=True, pass_fds=lease.file_descriptors)
    lease.process_group_id = process.pid
    try:
        with pytest.raises(RuntimeError, match="process group is alive"):
            lease.release()
        assert pool.available_count == 0
    finally:
        terminate_process_group(process, grace_seconds=0)
        lease.release()
    assert pool.available_count == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX inherited file locks")
def test_controller_exit_does_not_release_a_workers_gpu_lock(tmp_path):
    # A separate controller exits without release; the worker must retain the
    # inherited flock rather than letting a new Studio allocate its GPU.
    script = (
        "import os,subprocess,sys; from pathlib import Path; "
        "from worldfoundry.runtime.device_pool import CudaDeviceLeasePool; "
        "pool=CudaDeviceLeasePool(('0',),lock_dir=Path(sys.argv[1])); lease=pool.acquire(); "
        "worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],"
        "start_new_session=True,pass_fds=lease.file_descriptors,stdout=subprocess.DEVNULL); "
        "print(worker.pid,flush=True); os._exit(0)"
    )
    controller = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    pid = int(controller.stdout.readline())
    controller.wait(timeout=5)
    controller.stdout.close()
    pool = CudaDeviceLeasePool(("0",), lock_dir=tmp_path)
    try:
        with pytest.raises(TimeoutError):
            pool.acquire(deadline=time.monotonic() + 0.1)
    finally:
        terminate_owned_process_group(pid, grace_seconds=0)
    with pool.acquire(deadline=time.monotonic() + 1):
        pass


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux group ownership metadata")
def test_surviving_descendants_block_gpu_reallocation_after_both_owners_die(tmp_path):
    # Nested vendor launches may close inherited lease FDs. Ownership metadata
    # must still block allocation after the controller and worker leader die.
    parent_script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "time.sleep(120)"
    )
    script = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        "from worldfoundry.runtime.device_pool import CudaDeviceLeasePool; "
        "pool=CudaDeviceLeasePool(('0',),lock_dir=Path(sys.argv[1])); lease=pool.acquire(); "
        f"worker=subprocess.Popen([sys.executable,'-c',{parent_script!r}],"
        "start_new_session=True,pass_fds=lease.file_descriptors,stdout=subprocess.DEVNULL); "
        "lease.process_group_id=worker.pid; time.sleep(0.15); "
        "print(worker.pid,flush=True); os._exit(0)"
    )
    controller = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    pid = int(controller.stdout.readline())
    controller.wait(timeout=5)
    controller.stdout.close()
    pool = CudaDeviceLeasePool(("0",), lock_dir=tmp_path)
    try:
        import signal

        os.kill(pid, signal.SIGKILL)
        with pytest.raises(TimeoutError):
            pool.acquire(deadline=time.monotonic() + 0.15)
    finally:
        terminate_owned_process_group(pid, grace_seconds=0)
    with pool.acquire(deadline=time.monotonic() + 1):
        pass
