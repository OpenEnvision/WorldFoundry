"""Import-light CUDA device discovery and worker grouping helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_DISABLED_DEVICE_VALUES = frozenset({"", "-1", "none", "void"})
_ALL_DEVICE_VALUES = frozenset({"all"})


@dataclass
class CudaDeviceLease:
    """A reversible reservation of one or more physical CUDA device tokens."""

    _pool: "CudaDeviceLeasePool" = field(repr=False)
    tokens: tuple[str, ...]
    _released: bool = field(default=False, init=False, repr=False)
    _process_group_id: int | None = field(default=None, init=False, repr=False)

    @property
    def process_group_id(self) -> int | None:
        return self._process_group_id

    @process_group_id.setter
    def process_group_id(self, pgid: int) -> None:
        self._process_group_id = pgid
        self._pool.record_process_group(self.tokens, pgid)

    @property
    def file_descriptors(self) -> tuple[int, ...]:
        """Lease locks inherited by a worker so a controller crash cannot unlock it."""
        return self._pool.file_descriptors(self.tokens)

    @property
    def visible_devices(self) -> str:
        """Return the value suitable for ``CUDA_VISIBLE_DEVICES``."""

        return ",".join(self.tokens)

    @property
    def allocation_waiting(self) -> bool:
        """Return whether another worker is waiting for devices from this pool."""

        return self._pool.waiting_count > 0

    def release(self) -> None:
        """Return the reserved devices to the pool; repeated calls are harmless."""

        if self._released:
            return
        if self.process_group_id is not None and os.name == "posix":
            from worldfoundry.core.execution.process import process_group_alive

            if process_group_alive(self.process_group_id):
                raise RuntimeError("cannot release CUDA devices while their process group is alive")
        self._released = True
        self._pool.release(self.tokens)

    def __enter__(self) -> "CudaDeviceLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class CudaDeviceLeasePool:
    """Thread-safe allocator for non-overlapping CUDA worker assignments."""

    def __init__(
        self, devices: Sequence[str], *, lock_dir: Path | None = None, lock_keys: Mapping[str, str] | None = None
    ) -> None:
        normalized = normalize_cuda_device_groups(tuple(str(device) for device in devices))
        self._devices = tuple(normalized)
        self._available = list(self._devices)
        self._leased: set[str] = set()
        self._waiting = 0
        self._condition = threading.Condition()
        self._lock_dir = lock_dir
        self._lock_keys = dict(lock_keys or {})
        physical_keys = [
            self._lock_keys.get(item, item) for device in self._devices for item in cuda_device_tokens(device)
        ]
        if len(set(physical_keys)) != len(physical_keys):
            raise ValueError("CUDA device aliases must not refer to the same physical GPU")
        self._file_locks: dict[str, tuple[int, ...]] = {}
        if lock_dir is not None:
            lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    @property
    def available_count(self) -> int:
        with self._condition:
            return len(self._available)

    @property
    def waiting_count(self) -> int:
        with self._condition:
            return self._waiting

    def acquire(
        self,
        count: int = 1,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        poll_interval: float = 0.1,
        deadline: float | None = None,
        tokens: Sequence[str] | None = None,
    ) -> CudaDeviceLease:
        """Wait for and reserve ``count`` devices, honoring cooperative cancellation."""

        pinned = tuple(tokens) if tokens is not None else None
        if pinned is not None and (
            not pinned or len(set(pinned)) != len(pinned) or any(token not in self._devices for token in pinned)
        ):
            raise ValueError("requested CUDA device tokens are outside this pool or duplicated")
        requested = len(pinned) if pinned is not None else max(int(count), 1)
        if requested > len(self._devices):
            raise RuntimeError(f"requested {requested} CUDA devices, but only {len(self._devices)} are available")
        with self._condition:
            self._waiting += 1
            try:
                while True:
                    if cancel_requested is not None and cancel_requested():
                        raise RuntimeError("CUDA device allocation cancelled")
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("timed out waiting for CUDA devices")
                    choices = (pinned,) if pinned is not None else itertools.combinations(self._available, requested)
                    candidates = next(
                        (
                            choice
                            for choice in choices
                            if all(t in self._available for t in choice) and self._try_file_locks(choice)
                        ),
                        None,
                    )
                    if candidates is not None:
                        self._available = [t for t in self._available if t not in candidates]
                        self._leased.update(candidates)
                        break
                    wait = max(float(poll_interval), 0.01)
                    if deadline is not None:
                        wait = min(wait, max(deadline - time.monotonic(), 0.0))
                    self._condition.wait(timeout=wait)
            finally:
                self._waiting -= 1
        return CudaDeviceLease(self, candidates)

    def _try_file_locks(self, tokens: Sequence[str]) -> bool:
        if self._lock_dir is None or os.name != "posix":
            return True
        import fcntl

        acquired: dict[str, tuple[int, ...]] = {}
        try:
            for token in tokens:
                descriptors: list[int] = []
                acquired[token] = ()
                for item in cuda_device_tokens(token):
                    key = self._lock_keys.get(item, item)
                    name = hashlib.sha256(key.encode()).hexdigest() + ".lock"
                    fd = os.open(self._lock_dir / name, os.O_CREAT | os.O_RDWR, 0o600)
                    descriptors.append(fd)
                    acquired[token] = tuple(descriptors)
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if self._previous_group_alive(fd):
                        raise BlockingIOError("GPU's previous worker descendants are still alive")
        except BaseException as exc:
            for descriptors in acquired.values():
                for fd in descriptors:
                    os.close(fd)
            if isinstance(exc, BlockingIOError):
                return False
            raise
        self._file_locks.update(acquired)
        return True

    @staticmethod
    def _previous_group_alive(fd: int) -> bool:
        """Conservatively retain admission if both owners died but descendants survived."""
        from worldfoundry.core.execution.process import process_group_alive, process_identity

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            owner = json.loads(os.read(fd, 4096))
            pgid = int(owner["pgid"])
            identity = owner["identity"]
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except (OSError, ValueError, TypeError, KeyError):
            return False
        if not isinstance(identity, dict) or identity.get("boot_id") != boot or pgid <= 1:
            return False
        current = process_identity(pgid)
        # A different leader incarnation proves that the original group has
        # gone. With no leader, surviving members must still block allocation.
        return (current is None or current == identity) and process_group_alive(pgid)

    def record_process_group(self, tokens: Sequence[str], pgid: int) -> None:
        from worldfoundry.core.execution.process import process_identity

        owner = json.dumps({"pgid": pgid, "identity": process_identity(pgid)}).encode()
        with self._condition:
            for token in tokens:
                for fd in self._file_locks.get(token, ()):
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.ftruncate(fd, 0)
                    os.write(fd, owner)

    def file_descriptors(self, tokens: Sequence[str]) -> tuple[int, ...]:
        with self._condition:
            return tuple(fd for token in tokens for fd in self._file_locks.get(token, ()))

    def release(self, tokens: Sequence[str]) -> None:
        """Release previously leased tokens and wake waiting allocators."""

        with self._condition:
            released = {str(token) for token in tokens if str(token) in self._leased}
            if not released:
                return
            self._leased.difference_update(released)
            for token in released:
                for fd in self._file_locks.pop(token, ()):
                    # Closing, rather than LOCK_UN, retains a worker's inherited
                    # lock until it too exits and closes its descriptor.
                    os.close(fd)
            self._available = [device for device in self._devices if device not in self._leased]
            self._condition.notify_all()


def cuda_device_tokens(value: object) -> tuple[str, ...]:
    """Normalize a comma-separated CUDA visibility value into device tokens."""
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    tokens = tuple(token.strip() for item in raw_items for token in item.split(",") if token.strip())
    if len(tokens) == 1 and tokens[0].lower() in _DISABLED_DEVICE_VALUES:
        return ()
    return tokens


def normalize_cuda_device_groups(values: Sequence[str] | None) -> tuple[str, ...]:
    """Validate non-overlapping worker groups used as CUDA_VISIBLE_DEVICES values."""
    groups: list[str] = []
    assigned: set[str] = set()
    raw_groups: Sequence[str] = (values,) if isinstance(values, str) else (values or ())
    for raw_group in raw_groups:
        tokens = cuda_device_tokens(raw_group)
        if not tokens:
            raise ValueError("CUDA worker device groups cannot be empty")
        lowered = {token.lower() for token in tokens}
        if lowered & _ALL_DEVICE_VALUES:
            raise ValueError("CUDA worker device groups must list concrete device ids or UUIDs, not 'all'")
        if lowered & _DISABLED_DEVICE_VALUES:
            raise ValueError("CUDA worker device groups must list enabled device ids or UUIDs")
        if len(lowered) != len(tokens):
            raise ValueError(f"CUDA worker device group contains duplicates: {raw_group!r}")
        overlap = assigned.intersection(lowered)
        if overlap:
            duplicate = ", ".join(sorted(overlap))
            raise ValueError(f"CUDA devices cannot be assigned to multiple model workers: {duplicate}")
        assigned.update(lowered)
        groups.append(",".join(tokens))
    return tuple(groups)


def discover_cuda_device_tokens(
    environ: Mapping[str, str] | None = None,
    *,
    timeout_seconds: float = 3.0,
) -> tuple[str, ...]:
    """Discover visible NVIDIA device ids without importing or initializing Torch."""
    env = os.environ if environ is None else environ
    if "CUDA_VISIBLE_DEVICES" in env:
        configured = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip()
        if configured.lower() not in _ALL_DEVICE_VALUES:
            return cuda_device_tokens(configured)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=max(float(timeout_seconds), 0.1),
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            completed = None
        if completed is not None and completed.returncode == 0:
            devices = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
            if devices:
                # NVIDIA_VISIBLE_DEVICES is interpreted by the container runtime and may
                # contain host indices that are remapped to different local CUDA indices.
                return devices

    container_devices = str(env.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
    if container_devices and container_devices.lower() not in _ALL_DEVICE_VALUES:
        return cuda_device_tokens(container_devices)
    return ()


def cuda_device_discovery_source(environ: Mapping[str, str] | None = None) -> str:
    """Describe whether concrete CUDA visibility comes from the environment."""
    env = os.environ if environ is None else environ
    configured = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if configured and configured.lower() not in _ALL_DEVICE_VALUES:
        return "environment"
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    container_devices = str(env.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
    if container_devices and container_devices.lower() not in _ALL_DEVICE_VALUES:
        return "environment"
    return "unavailable"


def default_cuda_device_groups(
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return one non-overlapping model-worker group per visible CUDA device."""
    return normalize_cuda_device_groups(discover_cuda_device_tokens(environ))


__all__ = [
    "CudaDeviceLease",
    "CudaDeviceLeasePool",
    "cuda_device_discovery_source",
    "cuda_device_tokens",
    "default_cuda_device_groups",
    "discover_cuda_device_tokens",
    "normalize_cuda_device_groups",
]
