from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

from worldfoundry.runtime.platforms import (
    AcceleratorDescriptor,
    CapabilitySet,
    CudaPlatformProvider,
    MemoryInfo,
    PlatformKind,
    RocmPlatformProvider,
    detect_accelerators,
)


@dataclass
class FakeProvider:
    kind: PlatformKind
    devices: list[AcceleratorDescriptor]

    def detect(self) -> list[AcceleratorDescriptor]:
        return list(self.devices)


def _device(kind: PlatformKind, index: int = 0) -> AcceleratorDescriptor:
    return AcceleratorDescriptor(
        id=f"{kind.value}:{index}",
        platform=kind,
        vendor="test-vendor",
        name="Test Accelerator",
        arch="test-arch",
        index=index,
        memory=MemoryInfo(total_bytes=1024, free_bytes=768),
        capabilities=CapabilitySet(
            dtypes=("float32", "bfloat16"),
            supports_compile=True,
            features=frozenset(("test-feature",)),
        ),
        metadata={"driver": "test"},
    )


def test_platform_package_import_does_not_import_torch() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('platform package imported torch eagerly')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import worldfoundry.runtime.platforms
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_accelerator_descriptor_is_json_serializable() -> None:
    descriptor = _device(PlatformKind.CUDA)

    payload = descriptor.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["platform"] == "cuda"
    assert payload["memory"]["total_bytes"] == 1024
    assert payload["capabilities"]["features"] == ["test-feature"]


def test_preferred_platform_is_detected_and_returned_first() -> None:
    cuda = FakeProvider(PlatformKind.CUDA, [_device(PlatformKind.CUDA)])
    xpu = FakeProvider(PlatformKind.XPU, [_device(PlatformKind.XPU)])

    devices = detect_accelerators(preferred="xpu", providers=[cuda, xpu])

    assert [device.platform for device in devices] == [
        PlatformKind.XPU,
        PlatformKind.CUDA,
    ]


def test_cpu_is_fallback_when_no_accelerator_is_detected() -> None:
    devices = detect_accelerators(
        providers=[FakeProvider(PlatformKind.CUDA, [])]
    )

    assert len(devices) == 1
    assert devices[0].platform is PlatformKind.CPU
    assert devices[0].id == "cpu:0"


def test_supplied_cpu_provider_is_used_only_as_fallback() -> None:
    fake_cpu = _device(PlatformKind.CPU)
    devices = detect_accelerators(
        providers=[
            FakeProvider(PlatformKind.CPU, [fake_cpu]),
            FakeProvider(PlatformKind.ROCM, []),
        ]
    )

    assert devices == [fake_cpu]


def test_rocm_is_distinguished_from_cuda_by_torch_hip(monkeypatch) -> None:
    class FakeCuda:
        CUDAGraph = object

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int):
            assert index == 0
            return SimpleNamespace(
                name="Fake AMD GPU",
                gcnArchName="gfx942:sramecc+:xnack-",
                total_memory=2048,
            )

        @staticmethod
        def is_bf16_supported(**_kwargs) -> bool:
            return True

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="6.3", cuda=None),
        cuda=FakeCuda(),
        compile=lambda function: function,
        distributed=None,
    )
    monkeypatch.setattr(
        "worldfoundry.runtime.platforms.providers._load_torch",
        lambda: fake_torch,
    )

    assert CudaPlatformProvider().detect() == []
    devices = RocmPlatformProvider().detect()

    assert len(devices) == 1
    assert devices[0].platform is PlatformKind.ROCM
    assert devices[0].vendor == "amd"
    assert devices[0].arch == "gfx942"
