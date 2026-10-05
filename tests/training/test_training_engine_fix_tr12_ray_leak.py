"""TR-12/13: RayDevicePool must clean placement groups if ready() fails."""

from __future__ import annotations

import enum
import unittest
from types import SimpleNamespace

if not hasattr(enum, "StrEnum"):
    raise unittest.SkipTest("StrEnum requires Python 3.11+")

from worldfoundry.training.distributed.ray_runtime import RayDevicePool, RayDevicePoolConfig


class _FakePlacementGroup:
    def ready(self) -> str:
        return "ready-ref"


class _FakeRay:
    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False
        self.last_get: tuple[object, float | None] | None = None

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, **_kwargs: object) -> None:
        self.initialized = True

    def get(self, refs: object, timeout: float | None = None) -> None:
        self.last_get = (refs, timeout)
        raise RuntimeError("placement group not ready")

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.initialized = False


class TestRayLeakUnittest(unittest.TestCase):
    def test_setup_failure_removes_placement_groups(self) -> None:
        created: list[_FakePlacementGroup] = []
        removed: list[object] = []
        fake_ray = _FakeRay()

        def _placement_group(_bundles: object, strategy: str = "STRICT_PACK") -> _FakePlacementGroup:
            group = _FakePlacementGroup()
            created.append(group)
            return group

        def _import_module(name: str) -> object:
            if name == "ray":
                return fake_ray
            if name == "ray.util.placement_group":
                return SimpleNamespace(
                    placement_group=_placement_group,
                    remove_placement_group=removed.append,
                )
            raise ImportError(name)

        import worldfoundry.training.distributed.ray_runtime as ray_runtime

        original = ray_runtime.import_module
        ray_runtime.import_module = _import_module  # type: ignore[method-assign]
        try:
            pool = RayDevicePool(
                RayDevicePoolConfig(
                    num_devices=2,
                    devices_per_node=2,
                    accelerator_resource="CPU",
                    placement_ready_timeout_s=12.5,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "placement group not ready"):
                pool.setup()
            self.assertEqual(removed, created)
            self.assertIsNotNone(fake_ray.last_get)
            self.assertEqual(fake_ray.last_get[1], 12.5)
            self.assertIsNone(pool._ray)
            self.assertEqual(pool._placement_groups, ())
            self.assertTrue(fake_ray.shutdown_called)
        finally:
            ray_runtime.import_module = original  # type: ignore[method-assign]


if __name__ == "__main__":
    unittest.main()
