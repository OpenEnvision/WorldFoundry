"""Attention package probes must not initialize optional extensions."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BACKENDS = Path(__file__).resolve().parents[3] / "worldfoundry/core/attention/backends.py"


class AttentionProbeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        path_patch = patch.object(sys, "path", [str(self.root), *sys.path])
        path_patch.start()
        self.addCleanup(path_patch.stop)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), version=SimpleNamespace(hip=None))
        with patch.dict(sys.modules, {"torch": torch}):
            self.backends = runpy.run_path(str(BACKENDS))

    def package(self, name):
        root = self.root / name
        root.mkdir()
        (root / "__init__.py").write_text(
            "from pathlib import Path\n"
            "Path(__file__).with_name('imported').touch()\n"
            "raise ImportError('extension ABI mismatch')\n"
        )
        (root / "kernel.py").write_text("raise AssertionError('kernel imported during probe')\n")
        return root

    def test_auto_and_torch_work_with_an_unloadable_optional_extension(self):
        package = self.package("flash_attn")
        for preferred in ("auto", "torch"):
            with self.subTest(preferred=preferred):
                self.assertEqual(self.backends["resolve_attention_backend"](preferred), "torch")
        self.assertFalse((package / "imported").exists())

    def test_dotted_probes_check_child_presence_and_hardware_without_imports(self):
        package = self.package("probe_fixture")
        for child, supported, available, usable in (
            ("kernel", True, True, True),
            ("kernel", False, True, False),
            ("missing", True, False, False),
        ):
            with self.subTest(child=child, supported=supported):
                result = self.backends["_package_capability"](
                    name="fixture", package=f"probe_fixture.{child}", usable_if=supported,
                    unavailable_reason="missing package", unusable_reason="unsupported hardware",
                )
                self.assertEqual((result.available, result.usable), (available, usable))
        self.assertFalse((package / "imported").exists())
        self.assertNotIn("probe_fixture", sys.modules)


if __name__ == "__main__":
    unittest.main()
