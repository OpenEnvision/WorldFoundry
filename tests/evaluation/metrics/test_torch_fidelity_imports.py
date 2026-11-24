"""The bundled backend must coexist with an installed torch-fidelity package."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import unittest


class TorchFidelityImportTests(unittest.TestCase):
    def test_torchmetrics_import_order_keeps_the_bundled_backend(self):
        try:
            importlib.metadata.version("torch-fidelity")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("requires an installed torch-fidelity alongside the bundled backend")
        program = """
import inspect
import sys
from pathlib import Path
from worldfoundry.evals.tasks.metrics._shared.torch_fidelity import calculate_metrics, vendor_root

if sys.argv[1] == 'external-first':
    import torchmetrics
    import torch_fidelity
    external = torch_fidelity

backend = calculate_metrics()
assert Path(inspect.getsourcefile(backend)) == vendor_root() / 'torch_fidelity' / 'metrics.py'
import torchmetrics
import torch_fidelity
assert not Path(torch_fidelity.__file__).is_relative_to(vendor_root())
if sys.argv[1] == 'external-first':
    assert torch_fidelity is external
assert calculate_metrics() is backend
"""
        for order in ("external-first", "bundled-first"):
            with self.subTest(order=order):
                result = subprocess.run(
                    [sys.executable, "-B", "-c", program, order], capture_output=True, text=True, timeout=60
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
