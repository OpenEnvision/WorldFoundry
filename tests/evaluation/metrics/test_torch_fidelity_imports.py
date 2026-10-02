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
from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics, vendor_root

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

import torch
from unittest.mock import patch
from worldfoundry.evaluation.tasks.metrics import compute_ppl
from worldfoundry.evaluation.tasks.metrics._shared.vendor.torch_fidelity import metric_ppl

class Generator(torch.nn.Module):
    def forward(self, z, labels):
        return (z[:, :1] + labels[:, None])[:, :, None, None].expand(-1, 3, 8, 8)

def distance(a, b):
    return (a - b).square().flatten(1).mean(1)

model = torch_fidelity.GenerativeModelModuleWrapper(Generator(), 2, 'normal', 2)
with patch.object(metric_ppl, 'create_sample_similarity', return_value=distance):
    score = compute_ppl(model, cuda=False, num_samples=5, batch_size=2, verbose=False)
assert score['perceptual_path_length_mean'] > 0
"""
        for order in ("external-first", "bundled-first"):
            with self.subTest(order=order):
                result = subprocess.run(
                    [sys.executable, "-B", "-c", program, order], capture_output=True, text=True, timeout=60
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
