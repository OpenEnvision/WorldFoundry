"""Registry metadata stays available without optional metric runtimes."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class MetricDiscoveryTests(unittest.TestCase):
    def run_python(self, code):
        result = subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(code)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_metric_declarations_load_without_optional_dependencies(self):
        self.run_python('''
            import importlib.abc
            import sys
            class BlockOptional(importlib.abc.MetaPathFinder):
                attempts = []
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'numpy', 'PIL', 'torch', 'scipy', 'cv2', 'torchmetrics', 'transformers'}:
                        self.attempts.append(fullname)
                        raise ModuleNotFoundError(fullname)
            blocker = BlockOptional()
            sys.meta_path.insert(0, blocker)
            from worldfoundry.evaluation.tasks.metrics.registry import (
                _DISCOVERABLE_METRIC_PACKAGES, default_metric_registry, load_metric_module_specs,
                list_metric_registry_entries,
            )
            registry = default_metric_registry()
            entries = {entry.id: entry for entry in list_metric_registry_entries()}
            for package in _DISCOVERABLE_METRIC_PACKAGES:
                for spec in load_metric_module_specs(package):
                    assert spec.id in entries, spec.id
                    assert registry.get(spec.id).higher_is_better == spec.higher_is_better, spec.id
            assert registry.get('clip-fid').id == 'fid'
            assert registry.get('fid').higher_is_better is False
            assert not blocker.attempts, blocker.attempts
        ''')

    def test_lazy_compute_alias_survives_repeated_calls(self):
        self.run_python('''
            from unittest.mock import patch
            from worldfoundry.evaluation.tasks.metrics import fid
            with patch('worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity.calculate_metrics',
                       return_value=lambda **kwargs: {'frechet_inception_distance': 0.25}):
                for call in (lambda: fid.compute('real', 'generated', cuda=False),
                             lambda: fid.compute('real', 'generated', cuda=False),
                             lambda: fid.compute_clip_fid('real', 'generated', cuda=False)):
                    assert call() == 0.25
        ''')


if __name__ == "__main__":
    unittest.main()
