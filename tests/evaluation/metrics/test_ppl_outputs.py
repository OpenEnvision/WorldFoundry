"""Compare PPL facade outputs with the in-tree computation using a tiny generator."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from worldfoundry.evaluation.tasks.metrics import summarize_distribution_metrics
from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics
from worldfoundry.evaluation.tasks.metrics._shared.vendor.torch_fidelity import metric_ppl
from worldfoundry.evaluation.tasks.metrics._shared.vendor.torch_fidelity.generative_model_base import (
    GenerativeModelBase,
)
from worldfoundry.evaluation.tasks.metrics._shared.vendor.torch_fidelity.generative_model_modulewrapper import (
    GenerativeModelModuleWrapper,
)
from worldfoundry.evaluation.tasks.metrics.ppl import compute_ppl


class ToyGenerator(GenerativeModelBase):
    z_size = 2
    z_type = "normal"
    num_classes = 0

    def forward(self, z):
        return z[:, :1, None, None].expand(-1, 3, 8, 8)


class SquaredDistance(torch.nn.Module):
    def forward(self, a, b):
        return (a - b).square().flatten(1).mean(1)


class PplOutputTests(unittest.TestCase):
    def test_facade_preserves_reduced_and_per_sample_outputs(self):
        backend = calculate_metrics()
        for reduction in ("mean", "none"):
            with self.subTest(reduction=reduction):
                options = dict(
                    cuda=False,
                    batch_size=2,
                    ppl_reduction=reduction,
                    ppl_discard_percentile_lower=None,
                    ppl_discard_percentile_higher=None,
                    ppl_epsilon=0.01,
                    verbose=False,
                )
                with patch.object(metric_ppl, "create_sample_similarity", return_value=SquaredDistance()):
                    native = backend(input1=ToyGenerator(), input1_model_num_samples=5, ppl=True, **options)
                    wrapped = compute_ppl(ToyGenerator(), num_samples=5, **options)
                self.assertEqual(wrapped.keys(), native.keys())
                for key in native:
                    np.testing.assert_array_equal(wrapped[key], native[key])
                summary = summarize_distribution_metrics(wrapped)
                self.assertEqual(summary["ppl_mean"], native["perceptual_path_length_mean"])
                self.assertEqual(summary["ppl_std"], native["perceptual_path_length_std"])
                if reduction == "none":
                    self.assertEqual(wrapped["perceptual_path_length_raw"].shape, (5,))
                    np.testing.assert_array_equal(summary["ppl_raw"], native["perceptual_path_length_raw"])

    def test_default_and_one_sided_percentile_filters_keep_expected_samples(self):
        options = dict(
            cuda=False, num_samples=101, batch_size=16, ppl_epsilon=0.01, ppl_reduction="none", verbose=False
        )
        with patch.object(metric_ppl, "create_sample_similarity", return_value=SquaredDistance()):
            raw = compute_ppl(
                ToyGenerator(), ppl_discard_percentile_lower=None, ppl_discard_percentile_higher=None, **options
            )["perceptual_path_length_raw"]
            ordered = np.sort(raw)
            for lower, upper, filters in (
                (1, 99, {}),
                (None, 75, {"ppl_discard_percentile_lower": None, "ppl_discard_percentile_higher": 75}),
                (25, None, {"ppl_discard_percentile_lower": 25, "ppl_discard_percentile_higher": None}),
                (25, 75, {"ppl_discard_percentile_lower": 25, "ppl_discard_percentile_higher": 75}),
            ):
                with self.subTest(lower=lower, upper=upper):
                    lo = -np.inf if lower is None else ordered[int(np.floor((len(raw) - 1) * lower / 100))]
                    hi = np.inf if upper is None else ordered[int(np.ceil((len(raw) - 1) * upper / 100))]
                    expected = raw[(raw >= lo) & (raw <= hi)]
                    result = compute_ppl(ToyGenerator(), **options, **filters)
                    self.assertGreater(len(expected), 0)
                    np.testing.assert_array_equal(result["perceptual_path_length_raw"], expected)
                    self.assertEqual(result["perceptual_path_length_mean"], float(expected.mean()))
                    self.assertEqual(result["perceptual_path_length_std"], float(expected.std()))

    def test_conditional_generator_uses_its_declared_classes_and_paired_labels(self):
        class ConditionalGenerator(ToyGenerator):
            num_classes = 2

            def __init__(self):
                super().__init__()
                self.label_batches = []

            def forward(self, z, labels):
                self.label_batches.append(labels.clone())
                return super().forward(z) + labels[:, None, None, None]

        with patch.object(metric_ppl, "create_sample_similarity", return_value=SquaredDistance()):
            for wrapped in (False, True):
                with self.subTest(wrapped=wrapped):
                    generator = ConditionalGenerator()
                    model = GenerativeModelModuleWrapper(generator, 2, "normal", 2) if wrapped else generator
                    result = compute_ppl(
                        model,
                        cuda=False,
                        num_samples=5,
                        batch_size=2,
                        ppl_epsilon=0.01,
                        ppl_reduction="none",
                        ppl_discard_percentile_lower=None,
                        ppl_discard_percentile_higher=None,
                        verbose=False,
                    )
                    self.assertTrue(np.isfinite(result["perceptual_path_length_mean"]))
                    self.assertEqual(result["perceptual_path_length_raw"].shape, (5,))
                    self.assertEqual([len(labels) for labels in generator.label_batches], [4, 4, 2])
                    for labels in generator.label_batches:
                        first, second = labels.chunk(2)
                        torch.testing.assert_close(first, second)
                        self.assertTrue(torch.all((labels >= 0) & (labels < generator.num_classes)))


if __name__ == "__main__":
    unittest.main()
