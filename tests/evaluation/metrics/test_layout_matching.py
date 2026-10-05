"""Layout matching and object-count regressions."""

from __future__ import annotations

import unittest
from itertools import permutations

import numpy as np

from worldfoundry.evaluation.tasks.metrics.lqs import compute_lqs
from worldfoundry.evaluation.tasks.metrics.lqs.wrapper import _match_boxes


class LayoutMatchingTests(unittest.TestCase):
    def test_missing_and_extra_objects_contribute_to_recall_and_precision(self):
        first = {"label": "chair", "bbox": [0, 0, 10, 10]}
        second = {"label": "chair", "bbox": [20, 0, 30, 10]}
        missing = compute_lqs([first, second], [second])
        extra = compute_lqs([second], [first, second])
        self.assertEqual((missing["lr"], missing["lp"], missing["lqs"]), (0.5, 1.0, 3.5))
        self.assertEqual((extra["lr"], extra["lp"], extra["lqs"]), (1.0, 0.5, 3.5))
        displaced = {"label": "chair", "bbox": [40, 0, 50, 10]}
        self.assertLess(compute_lqs([first, second], [displaced])["lqs"], missing["lqs"])
        self.assertEqual(compute_lqs([first, second], [])["lqs"], 0)

    def test_assignment_matches_exhaustive_minimum_on_small_layouts(self):
        rng = np.random.default_rng(15)
        for n_gt, n_pred in ((3, 3), (2, 4), (4, 2)):
            with self.subTest(n_gt=n_gt, n_pred=n_pred):
                gt = [{"label": "chair", "bbox": [*xy, 1, 1, 0]} for xy in rng.uniform(0, 20, (n_gt, 2))]
                pred = [{"label": "chair", "bbox": [*xy, 1, 1, 0]} for xy in rng.uniform(0, 20, (n_pred, 2))]

                def distance(a, b):
                    return np.linalg.norm(np.array(a["bbox"][:2]) - np.array(b["bbox"][:2]))

                smaller, larger = (gt, pred) if n_gt <= n_pred else (pred, gt)
                expected = min(
                    sum(distance(a, b) for a, b in zip(smaller, order)) for order in permutations(larger, len(smaller))
                )
                pairs = _match_boxes(gt, pred)
                self.assertEqual(len(pairs), min(n_gt, n_pred))
                self.assertAlmostEqual(sum(distance(a, b) for a, b in pairs), expected)

    def test_many_same_category_objects_match_in_any_order(self):
        layout = [{"label": "chair", "bbox": [i * 2, 0, i * 2 + 1, 1]} for i in range(12)]
        self.assertEqual(compute_lqs(layout, layout[::-1])["lqs"], 4)


if __name__ == "__main__":
    unittest.main()
