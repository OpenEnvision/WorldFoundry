"""Crop-source and bbox geometry regressions using real image files."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from worldfoundry.evaluation.tasks.metrics._shared.bbox import bbox_iou
from worldfoundry.evaluation.tasks.metrics.fid.scene import compute_scene_fid, extract_object_crops
from worldfoundry.evaluation.tasks.metrics.lqs import compute_lqs
from worldfoundry.evaluation.tasks.metrics.object_wise_consistency import compute_object_wise_consistency


class SceneInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="scene-inputs-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.reference = self.root / "reference"
        self.generated = self.root / "generated"
        for root, color in ((self.reference, "red"), (self.generated, "blue")):
            (root / "nested").mkdir(parents=True)
            Image.new("RGB", (64, 64), color).save(root / "nested/image.png")
            Image.new("RGB", (64, 64), "green").save(root / "image.png")
        self.boxes = self.root / "boxes.json"

    def test_reused_manifest_preserves_generated_root_and_nested_paths(self):
        for key in (str(self.reference / "nested/image.png"), "nested/image.png"):
            with self.subTest(key=key):
                self.boxes.write_text(json.dumps({key: [[0, 0, 64, 64]]}))

                def verify(reference, generated, **kwargs):
                    with (
                        Image.open(next(reference.glob("*.png"))) as ref,
                        Image.open(next(generated.glob("*.png"))) as gen,
                    ):
                        self.assertEqual(ref.getpixel((0, 0)), (255, 0, 0))
                        self.assertEqual(gen.getpixel((0, 0)), (0, 0, 255))
                    return 3.0

                with patch("worldfoundry.evaluation.tasks.metrics.fid.scene.compute_fid", side_effect=verify):
                    self.assertEqual(
                        compute_scene_fid(self.reference, self.generated, reference_bboxes_json=self.boxes, cuda=False),
                        3.0,
                    )

    def test_image_root_takes_precedence_over_working_directory(self):
        Image.new("RGB", (64, 64), "red").save(self.root / "image.png")
        self.boxes.write_text(json.dumps({"image.png": [[0, 0, 64, 64]]}))
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            crops = extract_object_crops(self.generated, self.boxes, self.root / "crops")
        finally:
            os.chdir(previous_cwd)
        with Image.open(next(crops.glob("*.png"))) as crop:
            self.assertEqual(crop.getpixel((0, 0)), (0, 128, 0))

    def test_separate_generated_manifest_uses_its_own_paths_and_boxes(self):
        generated_image = self.generated / "other.png"
        (self.generated / "nested/image.png").rename(generated_image)
        self.boxes.write_text(json.dumps({str(self.reference / "nested/image.png"): [[0, 0, 64, 64]]}))
        generated_boxes = self.root / "generated-boxes.json"
        generated_boxes.write_text(json.dumps({str(generated_image): [[0, 0, 40, 40]]}))

        def verify(reference, generated, **kwargs):
            with Image.open(next(reference.glob("*.png"))) as ref, Image.open(next(generated.glob("*.png"))) as gen:
                self.assertEqual(ref.size, (64, 64))
                self.assertEqual(gen.size, (40, 40))
                self.assertEqual(gen.getpixel((0, 0)), (0, 0, 255))
            return 3.0

        with patch("worldfoundry.evaluation.tasks.metrics.fid.scene.compute_fid", side_effect=verify):
            self.assertEqual(
                compute_scene_fid(
                    self.reference,
                    self.generated,
                    reference_bboxes_json=self.boxes,
                    generated_bboxes_json=generated_boxes,
                    cuda=False,
                ),
                3.0,
            )

    def test_missing_generated_image_does_not_fall_back_to_reference(self):
        (self.generated / "nested/image.png").unlink()
        self.boxes.write_text(json.dumps({str(self.reference / "nested/image.png"): [[0, 0, 64, 64]]}))
        with (
            self.assertRaises(FileNotFoundError),
            patch("worldfoundry.evaluation.tasks.metrics.fid.scene.compute_fid") as fid,
        ):
            compute_scene_fid(self.reference, self.generated, reference_bboxes_json=self.boxes, cuda=False)
        fid.assert_not_called()

    def test_five_value_boxes_share_geometry_across_metrics(self):
        first, second = [10, 12, 15, 20, 0.9], [15, 12, 15, 20, 0.1]
        self.assertAlmostEqual(bbox_iou(first, second), 0.5)
        corners = [{"label": "object", "bbox": [10, 12, 25, 32]}]
        sized = [{"label": "object", "bbox": first}]
        self.assertEqual(compute_lqs(corners, corners), compute_lqs(corners, sized))
        self.boxes.write_text(json.dumps({"nested/image.png": [first]}))
        crops = extract_object_crops(self.reference, self.boxes, self.root / "crops", min_crop_size=1)
        with Image.open(next(crops.glob("*.png"))) as crop:
            self.assertEqual(crop.size, (15, 20))

    def test_four_value_boxes_preserve_iou_across_coordinate_scales(self):
        for scale in (1, 100):
            with self.subTest(scale=scale):
                guidance = [([v / scale for v in (10, 10, 20, 20)], "chair")]
                detections = [([v / scale for v in (14, 10, 24, 20)], "chair")]
                score = compute_object_wise_consistency(guidance, detections)
                self.assertAlmostEqual(score["object_wise_iou_mean"], 3 / 7)
                self.assertEqual(score["object_wise_success_rate"], 0)


if __name__ == "__main__":
    unittest.main()
