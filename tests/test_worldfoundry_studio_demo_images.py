from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "gradio" dependency at import time; skip when it is unavailable.
pytest.importorskip("gradio")

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from worldfoundry.studio.ui.tray import _on_demo_image_select, _use_demo_image_as_input


class WorldFoundryStudioDemoImagesTest(unittest.TestCase):
    def test_use_demo_image_as_input_loads_selected_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.png"
            Image.new("RGB", (18, 12), "purple").save(image_path)

            image_value, input_path, video_value, status = _use_demo_image_as_input(0, [image_path])

            self.assertIsNotNone(image_value)
            self.assertEqual(image_value.size, (18, 12))
            self.assertEqual(input_path, "")
            self.assertIsNone(video_value)
            self.assertIn("loaded example image into Main Image", status)

    def test_on_demo_image_select_supports_explicit_tray_slot_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_a = Path(tmp_dir) / "image_a.png"
            image_b = Path(tmp_dir) / "image_b.png"
            Image.new("RGB", (16, 12), "red").save(image_a)
            Image.new("RGB", (20, 14), "blue").save(image_b)

            event = SimpleNamespace(index=3)
            image_value, input_path, video_value, status = _on_demo_image_select(
                event,
                demo_image_paths=[image_a, image_b, image_a, image_b],
            )

            self.assertIsNotNone(image_value)
            self.assertEqual(image_value.size, (20, 14))
            self.assertEqual(input_path, "")
            self.assertIsNone(video_value)
            self.assertIn("loaded example image into Main Image", status)


if __name__ == "__main__":
    unittest.main()
