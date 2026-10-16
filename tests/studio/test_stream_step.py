from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import BaseRuntimeDriver, PipelineContext, PreparedInputs, StudioManager


class _DummyMemory:
    def __init__(self) -> None:
        self.actions = []
        self.records = []

    def manage(self, action: str = "reset") -> None:
        self.actions.append(action)

    def record(self, value, metadata=None) -> None:
        self.records.append({"value": value, "metadata": metadata})


class _DummyRequiredImagesStreamPipeline:
    def __init__(self) -> None:
        self.memory_module = _DummyMemory()
        self.calls = []

    def stream(self, images, interactions):
        self.calls.append({"images": images, "interactions": interactions})
        return Image.new("RGB", (16, 16), "blue")


class WorldFoundryStudioStreamStepTest(unittest.TestCase):
    def test_stream_auto_seeds_memory_before_first_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "main_image.png"
            Image.new("RGB", (16, 16), "red").save(image_path)
            manager = StudioManager(workspace_root=str(tmp_path / "studio"))
            driver = BaseRuntimeDriver()
            pipeline = _DummyRequiredImagesStreamPipeline()
            ctx = PipelineContext(
                entry=find_entry("infinite-world"),
                pipeline=pipeline,
                cache_key="infinite-world",
                backend="from_pretrained",
                model_ref="",
                endpoint="",
                load_kwargs={},
                device="cpu",
                state={},
            )

            request = PreparedInputs(
                prompt="seed world",
                input_path="",
                image=Image.new("RGB", (16, 16), "red"),
                image_path=str(image_path),
                video_path=None,
                last_frame=None,
                last_frame_path=None,
                reference_images=[],
                reference_image_paths=[],
                interactions=["left"],
                camera_view=None,
                task_type="",
                intrinsics=None,
                meta_path="",
                panorama_path="",
                scene_name="",
                fps=8,
                num_frames=0,
                output_dir=str(tmp_path / "run_auto_seed"),
                output_path=str(tmp_path / "run_auto_seed" / "preview.mp4"),
                call_kwargs={},
                load_kwargs={},
                model_ref="",
                backend="from_pretrained",
                endpoint="",
                api_key="",
                device="cpu",
            )
            Path(request.output_dir).mkdir(parents=True, exist_ok=True)

            record = driver.run_continue(manager, ctx, request)

            self.assertEqual(pipeline.memory_module.actions, ["reset"])
            self.assertEqual(len(pipeline.memory_module.records), 1)
            self.assertEqual(
                pipeline.memory_module.records[0]["metadata"],
                {"prompt": "seed world", "mode": "init"},
            )
            self.assertEqual(len(pipeline.calls), 1)
            self.assertIsNone(pipeline.calls[0]["images"])
            self.assertEqual(pipeline.calls[0]["interactions"], ["left"])
            self.assertTrue(ctx.state["studio_stream_initialized"])
            self.assertEqual(record.status, "succeeded")
            self.assertIsNotNone(record.preview_image)

    def test_stream_keeps_explicit_none_for_required_images_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manager = StudioManager(workspace_root=str(tmp_path / "studio"))
            driver = BaseRuntimeDriver()
            pipeline = _DummyRequiredImagesStreamPipeline()
            ctx = PipelineContext(
                entry=find_entry("infinite-world"),
                pipeline=pipeline,
                cache_key="infinite-world",
                backend="from_pretrained",
                model_ref="",
                endpoint="",
                load_kwargs={},
                device="cpu",
                state={"studio_stream_initialized": True},
            )

            request = PreparedInputs(
                prompt="",
                input_path="",
                image=Image.new("RGB", (16, 16), "red"),
                image_path=str(tmp_path / "main_image.png"),
                video_path=None,
                last_frame=None,
                last_frame_path=None,
                reference_images=[],
                reference_image_paths=[],
                interactions=["left"],
                camera_view=None,
                task_type="",
                intrinsics=None,
                meta_path="",
                panorama_path="",
                scene_name="",
                fps=8,
                num_frames=0,
                output_dir=str(tmp_path / "run"),
                output_path=str(tmp_path / "run" / "preview.mp4"),
                call_kwargs={},
                load_kwargs={},
                model_ref="",
                backend="from_pretrained",
                endpoint="",
                api_key="",
                device="cpu",
            )
            Path(request.output_dir).mkdir(parents=True, exist_ok=True)

            record = driver.run_continue(manager, ctx, request)

            self.assertEqual(len(pipeline.calls), 1)
            self.assertIsNone(pipeline.calls[0]["images"])
            self.assertEqual(pipeline.calls[0]["interactions"], ["left"])
            self.assertEqual(record.status, "succeeded")
            self.assertIsNotNone(record.preview_image)


if __name__ == "__main__":
    unittest.main()
