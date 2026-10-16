from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import BaseRuntimeDriver, PipelineContext, PreparedInputs, StudioManager


class _DummyMemory:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.records: list[dict[str, object]] = []

    def manage(self, action: str = "reset") -> None:
        self.actions.append(action)

    def record(self, value, metadata=None) -> None:
        self.records.append({"value": value, "metadata": metadata})


class _DummyMemoryStreamPipeline:
    def __init__(self) -> None:
        self.memory_module = _DummyMemory()

    def stream(self, images, interactions=None):
        raise AssertionError("run_init() should not call stream()")


class WorldFoundryStudioStreamInitTest(unittest.TestCase):
    def test_run_init_seeds_memory_without_generating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "main_image.png"
            Image.new("RGB", (24, 24), "green").save(image_path)
            with Image.open(image_path) as source_image:
                seed_image = source_image.convert("RGB")

            manager = StudioManager(workspace_root=str(tmp_path / "studio"))
            driver = BaseRuntimeDriver()
            pipeline = _DummyMemoryStreamPipeline()
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
                image=seed_image,
                image_path=str(image_path),
                video_path=None,
                last_frame=None,
                last_frame_path=None,
                reference_images=[],
                reference_image_paths=[],
                interactions=None,
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

            record = driver.run_init(manager, ctx, request)

            self.assertEqual(pipeline.memory_module.actions, ["reset"])
            self.assertEqual(len(pipeline.memory_module.records), 1)
            self.assertEqual(
                pipeline.memory_module.records[0]["metadata"],
                {"prompt": "seed world", "mode": "init"},
            )
            self.assertTrue(ctx.state["studio_stream_initialized"])
            self.assertEqual(record.mode, "init")
            self.assertEqual(record.status, "succeeded")
            self.assertEqual(record.preview_image, str(image_path))
            self.assertTrue(Path(record.manifest_path).exists())


if __name__ == "__main__":
    unittest.main()
