from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import BaseRuntimeDriver, PipelineContext, PreparedInputs


class _LazyStreamPipeline:
    def __init__(self) -> None:
        self.consumed = 0

    def stream(self, images=None):
        def chunks():
            for value in ("first", "second"):
                self.consumed += 1
                yield value

        return chunks()


def _request(tmp_path: Path) -> PreparedInputs:
    return PreparedInputs(
        prompt="",
        input_path="",
        image=None,
        image_path=None,
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
        fps=0,
        num_frames=0,
        output_dir=str(tmp_path),
        output_path=str(tmp_path / "preview.mp4"),
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )


class WorldFoundryStudioLazyStreamTest(unittest.TestCase):
    def test_non_materializing_stream_is_not_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _LazyStreamPipeline()
            ctx = PipelineContext(
                entry=find_entry("infinite-world"),
                pipeline=pipeline,
                cache_key="lazy-stream",
                backend="from_pretrained",
                model_ref="",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            result = BaseRuntimeDriver()._invoke(
                ctx,
                _request(Path(tmp_dir)),
                mode="stream",
                materialize_outputs=False,
            )

            self.assertEqual(pipeline.consumed, 0)
            self.assertEqual(next(result), "first")
            self.assertEqual(pipeline.consumed, 1)

    def test_materializing_stream_preserves_existing_list_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = _LazyStreamPipeline()
            ctx = PipelineContext(
                entry=find_entry("infinite-world"),
                pipeline=pipeline,
                cache_key="materialized-stream",
                backend="from_pretrained",
                model_ref="",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            result = BaseRuntimeDriver()._invoke(ctx, _request(Path(tmp_dir)), mode="stream")

            self.assertEqual(result, ["first", "second"])
            self.assertEqual(pipeline.consumed, 2)


if __name__ == "__main__":
    unittest.main()
