from __future__ import annotations

import numpy as np
from PIL import Image

from worldfoundry.pipelines.infinite_world.pipeline_infinite_world import InfiniteWorldPipeline
from worldfoundry.studio.inference import catalog as catalog
from worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime import plan


def test_catalog_normalizes_checkpoint_directory_to_model_root(monkeypatch, tmp_path) -> None:
    model_root = tmp_path / "Infinite-World"
    checkpoint_dir = model_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "infinite_world_model.ckpt").touch()
    monkeypatch.setattr(catalog, "_cache_candidates", lambda *_: [str(checkpoint_dir)])

    assert catalog._infinite_world_default_ref() == str(model_root.resolve())


def test_runtime_accepts_unified_model_path_mapping(monkeypatch, tmp_path) -> None:
    model_root = tmp_path / "Infinite-World"
    for relative in plan.REQUIRED_RUNTIME_FILES:
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(plan, "missing_python_modules", lambda: [])

    assert plan.resolve_in_tree_model_root({"model_path": str(model_root)}) == model_root.resolve()


def test_pipeline_materializes_requested_output(monkeypatch, tmp_path) -> None:
    class Synthesis:
        validation_num_frames = 9

        def predict(self, **_):
            frames = np.zeros((9, 2, 2, 3), dtype=np.uint8)
            return {
                "video": frames.astype(np.float32) / 255.0,
                "video_uint8": frames,
                "num_frames": 9,
                "num_chunks": 1,
            }

    pipeline = InfiniteWorldPipeline(synthesis_model=Synthesis())
    monkeypatch.setattr(
        pipeline,
        "process",
        lambda **_: {
            "condition_video": object(),
            "num_condition_frames": 1,
            "operator_condition": {
                "actions": ["forward"] * 9,
                "move_ids": object(),
                "view_ids": object(),
            },
            "prompt": "test",
        },
    )
    writes = []
    monkeypatch.setattr(
        "worldfoundry.pipelines.infinite_world.pipeline_infinite_world.write_video",
        lambda frames, path, fps: writes.append((frames, path, fps)),
    )
    output_path = tmp_path / "infinite-world.mp4"

    result = pipeline(
        images=Image.new("RGB", (2, 2)),
        interactions=["forward"],
        output_path=output_path,
        fps=12,
        return_dict=True,
    )

    assert result["artifact_path"] == str(output_path.resolve())
    assert result["fps"] == 12
    assert writes[0][1:] == (output_path.resolve(), 12)
