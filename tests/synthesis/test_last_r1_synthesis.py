"""Public LaST-R1 contract checks that do not require checkpoint loading."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from worldfoundry.synthesis.action_generation.last_r1 import LastR1Synthesis
from worldfoundry.synthesis.action_generation.last_r1 import last_r1_synthesis


def test_numpy_image_is_written_and_action_artifact_is_returned(tmp_path, monkeypatch):
    synthesis = object.__new__(LastR1Synthesis)
    synthesis.checkpoint_path = tmp_path / "checkpoint"
    synthesis.source_root = tmp_path / "source"
    synthesis.python_executable = sys.executable
    synthesis.device = "cuda:2"
    monkeypatch.setattr(synthesis, "preflight", lambda: {"status": "ready"})

    class CompletedWorker:
        def __init__(self, command, **_kwargs):
            request = json.loads(Path(command[-1]).read_text())
            with Image.open(request["image_path"]) as image:
                assert image.size == (16, 12)
                assert image.mode == "RGB"
            run_dir = Path(request["run_dir"])
            (run_dir / "result.json").write_text(
                json.dumps({"status": "success", "action_shape": [1, 8, 7], "all_finite": True})
            )

        def wait(self, timeout):
            assert timeout > 0
            return 0

    monkeypatch.setattr(last_r1_synthesis.subprocess, "Popen", CompletedWorker)
    artifact = tmp_path / "action_trace.json"
    result = synthesis.predict(
        prompt="Pick up the block",
        image=np.zeros((12, 16, 3), dtype=np.uint8),
        output_path=artifact,
    )

    assert result["artifact_path"] == str(artifact)
    assert json.loads(artifact.read_text())["action_shape"] == [1, 8, 7]


def test_studio_discovers_the_public_pipeline():
    from worldfoundry.studio.inference.catalog import find_entry, find_runtime_entry

    for lookup in (find_entry, find_runtime_entry):
        entry = lookup("last-r1")
        assert entry.module_path == "worldfoundry.pipelines.component_pipelines"
        assert entry.class_name == "LastR1Pipeline"
        assert entry.category == "Embodied Action"
