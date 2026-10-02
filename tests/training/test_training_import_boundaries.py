from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_recipe_import_does_not_initialize_execution_engines_or_reward_models() -> None:
    root = Path(__file__).resolve().parents[2]
    probe = """
import json
import sys

import worldfoundry.training.recipes

loaded = sorted(sys.modules)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(result.stdout))

    assert not {
        name
        for name in loaded
        if name == "worldfoundry.training.engine" or name.startswith("worldfoundry.training.engine.")
    }
    assert "transformers" not in loaded
    assert "worldfoundry.training.post_training.builders" not in loaded
    assert "worldfoundry.training.post_training.videoalign" not in loaded
