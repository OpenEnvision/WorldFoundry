from __future__ import annotations

import sys
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_runtime.runtime import (
    ENTRYPOINT_MODULE,
    REQUIRED_DATA_FILES,
    REQUIRED_RUNTIME_FILES,
    RUNTIME_ROOT,
    MatrixGame35Assets,
    MatrixGame35Runtime,
)


def test_matrix_game_3p5_required_runtime_files_are_packaged() -> None:
    missing = [
        str(path)
        for path in (
            *(RUNTIME_ROOT / relative for relative in REQUIRED_RUNTIME_FILES),
            *REQUIRED_DATA_FILES,
        )
        if not path.is_file()
    ]

    assert missing == []


@pytest.mark.unit
@pytest.mark.parametrize("person", ["first", "third"])
def test_matrix_game_3p5_plan_uses_its_package_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, person: str
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    runtime = MatrixGame35Runtime(model_id=f"matrix-game-3.5-{person}-person", device="cuda:2")
    assets = MatrixGame35Assets(
        checkpoint=tmp_path / f"{person}-person.safetensors",
        wan_dir=tmp_path / "wan",
        tokenizer_dir=tmp_path / "tokenizer",
        da3_dir=tmp_path / "da3",
        python_executable=Path(sys.executable),
    )
    monkeypatch.setattr(runtime, "resolve_assets", lambda: assets)
    plan = runtime.build_plan(
        image_path=tmp_path / "input.png",
        camera_path=tmp_path / "camera.npz",
        prompt="a city street",
        caption_path=None,
        refs_dir=tmp_path / "refs" if person == "third" else None,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        run_name="sample",
        num_blocks=1,
        steps=25,
        cfg_scale=5.0,
        seed=3407,
        camera_convention="c2w",
        keep_workspace=True,
    )

    assert plan.command[:3] == (sys.executable, "-m", ENTRYPOINT_MODULE)
    assert plan.command[plan.command.index("--person") + 1] == person
    assert plan.command[plan.command.index("--steps") + 1] == "25"
    assert plan.command[plan.command.index("--seed") + 1] == "3407"
    assert "--keep-workspace" in plan.command
    assert ("--refs" in plan.command) is (person == "third")
    assert "PYTHONPATH" not in plan.env
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "2"
    assert plan.env["DA3_MODEL_PATH"] == str(assets.da3_dir)
