# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Package launch, local asset and plan contracts for both Uni3C modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from worldfoundry.synthesis.visual_generation.uni3c import uni3c_synthesis as synthesis

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "mode,gpus,options,flags",
    [
        ("camera", 1, {}, []),
        ("camera", 2, {}, ["--enable_sp", "--fsdp"]),
        ("camera", 2, {"fsdp": False}, ["--enable_sp"]),
        ("unified", 1, {}, []),
        ("unified", 1, {"save_gpu_memory": True}, ["--save-gpu-memory"]),
        ("unified", 2, {}, ["--enable_sp"]),
    ],
)
def test_uni3c_command_preserves_mode_and_acceleration_options(
    mode: str, gpus: int, options: dict[str, Any], flags: list[str]
) -> None:
    model = synthesis.Uni3CSynthesis()
    assets = {"controller_path": "/controller", "base_model_path": "/base", "model_path": "/unified"}
    command = model._command(
        mode=mode,
        reference_image="/image.png",
        render_path="/render",
        output=Path("/output.mp4"),
        prompt="orchard",
        num_frames=9,
        max_area=114688,
        seed=17,
        fps=12,
        num_gpus=gpus,
        assets=assets,
        options={"python_executable": "/env/python", **options},
    )
    module = "worldfoundry.synthesis.visual_generation.uni3c.uni3c_runtime." + (
        "cam_control" if mode == "camera" else "uni3c_inference"
    )
    prefix = (
        ["/env/python", "-m"]
        if gpus == 1
        else ["/env/python", "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m"]
    )
    assert command[: len(prefix) + 1] == [*prefix, module]
    assert "--standalone" not in command
    assert command[command.index("--nframe") + 1] == "9"
    assert command[command.index("--config_path") + 1] == "/controller/config.json"
    assert command[command.index("--controlnet_path") + 1] == "/controller/controlnet.pth"
    for flag in ("--enable_sp", "--fsdp", "--save-gpu-memory"):
        assert (flag in command) == (flag in flags)
    assert ("--model_path" in command) == (mode == "unified")


@pytest.mark.parametrize("mode", ["camera", "unified"])
@pytest.mark.parametrize("plan_only", [False, True])
def test_uni3c_predict_materializes_assets_and_package_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, plan_only: bool
) -> None:
    controller, base, unified, render = (tmp_path / name for name in ("controller", "base", "unified", "render"))
    for path in (controller, base, unified, render):
        path.mkdir()
    for name in ("config.json", "controlnet.pth"):
        (controller / name).touch()
    for name in ("render.mp4", "render_mask.mp4", "cam_info.json", "smpl_render.mp4", "hand_render.mp4"):
        (render / name).touch()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def execute(command: list[str], **kwargs: Any) -> dict[str, str]:
        calls.append((command, kwargs))
        return {"video": str(kwargs["output_path"]), "status": "succeeded"}

    monkeypatch.setattr(synthesis, "execute_in_tree", execute)
    model = synthesis.Uni3CSynthesis.from_pretrained(
        controller, base_model_path=base, model_path=unified, mode=mode, python_executable="/env/python"
    )
    image = Image.new("RGB", (4, 3), (12, 23, 34))
    result = model.predict(
        prompt="orchard",
        images=[image],
        interactions={"render_path": render},
        run_dir=tmp_path / "run",
        num_frames=9,
        fps=12,
        plan_only=plan_only,
    )
    plan = json.loads(Path(result["plan_path"]).read_text())
    assert plan["command"][1] == "-m"
    assert plan["command"][2].startswith("worldfoundry.synthesis.visual_generation.uni3c.")
    assert plan["mode"] == mode
    assert plan["missing_requirements"] == []
    with Image.open(plan["reference_image"]) as reference:
        assert reference.tobytes() == image.tobytes()
    if plan_only:
        assert result["status"] == "prepared"
        assert calls == []
    else:
        command, kwargs = calls[0]
        assert command == plan["command"]
        assert kwargs["cwd"] == synthesis.runtime_root()
        assert "python_paths" not in kwargs
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert result["status"] == "succeeded"


def test_uni3c_environment_uses_installed_package() -> None:
    import yaml

    from worldfoundry.core.io.paths import resolve_data_path

    manifest = yaml.safe_load(resolve_data_path("models", "runtime", "environments", "video", "uni3c.yaml").read_text())
    assert "pythonpath_dirs" not in manifest
    assert "xfuser[flash-attn]" in manifest["pip_packages"]
