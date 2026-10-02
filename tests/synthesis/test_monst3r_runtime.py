from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from worldfoundry.evaluation.utils import REPO_ROOT
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.base_models.three_dimensions.three_d_four_d.runtime import (
    ThreeDFourDRuntimeSynthesis,
    three_d_four_d_runtime_spec,
)


MONST3R_ROOT = REPO_ROOT / "worldfoundry" / "base_models" / "three_dimensions" / "general_3d" / "monst3r"
WORLDGEN_ROOT = (
    REPO_ROOT
    / "worldfoundry"
    / "synthesis"
    / "visual_generation"
    / "worldgen"
    / "worldgen_runtime"
)
KITCHEN_IMAGES = REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "vggt" / "examples" / "kitchen" / "images"
MONST3R_BASE_OPT = MONST3R_ROOT / "dust3r" / "cloud_opt" / "base_opt.py"


def test_monst3r_runtime_spec_uses_in_tree_demo() -> None:
    spec = three_d_four_d_runtime_spec("monst3r")

    assert spec.entrypoint == "demo.py"
    assert spec.command_kind == "monst3r_demo"
    assert spec.artifact_filename == "monst3r.glb"
    assert (MONST3R_ROOT / spec.entrypoint).is_file()


def test_monst3r_plan_uses_in_tree_source_and_staged_multi_image_dir(tmp_path: Path) -> None:
    spec = three_d_four_d_runtime_spec("monst3r")
    runtime = ThreeDFourDRuntimeSynthesis(
        spec=spec,
        source_root=MONST3R_ROOT,
        device="cpu",
        options={
            "weights": "/tmp/monst3r.pth",
            "flow_loss_weight": 0.0,
            "skip_pair_dynamic_mask": True,
            "silent": True,
        },
    )

    result = runtime.predict(
        images=[KITCHEN_IMAGES / "00.png", KITCHEN_IMAGES / "01.png"],
        output_path=tmp_path / "scene.glb",
        run_dir=tmp_path / "run",
        plan_only=True,
    )

    command = result["command"]
    input_dir = Path(command[command.index("--input_dir") + 1])
    assert result["status"] == "prepared"
    assert command[1] == str((MONST3R_ROOT / "demo.py").resolve())
    assert "--flow_loss_weight" in command
    assert command[command.index("--flow_loss_weight") + 1] == "0.0"
    assert "--skip_pair_dynamic_mask" in command
    assert input_dir == tmp_path / "run" / "monst3r_inputs"
    assert sorted(path.name for path in input_dir.iterdir()) == ["00000.png", "00001.png"]


def test_three_d_four_d_runtime_reuses_shared_torch_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    monkeypatch.delenv("TORCH_HOME", raising=False)
    runtime = ThreeDFourDRuntimeSynthesis(
        spec=three_d_four_d_runtime_spec("monst3r"),
        source_root=MONST3R_ROOT,
        device="cpu",
    )

    assert runtime._subprocess_env()["TORCH_HOME"] == str(tmp_path / "torch_hub")


def test_worldgen_plan_forwards_direct_panorama(tmp_path: Path) -> None:
    panorama = tmp_path / "panorama.png"
    panorama.touch()
    runtime = ThreeDFourDRuntimeSynthesis(
        spec=three_d_four_d_runtime_spec("worldgen"),
        source_root=WORLDGEN_ROOT,
        device="cuda",
    )

    result = runtime.predict(
        prompt="coastal city",
        pano_image=panorama,
        output_path=tmp_path / "scene.glb",
        run_dir=tmp_path / "run",
        return_mesh=True,
        plan_only=True,
    )

    command = result["command"]
    assert result["status"] == "prepared"
    assert command[command.index("--pano_image") + 1] == str(panorama)
    assert "--image" not in command
    assert "--return_mesh" in command


def test_monst3r_defaults_accept_official_safetensors_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").touch()
    (checkpoint_dir / "model.safetensors").touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    entry = find_entry("monst3r")

    assert entry.default_call_kwargs["weights"] == str(checkpoint_dir)


def test_monst3r_tum_export_does_not_require_evo(tmp_path: Path) -> None:
    tree = ast.parse(MONST3R_BASE_OPT.read_text())
    save_trajectory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_save_trajectory_tum_format"
    )
    namespace = {"np": np, "to_numpy": np.asarray}
    exec(
        compile(ast.Module(body=[save_trajectory], type_ignores=[]), str(MONST3R_BASE_OPT), "exec"),
        namespace,
    )
    output = tmp_path / "trajectory.txt"
    poses = np.array([[1, 2, 3, 1, 0, 0, 0], [4, 5, 6, 1, 0, 0, 0]], dtype=float)

    namespace["_save_trajectory_tum_format"]([poses, np.array([0.0, 1.0])], output)

    rows = np.loadtxt(output)
    assert rows.shape == (2, 8)
    np.testing.assert_allclose(rows[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(rows[:, 1:], poses)
