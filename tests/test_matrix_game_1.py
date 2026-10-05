from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.catalog.registry import discover_model_registry
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_1 import MatrixGame1Pipeline
from worldfoundry.studio.inference import catalog as studio_catalog
from worldfoundry.studio.inference.catalog import discover_catalog, find_entry
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_1_runtime import (
    DEFAULT_RUNTIME_CONFIG_ROOT,
    EXPECTED_CHECKPOINT_FILES,
    REQUIRED_RUNTIME_CONFIG_FILES,
    REQUIRED_RUNTIME_FILES,
    MatrixGame1Runtime,
    MatrixGame1RuntimePlan,
    RUNTIME_ROOT,
)
from worldfoundry.evaluation.models.runtime.assets import load_runtime_asset_profile_by_id
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile


def test_matrix_game_1_binding_and_catalog_are_registered() -> None:
    """Verify Matrix-Game-1 is discoverable across public registries.

    Args:
        None.
    """
    discover_catalog.cache_clear()
    model_registry = discover_model_registry()
    studio_entry = find_entry("matrix-game-1")
    zoo_entry = load_model_zoo_registry().get("matrix-game-1")

    registry_entry = model_registry.get("matrix-game-1")
    assert registry_entry.has_loader is True
    assert registry_entry.pipeline_target.endswith("pipeline_matrix_game_1:MatrixGame1Pipeline")
    assert model_registry.get("matrix-game-1").has_infer is True
    assert studio_entry.model_id == "matrix-game-1"
    assert studio_entry.default_model_ref.endswith(("Skywork--Matrix-Game", "Skywork/Matrix-Game"))
    assert zoo_entry.pipeline_target.endswith("pipeline_matrix_game_1:MatrixGame1Pipeline")
    assert zoo_entry.runner_entry_kind == "runnable_runner"


def test_matrix_game_1_catalog_resolves_persistent_env_layout(tmp_path: Path, monkeypatch) -> None:
    """Prefer the persistent arena ``envs`` layout used by Workspace installs."""

    project_root = tmp_path / "WorldFoundry"
    environment = tmp_path / "envs" / "matrix-game-1"
    environment.mkdir(parents=True)
    monkeypatch.setattr(studio_catalog, "_project_root", lambda: project_root)

    load_kwargs = studio_catalog._matrix_game_1_default_load_kwargs()

    assert load_kwargs["conda_dir"] == str(environment)


def test_matrix_game_1_preflight_reports_incomplete_local_resources(tmp_path: Path) -> None:
    """Verify Matrix-Game-1 fails clearly when checkpoint/runtime assets are absent.

    Args:
        tmp_path: Temporary directory used for isolated fake resources.
    """
    checkpoint_dir = tmp_path / "Skywork--Matrix-Game"
    conda_dir = tmp_path / "matrix-game-1.0"
    checkpoint_dir.mkdir()
    conda_dir.mkdir()

    pipe = MatrixGame1Pipeline.from_pretrained(
        model_path=str(checkpoint_dir),
        required_components={"conda_dir": str(conda_dir)},
        device="cpu",
    )
    preflight = pipe.preflight()

    assert preflight["status"] == "blocked"
    assert preflight["code_ready"] is True
    assert preflight["checkpoint_ready"] is False
    assert preflight["env_ready"] is False
    assert preflight["checkpoint_dir_exists"] is True
    assert preflight["conda_dir_exists"] is True
    assert preflight["runtime_root"] == str(RUNTIME_ROOT)
    assert preflight["runtime_config_root"] == str(DEFAULT_RUNTIME_CONFIG_ROOT)
    assert set(preflight["required_runtime_files"]) == set(REQUIRED_RUNTIME_FILES)
    assert set(preflight["required_runtime_config_files"]) == set(REQUIRED_RUNTIME_CONFIG_FILES)
    assert set(preflight["expected_checkpoint_files"]) == set(EXPECTED_CHECKPOINT_FILES)
    assert preflight["missing_runtime_config_files"] == []
    assert any("diffusion_pytorch_model-00001" in item for item in preflight["missing_assets"])
    assert any(item.endswith("/bin/python") for item in preflight["missing_runtime"])


def test_matrix_game_1_runtime_api_is_lightweight_and_in_tree() -> None:
    """Verify Matrix-Game-1 has an in-tree runtime facade that does not import heavy models.

    Args:
        None.
    """
    runtime = MatrixGame1Runtime(
        checkpoint_dir=str(Path.home() / ".cache" / "worldfoundry" / "checkpoints" / "hfd" / "Skywork--Matrix-Game"),
        conda_dir=str(Path.home() / ".cache" / "worldfoundry" / "conda" / "matrix-game-1.0"),
    )
    preflight = runtime.preflight()

    assert RUNTIME_ROOT.joinpath("inference_bench.py").is_file()
    assert not RUNTIME_ROOT.joinpath("environment.yml").exists()
    assert DEFAULT_RUNTIME_CONFIG_ROOT.joinpath("environment.yml").is_file()
    assert RUNTIME_ROOT.joinpath("matrixgame/sample/pipeline_matrixgame.py").is_file()
    assert not RUNTIME_ROOT.joinpath("matrixgame/model_variants/matrixgame_dit_src").exists()
    assert RUNTIME_ROOT.joinpath("flash-attention").exists() is False
    assert RUNTIME_ROOT.joinpath("apex").exists() is False
    assert preflight["code_ready"] is True
    assert isinstance(preflight["checkpoint_ready"], bool)


def test_matrix_game_1_pipeline_rejects_blocked_plan(tmp_path: Path) -> None:
    """Verify the pipeline fails fast instead of writing a fake preflight artifact.

    Args:
        tmp_path: Temporary directory used for the plan artifact.
    """
    pipe = MatrixGame1Pipeline.from_pretrained(
        model_path=str(tmp_path / "checkpoint"),
        required_components={"conda_dir": str(tmp_path / "env")},
        device="cpu",
    )
    output_path = tmp_path / "matrix_game_1_plan.json"
    with pytest.raises(RuntimeError, match="Matrix-Game-1 cannot run"):
        pipe(
            input_image=Image.new("RGB", (8, 8), "blue"),
            interaction_signal=["forward"],
            output_path=output_path,
            fps=12,
            prompt="minecraft scene",
        )

    assert not output_path.exists()


def test_matrix_game_1_public_execute_path_records_generated_video(tmp_path: Path) -> None:
    """Verify the public pipeline path can return a generated-video artifact when preflight is ready.

    Args:
        tmp_path: Temporary directory used for generated artifacts.
    """
    pipe = MatrixGame1Pipeline.from_pretrained(
        model_path=str(tmp_path / "checkpoint"),
        required_components={"conda_dir": str(tmp_path / "env")},
        device="cuda",
    )
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    generated = tmp_path / "outputs" / "matrix_game_1.mp4"
    generated.parent.mkdir()
    generated.write_bytes(b"mp4")
    plan = MatrixGame1RuntimePlan(
        command=("python", "inference_bench.py"),
        env={},
        workdir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "checkpoint"),
        output_dir=str(generated.parent),
    )
    synthesis = pipe.synthesis_model
    synthesis.runtime.preflight = lambda: {
        "status": "ready",
        "code_ready": True,
        "checkpoint_ready": True,
        "env_ready": True,
    }
    synthesis.runtime.build_plan = lambda **kwargs: plan
    synthesis.runtime.run_plan = lambda runtime_plan, **kwargs: {
        "ok": True,
        "status": "success",
        "returncode": 0,
        "duration_seconds": 0.1,
        "generated_count": 1,
        "generated_files": [str(generated)],
        "stdout_path": str(tmp_path / "stdout.log"),
        "stderr_path": str(tmp_path / "stderr.log"),
        "runtime_plan": runtime_plan.to_dict(),
    }

    report_path = tmp_path / "matrix_game_1_report.json"
    result = pipe(
        input_image=None,
        interaction_signal=["forward"],
        output_path=report_path,
        fps=12,
        prompt="minecraft scene",
        image_path=str(image_path),
        output_dir=tmp_path / "outputs",
        execute=True,
        inference_steps=1,
        return_dict=True,
    )

    assert result["status"] == "success"
    assert result["artifact_kind"] == "generated_video"
    assert result["artifact_path"] == str(generated)
    assert result["backend_quality"] == "public_runner_official_runner"
    assert report_path.is_file()


def test_matrix_game_1_build_plan_normalizes_image_file_for_official_runner(tmp_path: Path) -> None:
    """Verify file inputs are converted to the image directory expected by the runner.

    Args:
        tmp_path: Temporary directory used for the input image and output directory.
    """
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    runtime = MatrixGame1Runtime(checkpoint_dir=str(tmp_path / "checkpoint"))

    plan = runtime.build_plan(image_path=str(image_path), output_dir=tmp_path / "outputs")
    command = list(plan.command)

    assert command[command.index("--image_path") + 1] == str(tmp_path)


def test_matrix_game_1_runtime_profile_documents_world_model_runner_smoke() -> None:
    """Verify Matrix-Game-1 profile records the world-model runner smoke.

    Args:
        None.
    """
    profile = load_runtime_profile("matrix-game-1")
    assets = load_runtime_asset_profile_by_id("matrix-game-1", runtime_profiles={"matrix-game-1": profile})

    assert profile.task_family == "world_model"
    assert profile.artifact_kind == "generated_world"
    assert profile.artifact_filename == "world.mp4"
    assert profile.integration_status == "integrated"
    assert profile.backend_stage == "in_tree_runtime"
    assert assets.assets[0].repo_id == "Skywork/Matrix-Game"
    assert any("never routes through Matrix-Game-2" in note for note in profile.notes)
