from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.models.runtime import (
    RuntimeProfile,
    load_runtime_asset_profile_by_id,
    load_runtime_asset_profiles,
    load_runtime_environment_profile_by_id,
    load_runtime_environment_profiles,
    load_runtime_profile_manifest,
    load_runtime_profiles,
    validate_pipeline_aliases_against_bindings,
    validate_runtime_profile_references,
)
from worldfoundry.evaluation.models.catalog.registry import discover_model_registry


def test_runtime_environment_profiles_use_legacy_conda_specs(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy_envs"
    (legacy_root / "video").mkdir(parents=True)
    (legacy_root / "video" / "demo.yaml").write_text(
        "\n".join(
            [
                "model_id: demo-model",
                "env_name: worldfoundry-demo",
                "python: '3.11'",
                "cuda_profile: cpu",
                "conda_packages:",
                "- python",
                "pip_packages:",
                "- torch",
                "validation_imports:",
                "- torch",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profiles = load_runtime_environment_profiles(
        root=tmp_path / "missing_target",
        legacy_root=legacy_root,
        env_root=tmp_path / "envs",
    )

    profile = profiles["demo-model"]
    assert profile.source == "legacy"
    assert profile.env_name == "worldfoundry-demo"
    assert profile.python == "3.11"
    assert profile.cuda_profile == "cpu"
    assert profile.to_dict()["env_prefix"].endswith("envs/worldfoundry-demo")


def test_runtime_profile_manifest_parses_target_schema(tmp_path: Path) -> None:
    profile_path = tmp_path / "demo.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "schema_version: 2",
                "profile_id: demo-profile",
                "model_id: demo-model",
                "name: Demo Target Runtime",
                "task_family: world_model",
                "groups:",
                "- world",
                "artifact:",
                "  kind: generated_world",
                "  filename: world.mp4",
                "inputs:",
                "  required:",
                "  - image",
                "  - interactions",
                "  optional:",
                "  - prompt",
                "execution:",
                "  default_device: cuda",
                "  environment: demo-env",
                "  assets: demo-assets",
                "  pipeline_binding: demo-binding",
                "output:",
                "  default_dir_env: WORLDFOUNDRY_OUTPUT_DIR",
                "notes:",
                "- target profile fixture",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = load_runtime_profile_manifest(profile_path)

    assert profile.schema_version == 2
    assert profile.model_id == "demo-model"
    assert profile.display_name == "Demo Target Runtime"
    assert profile.task_family == "world_model"
    assert profile.artifact_kind == "generated_world"
    assert profile.input_schema == {"required": ["image", "interactions"], "optional": ["prompt"]}
    assert profile.execution["environment"] == "demo-env"
    assert profile.output["default_dir_env"] == "WORLDFOUNDRY_OUTPUT_DIR"


def test_runtime_profile_manifest_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad.yaml"
    profile_path.write_text(
        "schema_version: 99\nprofile_id: bad\nmodel_id: bad\nname: Bad\ntask_family: video_generation\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_runtime_profile_manifest(profile_path)


def test_runtime_profile_target_tree_overrides_legacy_synthesis_and_resolves_env(tmp_path: Path) -> None:
    acquisition_root = tmp_path / "acquisition"
    target_root = tmp_path / "runtime_profiles"
    env_root = tmp_path / "envs"
    acquisition_root.mkdir()
    target_root.mkdir()
    env_root.mkdir()
    (acquisition_root / "demo.yaml").write_text(
        "\n".join(
            [
                "id: demo-model",
                "taxonomy_name: Legacy Runtime Name",
                "groups:",
                "- video",
                "integration:",
                "  status: planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (target_root / "demo.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "profile_id: target-demo-profile",
                "model_id: demo-model",
                "display_name: Target Runtime Name",
                "task_family: action_trace",
                "artifact:",
                "  kind: action_trace",
                "  filename: action_trace.json",
                "execution:",
                "  environment: demo-env",
                "  backend_stage: target_runtime",
                "  runtime_status: target_verified",
                "  integration_status: integrated",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (env_root / "demo-env.yaml").write_text(
        "\n".join(
            [
                "model_id: demo-env",
                "env_name: worldfoundry-demo-env",
                "python: '3.11'",
                "cuda_profile: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profiles = load_runtime_profiles(
        manifest_path=acquisition_root,
        profile_path=tmp_path / "missing_legacy_profiles",
        target_profile_path=target_root,
        conda_env_path=env_root,
        check_conda_env_exists=False,
    )

    profile = profiles["demo-model"]
    assert profile.display_name == "Target Runtime Name"
    assert profile.artifact_kind == "action_trace"
    assert profile.artifact_filename == "action_trace.json"
    assert profile.backend_stage == "target_runtime"
    assert profile.runtime_status == "target_verified"
    assert profile.integration_status == "integrated"
    assert profile.conda_env["env_name"] == "worldfoundry-demo-env"


def test_runtime_profile_target_tree_wins_over_legacy_profile_for_same_model_id(tmp_path: Path) -> None:
    legacy_root = tmp_path / "official_model_runtimes"
    target_root = tmp_path / "runtime_profiles"
    legacy_root.mkdir()
    target_root.mkdir()
    (legacy_root / "demo.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "model_id: demo-model",
                "display_name: Legacy Runtime",
                "task_family: video_generation",
                "execution:",
                "  backend_stage: legacy_runtime",
                "  pipeline_binding: legacy-binding",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (target_root / "demo.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "model_id: demo-model",
                "display_name: Target Runtime",
                "task_family: video_generation",
                "execution:",
                "  backend_stage: target_runtime",
                "  pipeline_binding: target-binding",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profiles = load_runtime_profiles(
        manifest_path=tmp_path / "missing_acquisition",
        profile_path=tmp_path / "missing_legacy_profile_overrides",
        target_profile_path=(legacy_root, target_root),
        conda_env_path=tmp_path / "missing_envs",
        check_conda_env_exists=False,
    )

    profile = profiles["demo-model"]
    assert profile.display_name == "Target Runtime"
    assert profile.backend_stage == "target_runtime"
    assert profile.execution["pipeline_binding"] == "target-binding"


def test_runtime_environment_target_manifest_overrides_legacy(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy_envs"
    target_root = tmp_path / "target_envs"
    legacy_root.mkdir()
    target_root.mkdir()
    (legacy_root / "demo.yaml").write_text(
        "model_id: demo-model\nenv_name: legacy-env\npython: '3.10'\n",
        encoding="utf-8",
    )
    (target_root / "demo.yaml").write_text(
        "\n".join(
            [
                "environment_id: target-demo",
                "model_id: demo-model",
                "python: '3.12'",
                "system:",
                "  cuda: cu121",
                "packages:",
                "  conda:",
                "  - python",
                "  pip:",
                "  - diffusers",
                "env:",
                "  required:",
                "  - WORLDFOUNDRY_MODEL_DIR",
                "  optional:",
                "  - WORLDFOUNDRY_OUTPUT_DIR",
                "commands:",
                "  setup:",
                "  - pip install -e .",
                "conda:",
                "  env_name: target-env",
                "validation:",
                "  imports:",
                "  - diffusers",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = load_runtime_environment_profile_by_id("demo-model", root=target_root, legacy_root=legacy_root)

    assert profile.source == "target"
    assert profile.schema_version is None
    assert profile.environment_id == "target-demo"
    assert profile.env_name == "target-env"
    assert profile.python == "3.12"
    assert profile.cuda_profile == "cu121"
    assert profile.pip_packages == ("diffusers",)
    assert profile.env_required == ("WORLDFOUNDRY_MODEL_DIR",)
    assert profile.env_optional == ("WORLDFOUNDRY_OUTPUT_DIR",)
    assert profile.setup_commands == ("pip install -e .",)


def test_runtime_environment_manifest_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "bad-env.yaml"
    path.write_text(
        "schema_version: 99\nenvironment_id: bad\nmodel_id: bad\nconda:\n  env_name: bad-env\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_runtime_environment_profile_by_id("bad", root=tmp_path)


def test_runtime_asset_profiles_can_be_synthesized_from_runtime_profiles(tmp_path: Path) -> None:
    runtime_profile = RuntimeProfile(
        model_id="demo-model",
        display_name="Demo Model",
        task_family="video_generation",
        source_repos=({"url": "https://example.com/demo/model.git", "revision": "abc123"},),
        checkpoints=(
            {
                "repo_id": "example/demo-model",
                "local_dir": str(tmp_path / "hf" / "example--demo-model"),
                "role": "primary_checkpoint",
            },
        ),
    )

    profiles = load_runtime_asset_profiles(
        root=tmp_path / "missing_target_assets",
        runtime_profiles={"demo-model": runtime_profile},
    )

    profile = profiles["demo-model"]
    assert profile.source == "runtime_profile"
    assert profile.assets[0].asset_id == "primary_checkpoint"
    assert profile.assets[0].repo_id == "example/demo-model"
    assert profile.source_repos[0]["revision"] == "abc123"


def test_runtime_asset_target_manifest_overrides_synthesized_profile(tmp_path: Path) -> None:
    runtime_profile = RuntimeProfile(
        model_id="demo-model",
        display_name="Demo Model",
        task_family="video_generation",
        checkpoints=({"repo_id": "example/legacy", "role": "legacy_checkpoint"},),
    )
    target_root = tmp_path / "assets"
    target_root.mkdir()
    (target_root / "demo.yaml").write_text(
        "\n".join(
            [
                "asset_profile_id: target-demo-assets",
                "model_id: demo-model",
                "roots:",
                "  hfd: ${WORLDFOUNDRY_HFD_ROOT}",
                "checkpoints:",
                "  primary:",
                "    repo_id: example/target",
                "    local_candidates:",
                "    - ${WORLDFOUNDRY_REPO_ROOT}/cache/demo-target",
                "components:",
                "  tokenizer:",
                "    kind: component",
                "    repo_id: example/tokenizer",
                "source_repos:",
                "- url: https://example.com/target.git",
                "notes:",
                "- target asset profile wins over synthesized runtime profile.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = load_runtime_asset_profile_by_id(
        "demo-model",
        root=target_root,
        runtime_profiles={"demo-model": runtime_profile},
    )

    assert profile.source == "target"
    assert profile.schema_version is None
    assert profile.asset_profile_id == "target-demo-assets"
    assert profile.roots["hfd"]
    assert profile.assets[0].asset_id == "primary"
    assert profile.assets[0].repo_id == "example/target"
    assert "cache/demo-target" in profile.assets[0].local_candidates[0]
    assert profile.components["tokenizer"]["repo_id"] == "example/tokenizer"
    assert profile.assets[1].asset_id == "tokenizer"


def test_runtime_asset_manifest_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "bad-assets.yaml"
    path.write_text(
        "schema_version: 99\nasset_profile_id: bad\nmodel_id: bad\ncheckpoints:\n  primary:\n    repo_id: example/bad\n",
        encoding="utf-8",
    )
    runtime_profile = RuntimeProfile(model_id="bad", display_name="Bad", task_family="video_generation")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_runtime_asset_profile_by_id("bad", root=tmp_path, runtime_profiles={"bad": runtime_profile})


def test_runtime_profiles_cover_matrix_game_1_binding_environment_and_assets() -> None:
    runtime_profiles = load_runtime_profiles(check_conda_env_exists=False)
    runtime_profile = runtime_profiles["matrix-game-1"]
    environment = load_runtime_environment_profile_by_id("matrix-game-1")
    assets = load_runtime_asset_profile_by_id("matrix-game-1", runtime_profiles=runtime_profiles)
    registry_entry = discover_model_registry().get("matrix-game-1")

    assert runtime_profile.display_name == "Matrix-Game 1.0"
    assert runtime_profile.schema_version is None
    assert runtime_profile.task_family == "world_model"
    assert runtime_profile.artifact_kind == "generated_world"
    assert runtime_profile.artifact_filename == "world.mp4"
    assert environment.source == "target"
    assert environment.schema_version is None
    assert environment.env_name == "worldfoundry-unified-cu128"
    assert "torch" in environment.validation_imports
    assert assets.source == "runtime_profile"
    assert assets.assets[0].repo_id == "Skywork/Matrix-Game"
    assert registry_entry.pipeline_target == "worldfoundry.pipelines.matrix_game.pipeline_matrix_game_1:MatrixGame1Pipeline"


def test_runtime_profile_validator_checks_references_artifacts_and_aliases(tmp_path: Path) -> None:
    bad_profile = RuntimeProfile(
        model_id="bad-model",
        display_name="Bad Model",
        task_family="video_generation",
        artifact_kind="not_a_real_artifact",
        artifact_filename="../bad",
        execution={
            "environment": "missing-env",
            "assets": "missing-assets",
            "pipeline_binding": "missing-binding",
        },
    )
    binding_root = tmp_path / "bindings"
    alias_root = tmp_path / "aliases"
    binding_root.mkdir()
    alias_root.mkdir()
    (binding_root / "alpha.yaml").write_text(
        "\n".join(
            [
                "binding_id: alpha",
                "model_id: alpha",
                "pipeline:",
                "  target: pkg.alpha:Pipeline",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (alias_root / "bad.yaml").write_text(
        "aliases:\n  bad-alias: missing-canonical\n",
        encoding="utf-8",
    )

    profile_issues = validate_runtime_profile_references(
        bad_profile,
        environment_root=str(tmp_path / "envs"),
        asset_root=str(tmp_path / "assets"),
        binding_root=str(binding_root),
    )
    alias_issues = validate_pipeline_aliases_against_bindings(
        alias_root=str(alias_root),
        binding_root=str(binding_root),
    )
    default_alias_issues = validate_pipeline_aliases_against_bindings()

    assert {issue.code for issue in profile_issues} == {
        "invalid_artifact_filename",
        "invalid_artifact_kind",
        "pipeline_binding_missing",
        "runtime_assets_missing",
        "runtime_environment_missing",
    }
    assert {issue.code for issue in alias_issues} == {"pipeline_alias_unknown_canonical"}
    assert default_alias_issues == ()
