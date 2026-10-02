from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

import pytest

# worldfoundry.studio.serving.workspace requires the optional fastapi dependency at
# import time; skip this module in environments without the [api] extra.
pytest.importorskip("fastapi")

from worldfoundry.evaluation.api import GenerationRequest, WorldModelConfig
from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models import resolve_model_zoo_config, resolve_model_zoo_runner
from worldfoundry.evaluation.models.pipelines.loading import build_pipeline_runner_spec
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile, load_runtime_profile_manifest
from worldfoundry.studio.inference.catalog import discover_catalog
from worldfoundry.studio.serving.workspace import _workspace_models


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
MODEL_DATA_DIR = REPO_ROOT / "worldfoundry" / "data" / "models"
SRC_ROOT = REPO_ROOT
UNIFIED_MODEL_RUNNER_TARGET = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"
PIPELINE_MODEL_IDS = (
    "longcat-video",
)
TARGET_PIPELINE_MODEL_IDS = (
    "4d-gs",
    "fantasyworld",
    "flashworld",
    "lagernvs",
    "lyra",
    "monst3r",
    "mvdiffusion",
    "neoverse",
    "recammaster",
    "shape-of-motion",
    "stable-virtual-camera",
    "wonderjourney",
    "wonderworld",
    "worldgen",
    "allegro",
    "cogvideox",
    "dynamicrafter",
    "easyanimate",
    "framepack",
    "hunyuanvideo-1.5",
    "hunyuanvideo",
    "i2vgen-xl",
    "ltx-2.x",
    "ltx-video",
    "magi-1",
    "mochi-1",
    "modelscope-t2v",
    "open-sora-plan",
    "open-sora",
    "skyreels-v2",
    "videocrafter",
    "wan2.1",
    "wan2.2",
    "emu3.5",
    "krea-realtime-video",
    "mmaudio",
    "omnivinci",
    "qwen2.5-omni",
    "sama-14b",
    "skyreels-v3",
    "spatial-ladder",
    "spatial-reasoner",
    "thinksound",
    "unianimate-dit",
    "wan2.1-vace",
    "being-h07",
    "eo1",
    "fastwam",
    "gaussian-actor",
    "hy-embodied",
    "last-r1",
    "libero-para",
    "multi-task-dit",
    "openpie-0.6",
    "pi0-fast",
    "pi0-worldfoundry",
    "pi0",
    "pi05",
    "real-time-chunking",
    "smolvla",
    "spirit-v1.5",
    "tdmpc",
    "vqbet",
    "wall-oss",
    "xvla",
    "cosmos-predict-2.5",
    "cosmos-transfer-2.5",
    "dreamx-world-5b-cam",
    "hunyuanworld-1",
    "hunyuanworld-voyager",
    "lingbot-world",
    "matrix-game-2",
    "viewcrafter",
    "adaworld",
    "ai2thor",
    "ctrl-world",
    "diamond",
    "dino-wm",
    "droid-w",
    "egowm",
    "genie-envisioner",
    "giga-world-0",
    "happyoyster",
    "hma",
    "leworldmodel",
    "mineworld",
    "mosaicmem",
    "motionbricks",
    "nwm",
    "oasis-500m",
    "omniforcing",
    "pointworld",
    "sana-wm",
    "shotstream",
    "simworld",
    "starwm",
    "tesseract",
    "uwm",
    "vggt-world",
    "vid2world",
    "wilddet3d",
    "wildworld",
    "worldgrow",
    "worldmem",
    "wow",
)
STUDIO_PIPELINE_MODEL_IDS = tuple(model_id for model_id in PIPELINE_MODEL_IDS if model_id != "vid2world")


def _pipeline_contract_from_target(target: str) -> dict[str, Any] | None:
    return _pipeline_contract_from_target_seen(target, seen=set())


def _pipeline_contract_from_target_seen(target: str, *, seen: set[str]) -> dict[str, Any] | None:
    if target in seen:
        return None
    seen.add(target)
    module_name, _, class_name = target.partition(":")
    path = _module_source_path(module_name)
    if path is None or not path.is_file() or not class_name:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return None
    if module_name == "worldfoundry.pipelines.component_pipelines":
        return _component_pipeline_contract(tree, class_name)
    class_defs = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    node = class_defs.get(class_name)
    if node is None:
        return None
    imports = _imported_class_targets(tree, module_name)
    return {
        "has_operator": _class_has_pipeline_contract(
            node,
            class_defs,
            imports=imports,
            seen_classes=set(),
            seen_targets=seen,
        ),
    }


def _module_source_path(module_name: str) -> Path | None:
    if not module_name.startswith("worldfoundry."):
        return None
    relative = Path(*module_name.split(".")).with_suffix(".py")
    path = SRC_ROOT / relative
    if path.exists():
        return path
    package_init = SRC_ROOT / Path(*module_name.split(".")) / "__init__.py"
    if package_init.exists():
        return package_init
    return path


def _class_has_pipeline_contract(
    node: ast.ClassDef,
    class_defs: Mapping[str, ast.ClassDef],
    *,
    imports: Mapping[str, str],
    seen_classes: set[str],
    seen_targets: set[str],
) -> bool:
    if node.name in seen_classes:
        return False
    seen_classes.add(node.name)
    assigned = _class_assigned_names(node)
    methods = _class_method_names(node)
    if {"OPERATOR_CLS", "OPERATOR_TARGET"} & assigned:
        return True
    if "get_operator" in methods:
        return True
    if _assigns_self_operator(node):
        return True
    if "from_pretrained" in methods and ("process" in methods or "__call__" in methods):
        return True
    for base in node.bases:
        base_name = _base_name(base)
        if base_name and base_name in class_defs and _class_has_pipeline_contract(
            class_defs[base_name],
            class_defs,
            imports=imports,
            seen_classes=seen_classes,
            seen_targets=seen_targets,
        ):
            return True
        imported_target = imports.get(base_name)
        if imported_target:
            contract = _pipeline_contract_from_target_seen(imported_target, seen=seen_targets)
            if contract is not None and contract.get("has_operator"):
                return True
    return False


def _class_assigned_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
    return names


def _class_method_names(node: ast.ClassDef) -> set[str]:
    return {
        item.name
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assigns_self_operator(node: ast.ClassDef) -> bool:
    for item in ast.walk(node):
        targets: list[ast.AST] = []
        if isinstance(item, ast.Assign):
            targets.extend(item.targets)
        elif isinstance(item, ast.AnnAssign):
            targets.append(item.target)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in {"operator", "operators"}
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return True
    return False


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _imported_class_targets(tree: ast.Module, module_name: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    for item in tree.body:
        if not isinstance(item, ast.ImportFrom):
            continue
        imported_module = _resolve_import_module(module_name, item.module or "", item.level)
        if not imported_module:
            continue
        for alias in item.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            targets[local_name] = f"{imported_module}:{alias.name}"
    return targets


def _resolve_import_module(module_name: str, imported_module: str, level: int) -> str:
    if level <= 0:
        return imported_module
    parts = module_name.split(".")[:-level]
    if imported_module:
        parts.extend(imported_module.split("."))
    return ".".join(parts)


def _component_pipeline_contract(tree: ast.Module, class_name: str) -> dict[str, Any] | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name):
            continue
        if node.value.func.id not in {"_component_pipeline_class", "_official_policy_pipeline"}:
            continue
        if any(isinstance(target, ast.Name) and target.id == class_name for target in node.targets):
            return {"has_operator": True}
    return None


def _runtime_profiles_by_id() -> dict[str, object]:
    profiles: dict[str, object] = {}
    for root in (MODEL_DATA_DIR / "runtime" / "profiles",):
        for path in sorted(root.rglob("*.y*ml")) if root.exists() else ():
            if path.name.startswith("_"):
                continue
            profile = load_runtime_profile_manifest(path)
            for key in (getattr(profile, "profile_id", ""), getattr(profile, "model_id", ""), path.stem):
                text = str(key or "").strip()
                if text:
                    profiles.setdefault(text, profile)
    return profiles


def _runtime_profile_id(runtime_profile: str | None, model_id: str) -> str:
    text = str(runtime_profile or "").strip()
    return text.removeprefix("runtime-profile:") or model_id


def test_profile_backed_models_have_individual_pipeline_targets() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    for model_id in PIPELINE_MODEL_IDS:
        entry = registry.get(model_id)
        assert entry.runtime_profile == f"runtime-profile:{model_id}"
        assert entry.runner_target == UNIFIED_MODEL_RUNNER_TARGET
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.")
        assert entry.pipeline_target.split(":", maxsplit=1)[0].split(".")[2] != "runtime_profile"
        assert entry.pipeline_target.count(":") == 1
        assert entry.integration_status == "integrated"
        assert entry.runner_entry_kind == "runnable_runner"
        assert entry.is_runnable_runner_entry is True


def test_longcat_declares_real_artifact_despite_pending_runtime() -> None:
    """
    Verify LongCat no longer declares request-plan artifacts.

    Args:
        None.
    """
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    entry = registry.get("longcat-video")
    profile = load_runtime_profile("longcat-video")

    assert entry.integration_status == "integrated"
    assert entry.output_artifacts == ("generated_video",)
    assert profile.integration_status == "integrated"
    assert profile.artifact_kind == "generated_video"


def test_target_models_register_real_unified_pipeline_routes() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)
    profiles = _runtime_profiles_by_id()

    for model_id in TARGET_PIPELINE_MODEL_IDS:
        entry = registry.get(model_id)
        profile = profiles[_runtime_profile_id(entry.runtime_profile, model_id)]

        assert entry.runtime_profile is not None
        assert entry.integration_status == "integrated"
        assert entry.runner_target == UNIFIED_MODEL_RUNNER_TARGET
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.")
        assert entry.pipeline_target.count(":") == 1
        contract = _pipeline_contract_from_target(entry.pipeline_target)
        assert contract is not None
        assert contract["has_operator"]
        assert profile.artifact_kind != "metadata_profile"
        assert profile.backend_stage != "metadata_only"
        assert profile.runtime_status != "metadata_only_no_runnable_pipeline"


def test_integrated_catalog_entries_expose_pipeline_operator_contracts() -> None:
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    for entry in registry.list():
        if entry.integration_status != "integrated":
            continue

        assert entry.runner_target == UNIFIED_MODEL_RUNNER_TARGET
        assert entry.pipeline_target is not None
        assert entry.pipeline_target.startswith("worldfoundry.pipelines.")
        contract = _pipeline_contract_from_target(entry.pipeline_target)
        assert contract is not None
        assert contract["has_operator"]


def test_pipeline_runner_spec_uses_resolved_runtime_profile_id() -> None:
    for model_id, expected_profile in (
        ("cogvideox", "cogvideox-5b-t2v"),
        ("dust3r", "dust3r-base-model"),
    ):
        resolved = resolve_model_zoo_config(model_id, manifest_dir=MODEL_CATALOG_DIR, runtime={"device": "cpu"})
        spec = build_pipeline_runner_spec(resolved.config)

        assert spec.model_id == model_id
        assert spec.runtime_profile_id == expected_profile
        assert spec.model_path["profile_id"] == expected_profile
        assert spec.model_path["runtime_profile"] == expected_profile


def test_profile_backed_models_are_in_studio_and_path_a_specs() -> None:
    studio_ids = {entry.model_id for entry in discover_catalog()}
    registry = load_model_zoo_registry(MODEL_CATALOG_DIR)

    for model_id in PIPELINE_MODEL_IDS:
        entry = registry.get(model_id)
        config = WorldModelConfig(
            model_id=model_id,
            runner=entry.runner_target,
            parameters={"model_id": model_id},
            runtime={"device": "cpu"},
            metadata={
                "pipeline_target": entry.pipeline_target,
                "runtime_profile": entry.runtime_profile,
            },
        )
        spec = build_pipeline_runner_spec(config)

        if model_id in STUDIO_PIPELINE_MODEL_IDS:
            assert model_id in studio_ids
        assert spec.model_id == model_id
        assert spec.pipeline_target == entry.pipeline_target
        assert spec.device == "cpu"


def test_world_model_workspace_specs_expose_model_specific_inputs() -> None:
    models = {row["id"]: row for row in _workspace_models()}
    expected_load_fields = {
        "diamond": {"pretrained", "pretrained_game", "pretrained_dir", "num_steps_initial_collect"},
        "dino-wm": {"config", "ckpt_base_path", "model_name", "model_epoch"},
        "leworldmodel": {"config_dir", "config_name", "policy", "cache_dir"},
        "starwm": {"input_file", "mode", "api_base", "served_model_id"},
    }

    for model_id, expected_fields in expected_load_fields.items():
        model = models[model_id]
        task = model["tasks"][0]
        load_fields = {
            field["field_id"]
            for field in task["inputs"]
            if field["target"] == "load_kwargs"
        }
        call_fields = {
            field["field_id"]
            for field in task["inputs"]
            if field["target"] == "call_kwargs"
        }

        assert model["workload_type"] == "world"
        assert model["template_id"] == "interactive-world"
        assert model["default_task_id"] != "default"
        assert expected_fields <= load_fields
        assert {"plan_only", "timeout_seconds"} <= call_fields


def test_profile_backed_longcat_runner_fails_without_execute(
    tmp_path: Path,
) -> None:
    """
    Verify LongCat fails fast instead of returning request-plan artifacts.

    Args:
        tmp_path: Temporary runner output directory.
    """
    resolved = resolve_model_zoo_runner(
        "longcat-video",
        manifest_dir=MODEL_CATALOG_DIR,
        runtime={"device": "cpu"},
    )

    results = resolved.runner.generate(
        [
            GenerationRequest(
                sample_id="longcat-video-smoke",
                task_name="longcat:smoke",
                inputs={"prompt": "walk forward through a blocky landscape"},
                generation_kwargs={"output_dir": str(tmp_path / "longcat-video")},
                output_schema={"generated_video": {"kind": "video"}, "generated_world": {"kind": "video"}},
            )
        ]
    )

    assert results[0].status == "failed"
    assert results[0].artifacts == {}
    assert (
        results[0].error
        == "RuntimeError: LongCat-Video requires execute=True; preflight artifacts are no longer emitted."
    )


def test_leworldmodel_uses_json_sidecar_for_workspace_video_output() -> None:
    from worldfoundry.synthesis.visual_generation.le_wm.worldfoundry_runtime import build_command

    command = build_command(
        {
            "python": "python",
            "entrypoint": "infer.py",
            "output_path": "/tmp/worldfoundry/leworldmodel.mp4",
            "output_dir": "/tmp/worldfoundry",
            "device": "cuda",
            "options": {},
        }
    )

    artifact_index = command.index("--artifact-path") + 1
    assert command[artifact_index].endswith("/leworldmodel.result.json")


def test_world_model_runtime_prefers_expected_video_artifact(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.runtime_manifest import WorldModelRuntimeSynthesis

    target = tmp_path / "leworldmodel.mp4"
    result_json = tmp_path / "leworldmodel.result.json"
    video = tmp_path / "env_0.mp4"
    result_json.write_text('{"status":"succeeded"}\n', encoding="utf-8")
    video.write_bytes(b"video")

    resolved = WorldModelRuntimeSynthesis._resolve_produced_artifact(target)

    assert resolved == video
