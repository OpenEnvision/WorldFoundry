"""Changes must never silently escape the affected-model regression plan."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "manual"
spec = importlib.util.spec_from_file_location("impact_under_test", TOOLS / "geometry_regression_impact.py")
assert spec is not None and spec.loader is not None
impact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impact)


@pytest.mark.parametrize(
    "path",
    [
        "tests/synthesis/test_matrix_game_2_checkpoint_conditioned_trajectory.py",
        "tests/synthesis/mg2_historical_reference.py",
    ],
)
def test_mg2_numerical_contract_changes_require_real_model_replay(path):
    root = TOOLS.parents[1]
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        root,
        [path],
    )
    assert plan["status"] == "planned"
    assert plan["selected_cases"] == [
        "matrix-game2-controls-short",
        "matrix-game-2-universal-ref-image-seed42-15f-checkpoint-replay",
    ]
    assert not plan["uncovered_paths"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "component,paths,expected_cases",
    [
        (
            "worldplay2",
            (
                "recipes/worldplay2.py",
                "runners/compressed_memory.py",
                "models/denoisers/worldplay2.py",
                "models/networks/wan/variants/worldplay2/model.py",
                "models/encoders/wan/variants/worldplay2.py",
                "models/initializers/wan/worldplay2.py",
                "schedulers/fixed_pdd.py",
            ),
            {
                "worldplay2-controls-short",
                "worldplay2-fast-controls-short",
                "worldplay2-ar-controls-short",
                "worldplay2-bi-controls-short",
            },
        ),
        (
            "helixworld",
            (
                "recipes/helixworld.py",
                "runners/joint_chunked.py",
                "models/denoisers/helixworld.py",
                "models/networks/ltx/variants/helixworld.py",
                "models/encoders/ltx/variants/helixworld/conditioning.py",
                "models/encoders/ltx/variants/helixworld/controls.py",
                "schedulers/clean_sample.py",
            ),
            {"helixworld-controls-short"},
        ),
    ],
)
def test_native_role_paths_keep_affected_checkpoint_replays(
    component: str,
    paths: tuple[str, ...],
    expected_cases: set[str],
) -> None:
    """Canonical role ownership must retain the configured real-checkpoint gates."""
    _, policy = impact.load_definitions(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
    )
    root = TOOLS.parents[1]
    for relative in paths:
        path = "worldfoundry/base_models/diffusion_model/" + relative
        assert (root / path).is_file()
        assert impact.matches(path, policy["components"][component]), path
    assert {case for case, owners in policy["cases"].items() if component in owners} == expected_cases


@pytest.mark.unit
@pytest.mark.parametrize(
    "path,expected_components,expected_cases",
    [
        (
            "worldfoundry/operators/worldfm_operator.py",
            {"worldfm"},
            {"worldfm"},
        ),
        (
            "worldfoundry/pipelines/worldfm/pipeline_worldfm.py",
            {"worldfm"},
            {"worldfm"},
        ),
        (
            "worldfoundry/pipelines/lingbot_world/distributed.py",
            {"lingbot-world", "lingbot-world-v2"},
            {
                "lingbot-world-controls-short",
                "lingbot-world-act-controls-short",
                "lingbot-world-v2-causal-short",
                "lingbot-world-v2-1p3b-causal-short",
            },
        ),
    ],
)
def test_model_owned_input_and_loading_paths_keep_checkpoint_replays(
    path: str,
    expected_components: set[str],
    expected_cases: set[str],
) -> None:
    _, policy = impact.load_definitions(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
    )
    assert (TOOLS.parents[1] / path).is_file()
    owners = {owner for owner, paths in policy["components"].items() if impact.matches(path, paths)}
    assert expected_components <= owners
    assert {case for case, components in policy["cases"].items() if owners.intersection(components)} == expected_cases


DECLARED_REPLAY_OWNERS = {
    "cogact": (
        (
            "worldfoundry/base_models/llm_mllm_core/action/cogact.py",
            "worldfoundry/base_models/llm_mllm_core/action/cogact_diffusion.py",
        ),
        {"cogact-action-short", "db-cogact-action-short"},
    ),
    "lingbot-vla": (
        (
            "worldfoundry/base_models/llm_mllm_core/action/lingbot/__init__.py",
            "worldfoundry/base_models/llm_mllm_core/action/lingbot/modeling.py",
            "worldfoundry/base_models/llm_mllm_core/action/lingbot/preprocessing.py",
        ),
        {"lingbot-vla-action-short", "lingbot-vla-v2-action-short"},
    ),
    "go1-rdt-embedder": (
        ("worldfoundry/base_models/llm_mllm_core/action/rdt.py",),
        {"go1-action-short"},
    ),
    "tinyvla-transformer": (
        ("worldfoundry/base_models/llm_mllm_core/action/transformer.py",),
        {"tinyvla-action-short"},
    ),
    "da3-dinov2": (
        (
            "worldfoundry/base_models/perception_core/general_perception/dinov2/variants/depth_anything_v3/vision_transformer.py",
            "worldfoundry/base_models/perception_core/general_perception/dinov2/variants/depth_anything_v3/layers/attention.py",
        ),
        {
            "da3-large-1.1",
            "da3metric-large",
            "da3nested-giant-large-1.1",
            "lyra2-forward",
            "alayaworld-spatial-two-chunks-short",
        },
    ),
    "krea-worker": (
        (
            "worldfoundry/synthesis/visual_generation/krea_realtime/worldfoundry_runner.py",
            "worldfoundry/synthesis/visual_generation/krea_realtime/krea_runtime/sample.py",
            "worldfoundry/synthesis/visual_generation/krea_realtime/krea_runtime/release_server.py",
            "worldfoundry/synthesis/visual_generation/krea_realtime/krea_runtime/pipeline/causal_inference.py",
            "worldfoundry/synthesis/visual_generation/krea_realtime/krea_runtime/runtime_utils/memory.py",
            "worldfoundry/synthesis/visual_generation/krea_realtime/krea_runtime/runtime_utils/vae_block3.py",
            "worldfoundry/synthesis/visual_generation/krea_realtime/krea_runtime/utils/wan_wrapper.py",
        ),
        {"krea-realtime-two-blocks-short"},
    ),
    "uniworld-entrypoint": (
        ("worldfoundry/pipelines/uniworld_view/upstream_entry.py",),
        {"uniworld-view-single-view-short"},
    ),
    "wonderworld-repvit": (
        ("worldfoundry/base_models/perception_core/segment/sam_v1/modeling/transformer.py",),
        {"wonderworld-scene-short"},
    ),
    "spatia-mapanything": (
        (
            "worldfoundry/base_models/three_dimensions/general_3d/mapanything/mapanything/models/mapanything/model.py",
            "worldfoundry/base_models/three_dimensions/general_3d/mapanything/mapanything/utils/geometry.py",
            "worldfoundry/base_models/three_dimensions/general_3d/mapanything/mapanything/utils/inference.py",
            "worldfoundry/base_models/three_dimensions/general_3d/mapanything/mapanything/utils/image.py",
            "worldfoundry/base_models/three_dimensions/general_3d/mapanything/mapanything/utils/multiview_confidence.py",
            "worldfoundry/base_models/three_dimensions/general_3d/mapanything/mapanything/utils/misc.py",
        ),
        {"spatia-checkpoint-replay"},
    ),
    "spatia-uniception": (
        (
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/encoders/dense_rep_encoder.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/encoders/dinov2.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/encoders/global_rep_encoder.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/info_sharing/alternating_attention_transformer.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/info_sharing/cross_attention_transformer.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/info_sharing/global_attention_transformer.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/prediction_heads/__init__.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/prediction_heads/adaptors.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/prediction_heads/dpt.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/prediction_heads/linear.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/prediction_heads/mlp_head.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/prediction_heads/pose_head.py",
            "worldfoundry/base_models/perception_core/general_perception/uniception/uniception/models/utils/transformer_blocks.py",
        ),
        {"spatia-checkpoint-replay"},
    ),
    "evoke-vigeo": (
        (
            "worldfoundry/base_models/three_dimensions/depth/vigeo/layers/mlp.py",
            "worldfoundry/base_models/three_dimensions/depth/vigeo/layers/patch_embed.py",
            "worldfoundry/base_models/three_dimensions/depth/vigeo/layers/rope.py",
            "worldfoundry/base_models/three_dimensions/depth/vigeo/layers/swiglu_ffn.py",
        ),
        {"evoke-checkpoint-replay"},
    ),
}

INFERENCE_GUIDE_PATHS = (
    "skills/index.md",
    "skills/worldfoundry-studio-serving/SKILL.md",
    "skills/worldfoundry-studio-serving/references/realtime-contract.md",
    "worldfoundry/base_models/README.md",
    "worldfoundry/base_models/three_dimensions/README.md",
)

EVALUATOR_CPU_PATHS = (
    "worldfoundry/base_models/perception_core/frame_interpolation/amt/motion_smoothness.py",
    "worldfoundry/base_models/perception_core/frame_interpolation/amt/networks/AMT-S.py",
    "worldfoundry/base_models/perception_core/frame_interpolation/amt/networks/blocks/feat_enc.py",
    "worldfoundry/base_models/perception_core/optical_flow/flowformerplusplus/core/FlowFormer/PerCostFormer3/cnn.py",
)

TRACKING_SCORER_PATHS = (
    "worldfoundry/base_models/perception_core/tracking/cotracker/models/core/cotracker/blocks.py",
    "worldfoundry/base_models/perception_core/tracking/dot/models/shelf/"
    "cotracker2_utils/models/core/cotracker/blocks.py",
    "worldfoundry/base_models/perception_core/tracking/dot/models/shelf/"
    "cotracker3_utils/models/core/cotracker/blocks.py",
)

UNREPRESENTED_REPLAY_PATHS = (
    "worldfoundry/base_models/three_dimensions/point_clouds/pi3_source.py",
    "worldfoundry/base_models/three_dimensions/depth/vigeo/layers/prior_dinov2.py",
    "worldfoundry/base_models/three_dimensions/depth/vigeo/layers/drop_path.py",
    "worldfoundry/operators/wow_operator.py",
    "worldfoundry/synthesis/action_generation/rdt_1b/modeling.py",
    "worldfoundry/base_models/perception_core/general_perception/dinov2/variants/vipe_dav3/vision_transformer.py",
    "worldfoundry/data/models/runtime/environments/three_d/geometry-priors.yaml",
    "worldfoundry/data/models/runtime/environments/video/studio_verified_unified.yaml",
)


@pytest.fixture(scope="module")
def declared_replay_ownership_plan() -> dict:
    paths = [path for owned, _ in DECLARED_REPLAY_OWNERS.values() for path in owned]
    return impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        TOOLS.parents[1],
        [*paths, *INFERENCE_GUIDE_PATHS, *UNREPRESENTED_REPLAY_PATHS, *EVALUATOR_CPU_PATHS, *TRACKING_SCORER_PATHS],
    )


@pytest.mark.unit
@pytest.mark.parametrize("paths,expected_cases", DECLARED_REPLAY_OWNERS.values(), ids=DECLARED_REPLAY_OWNERS)
def test_configured_and_historical_dependencies_have_exact_replay_owners(
    declared_replay_ownership_plan: dict,
    paths: tuple[str, ...],
    expected_cases: set[str],
) -> None:
    plan = declared_replay_ownership_plan
    assert not plan["graph_errors"]
    for path in paths:
        assert path not in plan["uncovered_paths"]
        owners = {
            case
            for case, reasons in plan["reasons"].items()
            if any(reason.get("path") == path and "declared_component" in reason["kind"] for reason in reasons)
        }
        assert owners == expected_cases


@pytest.mark.unit
def test_inference_guides_are_documentation_not_replay_dependencies(declared_replay_ownership_plan: dict) -> None:
    plan = declared_replay_ownership_plan
    assert set(INFERENCE_GUIDE_PATHS) <= set(plan["ignored_paths"])
    assert not set(INFERENCE_GUIDE_PATHS).intersection(plan["uncovered_paths"])
    assert all(
        reason.get("path") not in INFERENCE_GUIDE_PATHS for reasons in plan["reasons"].values() for reason in reasons
    )


@pytest.mark.unit
def test_exact_evaluator_changes_require_cpu_contracts(declared_replay_ownership_plan: dict) -> None:
    plan = declared_replay_ownership_plan
    assert set(EVALUATOR_CPU_PATHS).isdisjoint(plan["uncovered_paths"])
    proofs = {item["path"]: item["required_cpu_contracts"] for item in plan["cpu_only_changes"]}
    for path in EVALUATOR_CPU_PATHS:
        assert set(proofs[path]) == {
            "tests/base_models/perception_core/optical_flow/test_raft_features.py",
            "tests/evaluation/execution/test_vbench_sampling.py",
        }


@pytest.mark.unit
def test_tracking_scorer_changes_require_cpu_contracts(declared_replay_ownership_plan: dict) -> None:
    plan = declared_replay_ownership_plan
    assert set(TRACKING_SCORER_PATHS).isdisjoint(plan["uncovered_paths"])
    proofs = {item["path"]: item["required_cpu_contracts"] for item in plan["cpu_only_changes"]}
    for path in TRACKING_SCORER_PATHS:
        assert proofs[path] == ["tests/evaluation/execution/test_vbench_sampling.py"]


@pytest.mark.unit
def test_unrepresented_modes_and_environment_variants_remain_uncovered(declared_replay_ownership_plan: dict) -> None:
    plan = declared_replay_ownership_plan
    assert plan["status"] == "uncovered"
    assert set(UNREPRESENTED_REPLAY_PATHS) <= set(plan["uncovered_paths"])
    assert not set(UNREPRESENTED_REPLAY_PATHS).intersection(plan["ignored_paths"])
    assert all(
        reason.get("path") not in UNREPRESENTED_REPLAY_PATHS or "declared_component" not in reason["kind"]
        for reasons in plan["reasons"].values()
        for reason in reasons
    )


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "worldfoundry/__init__.py": "",
        "worldfoundry/models/__init__.py": "from . import parent_side_effect\n",
        "worldfoundry/models/parent_side_effect.py": "",
        "worldfoundry/models/a.py": 'from . import conditional\nif False:\n    import worldfoundry.models.hidden\nBACKEND = "worldfoundry.models.lazy:Model"\n',
        "worldfoundry/models/b.py": "",
        "worldfoundry/models/conditional.py": "from ..utils import math\n",
        "worldfoundry/models/hidden.py": "",
        "worldfoundry/models/lazy.py": "",
        "worldfoundry/utils/math.py": "",
        "worldfoundry/dynamic/unused.py": "",
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    matrix = root / "matrix.json"
    matrix.write_text(
        json.dumps({name: {"id": name, "target": f"worldfoundry.models.{name}:Model"} for name in ("a", "b")})
    )
    dependencies = root / "dependencies.json"
    dependencies.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_paths": ["worldfoundry/core/**", "matrix.json", "dependencies.json"],
                "ignored_paths": ["docs/**", "tests/**"],
                "components": {"a": ["worldfoundry/dynamic/**"], "b": ["worldfoundry/models/b.py"]},
                "cases": {"a": ["a"], "b": ["b"]},
            }
        )
    )
    git(root, "init", "-q")
    git(root, "config", "user.name", "Regression test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return root, matrix, dependencies


@pytest.mark.parametrize(
    "path",
    [
        "worldfoundry/models/a.py",
        "worldfoundry/models/conditional.py",
        "worldfoundry/models/hidden.py",
        "worldfoundry/models/lazy.py",
        "worldfoundry/utils/math.py",
        "worldfoundry/dynamic/unused.py",
    ],
)
def test_conditional_relative_lazy_and_declared_dynamic_dependencies_select_model(project, path):
    root, matrix, policy = project
    plan = impact.select_cases(matrix, policy, root, [path])
    assert plan["status"] == "planned"
    assert plan["selected_cases"] == ["a"]


def test_evaluation_only_paths_yield_to_inference_imports(project):
    root, matrix, policy = project
    path = "worldfoundry/models/scorer.py"
    (root / path).write_text("")
    git(root, "add", path)
    declarations = json.loads(policy.read_text())
    declarations["evaluation_only_paths"] = [path]
    policy.write_text(json.dumps(declarations))
    evaluation = impact.select_cases(matrix, policy, root, [path])
    assert evaluation["status"] == "no_inference_changes"
    assert evaluation["ignored_paths"] == [path]
    assert impact.select_cases(matrix, policy, root, ["worldfoundry/models/other.py"])["status"] == "uncovered"

    (root / "worldfoundry/models/b.py").write_text("from . import scorer\n")
    inference = impact.select_cases(matrix, policy, root, [path])
    assert inference["ignored_paths"] == []
    assert inference["selected_cases"] == ["b"]


@pytest.mark.parametrize(
    "path",
    [
        "worldfoundry/models/__init__.py",
        "worldfoundry/models/parent_side_effect.py",
        "worldfoundry/core/new_backend.py",
        "matrix.json",
    ],
)
def test_shared_and_parent_package_initializers_select_every_case(project, path):
    root, matrix, policy = project
    assert impact.select_cases(matrix, policy, root, [path])["selected_cases"] == ["a", "b"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("load", {"checkpoint_path": "/weights/model.safetensors"}),
        ("call", {"num_frames": 17}),
        ("sequence", [{"name": "a", "call": {"seed": 42}, "outputs": ["video"]}]),
        ("assets", {"input": "/inputs/image.png"}),
        ("array_contracts", {"video": {"shape": [1, 3, 17, 256, 448]}}),
        ("scope", "short real checkpoint replay"),
    ],
)
def test_committed_case_edit_requires_its_replay_and_preserves_other_cases(project, field, value):
    root, matrix, policy = project
    base = git(root, "rev-parse", "HEAD")
    cases = json.loads(matrix.read_text())
    cases["a"][field] = value
    matrix.write_text(json.dumps(cases))
    git(root, "add", "matrix.json")
    git(root, "commit", "-qm", "change one replay recipe")
    plan = impact.select_cases(matrix, policy, root, ["matrix.json"], base=base)
    assert plan["status"] == "planned"
    assert plan["selected_cases"] == ["a"]


def test_committed_matrix_formatting_has_no_numerical_changes(project):
    root, matrix, policy = project
    base = git(root, "rev-parse", "HEAD")
    matrix.write_text(json.dumps(json.loads(matrix.read_text()), sort_keys=True, indent=4))
    git(root, "add", "matrix.json")
    git(root, "commit", "-qm", "format recipes")
    plan = impact.select_cases(matrix, policy, root, ["matrix.json"], base=base)
    assert plan["status"] == "no_inference_changes"
    assert plan["selected_cases"] == []


@pytest.mark.parametrize("old_value,new_value", [(True, 1), (1, 1.0), (False, 0)])
def test_case_diff_preserves_json_value_types(project, old_value, new_value):
    root, matrix, policy = project
    cases = json.loads(matrix.read_text())
    cases["a"]["deterministic"] = old_value
    matrix.write_text(json.dumps(cases))
    git(root, "add", "matrix.json")
    git(root, "commit", "-qm", "typed case")
    base = git(root, "rev-parse", "HEAD")
    cases["a"]["deterministic"] = new_value
    matrix.write_text(json.dumps(cases))
    git(root, "add", "matrix.json")
    git(root, "commit", "-qm", "change JSON value type")
    assert impact.select_cases(matrix, policy, root, ["matrix.json"], base=base)["selected_cases"] == ["a"]


@pytest.mark.parametrize("failure", ["missing_base", "dirty_matrix", "omitted_change", "duplicate_base_key"])
def test_unproven_case_edit_keeps_conservative_full_replay(project, failure):
    root, matrix, policy = project
    if failure == "duplicate_base_key":
        matrix.write_text(matrix.read_text().replace('"a": {', '"a": {}, "a": {', 1))
        git(root, "add", "matrix.json")
        git(root, "commit", "-qm", "ambiguous historical definition")
    base = git(root, "rev-parse", "HEAD")
    cases = json.loads(matrix.read_text())
    cases["a"]["call"] = {"seed": 43}
    matrix.write_text(json.dumps(cases))
    if failure == "omitted_change":
        (root / "worldfoundry/models/b.py").write_text("CHANGED = True\n")
        git(root, "add", "worldfoundry/models/b.py")
    git(root, "add", "matrix.json")
    git(root, "commit", "-qm", "edit recipe")
    if failure == "dirty_matrix":
        cases["b"]["call"] = {"seed": 44}
        matrix.write_text(json.dumps(cases))
    if failure == "missing_base":
        base = "0" * 40
    assert impact.select_cases(matrix, policy, root, ["matrix.json"], base=base)["selected_cases"] == ["a", "b"]


def test_case_edit_cannot_hide_an_independent_model_change(project):
    root, matrix, policy = project
    base = git(root, "rev-parse", "HEAD")
    cases = json.loads(matrix.read_text())
    cases["a"]["call"] = {"seed": 43}
    matrix.write_text(json.dumps(cases))
    (root / "worldfoundry/models/b.py").write_text("CHANGED = True\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "recipe and independent model")
    changes = impact.changed_paths(root, base, "HEAD")
    assert impact.select_cases(matrix, policy, root, changes, base=base)["selected_cases"] == ["a", "b"]


def test_docs_need_cpu_gates_without_claiming_inference_passed(project):
    root, matrix, policy = project
    plan = impact.select_cases(matrix, policy, root, ["docs/validation.md"])
    assert plan["status"] == "no_inference_changes"
    assert plan["selected_cases"] == []
    assert plan["required_checks"] == ["public-cpu", "inference-tensors"]


def test_unintegrated_code_selects_full_matrix_and_remains_uncovered(project):
    root, matrix, policy = project
    plan = impact.select_cases(matrix, policy, root, ["worldfoundry/new_model/runtime.py"])
    assert plan["status"] == "uncovered"
    assert plan["uncovered_paths"] == ["worldfoundry/new_model/runtime.py"]
    assert plan["selected_cases"] == ["a", "b"]


def test_bad_import_graph_is_not_green_even_for_docs_change(project):
    root, matrix, policy = project
    (root / "worldfoundry/models/a.py").write_text("invalid python: ?")
    plan = impact.select_cases(matrix, policy, root, ["docs/update.md"])
    assert plan["status"] == "uncovered"
    assert plan["selected_cases"] == ["a", "b"]
    assert "a" in plan["graph_errors"]


def test_recorded_imports_add_selection_without_replacing_declarations(project, tmp_path):
    root, matrix, policy = project
    evidence = tmp_path / "trace" / "b"
    evidence.mkdir(parents=True)
    (evidence / "manifest.json").write_text(
        json.dumps(
            {"status": "passed", "case": {"id": "b"}, "source_hashes": {"worldfoundry/dynamic/unused.py": "sha"}}
        )
    )
    plan = impact.select_cases(matrix, policy, root, ["worldfoundry/dynamic/unused.py"], evidence=evidence.parent)
    assert plan["selected_cases"] == ["a", "b"]


def test_rename_and_delete_keep_old_dependency_paths(project):
    root, _, _ = project
    base = git(root, "rev-parse", "HEAD")
    git(root, "mv", "worldfoundry/models/hidden.py", "worldfoundry/models/renamed module.py")
    git(root, "rm", "worldfoundry/models/b.py")
    git(root, "commit", "-qm", "rename and delete")
    assert impact.changed_paths(root, base, "HEAD") == [
        "worldfoundry/models/b.py",
        "worldfoundry/models/hidden.py",
        "worldfoundry/models/renamed module.py",
    ]


@pytest.mark.parametrize("value", ["", ".", "./module.py", "../module.py", "/tmp/module.py", "a\\b.py", "a//b.py"])
def test_invalid_paths_cannot_escape_repository(value):
    with pytest.raises(ValueError):
        impact.relative_path(value)


def test_new_case_requires_explicit_dependency_declaration(project):
    _, matrix, policy = project
    cases = json.loads(matrix.read_text())
    cases["new"] = {"id": "new", "target": "worldfoundry.models.a:Model"}
    matrix.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="explicit dependency"):
        impact.load_definitions(matrix, policy)


def test_snapshot_inside_other_git_checkout_uses_its_own_files(project):
    root, _, _ = project
    snapshot = root / "tmp" / "snapshot" / "worldfoundry"
    snapshot.mkdir(parents=True)
    (snapshot / "unique.py").write_text("")
    assert impact.ImportGraph(snapshot.parent).closure("worldfoundry.unique:Model") == {"worldfoundry/unique.py"}


def test_public_matrix_has_complete_declarations_and_known_hy2_path():
    root = TOOLS.parents[1]
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        root,
        ["worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/runtime.py"],
    )
    assert not plan["graph_errors"]
    assert plan["status"] == "planned"
    assert "hyworldmirror-2" in plan["selected_cases"]


@pytest.mark.parametrize(
    "path,case_id",
    [
        ("worldfoundry/pipelines/cosmos/pipeline_cosmos_transfer2p5.py", "cosmos-transfer25-2b-edge-short"),
        ("worldfoundry/pipelines/gamma_world/pipeline_gamma_world.py", "gamma-world-causal-few-step-189f"),
        ("worldfoundry/synthesis/visual_generation/open_oasis/utils.py", "oasis500m-video-short"),
    ],
)
def test_public_world_model_changes_have_their_own_replay_case(path, case_id):
    root = TOOLS.parents[1]
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        root,
        [path],
    )
    assert plan["status"] == "planned"
    assert not plan["uncovered_paths"] and not plan["graph_errors"]
    assert case_id in plan["selected_cases"]
    assert any(reason["path"] == path and reason.get("components") for reason in plan["reasons"][case_id])


def test_public_matrix_still_rejects_an_unintegrated_world_model():
    root = TOOLS.parents[1]
    path = "worldfoundry/pipelines/unintegrated_world/pipeline_world.py"
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        root,
        [path],
    )
    assert plan["status"] == "uncovered"
    assert plan["uncovered_paths"] == [path]


REALTIME_CONTRACTS = "worldfoundry/core/execution/realtime/contracts.py"
REALTIME_FLOW_SAMPLING = "worldfoundry/core/execution/realtime/flow_sampling.py"
STUDIO_REALTIME_WEBRTC = "worldfoundry/studio/serving/realtime/webrtc.py"
MEDIA_PROCESSING_RTX = "worldfoundry/core/media/processing/rtx.py"
CORE_ATTENTION = "worldfoundry/core/attention/__init__.py"
NEW_REALTIME_MODULE = "worldfoundry/core/execution/realtime/unimported_session.py"
NEW_CORE_SUBPACKAGE = "worldfoundry/core/unlisted_subsystem/module.py"


@pytest.fixture(scope="module")
def realtime_layer_plan() -> dict:
    """Plan every realtime-layer probe path together; reasons stay attributed per path."""
    return impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        TOOLS.parents[1],
        [
            REALTIME_CONTRACTS,
            REALTIME_FLOW_SAMPLING,
            STUDIO_REALTIME_WEBRTC,
            MEDIA_PROCESSING_RTX,
            CORE_ATTENTION,
            NEW_REALTIME_MODULE,
            NEW_CORE_SUBPACKAGE,
        ],
    )


def cases_selected_by(plan: dict, path: str) -> dict[str, list[dict]]:
    """Return the selection reasons recorded for one changed path, keyed by case."""
    selected = {}
    for case, reasons in plan["reasons"].items():
        matched = [reason for reason in reasons if reason.get("path") == path]
        if matched:
            selected[case] = matched
    return selected


@pytest.mark.parametrize("path", [REALTIME_CONTRACTS, MEDIA_PROCESSING_RTX])
def test_realtime_layer_paths_select_cases_through_their_dependencies(realtime_layer_plan: dict, path: str) -> None:
    root = TOOLS.parents[1]
    assert (root / path).is_file()
    assert not realtime_layer_plan["graph_errors"]
    selected = cases_selected_by(realtime_layer_plan, path)
    assert selected
    for reasons in selected.values():
        for reason in reasons:
            assert reason["kind"] != "shared_component_or_case_matrix"
            assert set(reason["kind"].split("+")) <= {"declared_component", "static_import", "recorded_import"}


def test_studio_realtime_serving_changes_select_their_importers_or_cpu_contracts(realtime_layer_plan: dict) -> None:
    """A serving file selects exactly the cases that import it; with no importer its CPU contracts apply."""
    root = TOOLS.parents[1]
    assert (root / STUDIO_REALTIME_WEBRTC).is_file()
    cases, _ = impact.load_definitions(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
    )
    graph = impact.ImportGraph(root)
    importers = {name for name, case in cases.items() if STUDIO_REALTIME_WEBRTC in graph.closure(case["target"])}
    selected = cases_selected_by(realtime_layer_plan, STUDIO_REALTIME_WEBRTC)
    assert set(selected) == importers
    for reasons in selected.values():
        assert [reason["kind"] for reason in reasons] == ["static_import"]
    assert STUDIO_REALTIME_WEBRTC not in realtime_layer_plan["uncovered_paths"]
    contracts = [
        item["required_cpu_contracts"]
        for item in realtime_layer_plan["cpu_only_changes"]
        if item["path"] == STUDIO_REALTIME_WEBRTC
    ]
    declared = [
        "tests/studio_visualization/test_realtime_delivery.py",
        "tests/studio_visualization/test_realtime_presentation.py",
    ]
    assert contracts == ([] if importers else [declared])
    if not importers:
        assert set(declared) <= set(realtime_layer_plan["required_cpu_contracts"])


def test_realtime_contracts_select_their_offline_importers_only(realtime_layer_plan: dict) -> None:
    selected = cases_selected_by(realtime_layer_plan, REALTIME_CONTRACTS)
    for case in ("self-forcing-t2v-short", "matrix-game2-controls-short"):
        assert any("static_import" in reason["kind"] for reason in selected[case])
    assert set(selected) < set(realtime_layer_plan["reasons"])


def test_realtime_flow_sampling_selects_only_lingbot_cases(realtime_layer_plan: dict) -> None:
    _, policy = impact.load_definitions(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
    )
    lingbot_cases = {
        case
        for case, components in policy["cases"].items()
        if {"lingbot-world", "lingbot-world-v2"}.intersection(components)
    }
    selected = cases_selected_by(realtime_layer_plan, REALTIME_FLOW_SAMPLING)
    assert selected
    assert set(selected) <= lingbot_cases


def test_other_core_subtrees_remain_shared_by_every_case(realtime_layer_plan: dict) -> None:
    selected = cases_selected_by(realtime_layer_plan, CORE_ATTENTION)
    assert set(selected) == set(realtime_layer_plan["reasons"])
    assert len(selected) == len(json.loads((TOOLS / "geometry_regression_cases.json").read_text()))
    for reasons in selected.values():
        assert [reason["kind"] for reason in reasons] == ["shared_component_or_case_matrix"]


def test_base_model_geometry_registry_data_selects_checkpoint_replays() -> None:
    matrix = TOOLS / "geometry_regression_cases.json"
    paths = ["worldfoundry/data/base_models/capabilities/geometry.yaml", "worldfoundry/data/base_models/stacks.yaml"]
    plan = impact.select_cases(matrix, TOOLS / "geometry_regression_dependencies.json", TOOLS.parents[1], paths)

    assert plan["status"] == "planned"
    assert not plan["uncovered_paths"]
    assert set(plan["selected_cases"]) == set(json.loads(matrix.read_text()))
    for reasons in plan["reasons"].values():
        assert reasons == [{"path": path, "kind": "shared_component_or_case_matrix"} for path in paths]


def test_unimported_realtime_or_unlisted_core_files_keep_conservative_fallback(realtime_layer_plan: dict) -> None:
    assert realtime_layer_plan["status"] == "uncovered"
    assert realtime_layer_plan["uncovered_paths"] == sorted([NEW_REALTIME_MODULE, NEW_CORE_SUBPACKAGE])
    for reasons in realtime_layer_plan["reasons"].values():
        fallback = [reason for reason in reasons if reason["kind"] == "conservative_fallback"]
        assert fallback == [
            {
                "kind": "conservative_fallback",
                "paths": sorted([NEW_REALTIME_MODULE, NEW_CORE_SUBPACKAGE]),
                "graph_errors": [],
            }
        ]


VERIFICATION_SUITE = "tests/manual/geometry_regression_suite.py"
VERIFICATION_IMPACT = "tests/manual/geometry_regression_impact.py"
MG2_CASE = "matrix-game2-controls-short"
VERIFICATION_CONTRACTS = [
    "tests/runtime/test_geometry_regression_impact.py",
    "tests/runtime/test_geometry_regression_suite.py",
]


@pytest.fixture
def verification_project(project):
    root, _, _ = project
    files = {
        VERIFICATION_SUITE: "# Original replay orchestration\n",
        VERIFICATION_IMPACT: "# Original affected-case planning\n",
        "tests/manual/geometry_regression.py": "# Original numerical exporter\n",
        VERIFICATION_CONTRACTS[0]: "def test_complete_diff_proof(): pass\n",
        VERIFICATION_CONTRACTS[1]: "def test_replay_snapshot_safety(): pass\n",
        "docs/orchestration.md": "original guide\n",
        ".github/workflows/ci.yml": "name: validation\n",
        "Makefile": "test:\n\tpython -m pytest\n",
        ".gitignore": "worldfoundry/ignored_shadow.py\nignored_shadow.py\n",
        "MANIFEST.in": "include LICENSE\n",
        "LICENSE": "original project license\n",
        "requirements/model.txt": "original-runtime==1\n",
        "envs/model.yaml": "runtime: original\n",
        "assets/reference.bin": "original input bytes\n",
        "thirdparty/vendor/source.py": "# Original external runtime\n",
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    matrix = root / "tests/manual/geometry_regression_cases.json"
    cases = {name: {"id": name, "target": f"worldfoundry.models.{name}:Model"} for name in ("a", "b")}
    cases[MG2_CASE] = {"id": MG2_CASE, "target": "worldfoundry.models.a:Model"}
    matrix.write_text(json.dumps(cases))
    dependencies = root / "tests/manual/geometry_regression_dependencies.json"
    dependencies.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_paths": [
                    "worldfoundry/core/**",
                    "tests/manual/geometry_regression*.py",
                    str(matrix.relative_to(root)),
                    str(dependencies.relative_to(root)),
                    "MANIFEST.in",
                    "requirements/**",
                    "envs/**",
                    "assets/**",
                    "thirdparty/**",
                    "LICENSE",
                ],
                "ignored_paths": ["tests/**", "docs/**", ".github/**", "Makefile"],
                "components": {"a": ["worldfoundry/models/a.py"], "b": ["worldfoundry/models/b.py"]},
                "cases": {"a": ["a"], "b": ["b"], MG2_CASE: ["a"]},
            }
        )
    )
    git(root, "add", ".")
    git(root, "commit", "-qm", "complete committed verification fixture")
    return root, matrix, dependencies, git(root, "rev-parse", "HEAD")


def commit_verification_change(project, *, tools=(VERIFICATION_SUITE,), extra=None):
    root, _, _, base = project
    for name in tools:
        (root / name).write_text("# Changed verification orchestration\n")
    for name, value in (extra or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    git(root, "add", ".")
    git(root, "commit", "-qm", "verification changes")
    head = git(root, "rev-parse", "HEAD")
    return head, impact.changed_paths(root, base, head)


def verification_proofs(project, *, head, changes, base=None):
    root, matrix, dependencies, original = project
    return impact.verification_tooling_proofs(
        root, matrix, dependencies, changes, base=original if base is None else base, head=head
    )


def assert_conservative_verification_fallback(project, *, head, changes, base=None):
    root, matrix, dependencies, original = project
    reference = original if base is None else base
    assert impact.verification_tooling_proofs(root, matrix, dependencies, changes, base=reference, head=head) == []
    plan = impact.select_cases(matrix, dependencies, root, changes, base=reference, head=head)
    assert set(plan["selected_cases"]) == {"a", "b", MG2_CASE}
    assert not plan["cpu_only_changes"]
    assert not plan["required_cpu_contracts"]
    return plan


@pytest.mark.parametrize(
    "tool,contract", [(VERIFICATION_SUITE, VERIFICATION_CONTRACTS[1]), (VERIFICATION_IMPACT, VERIFICATION_CONTRACTS[0])]
)
def test_committed_verification_tool_change_requires_cpu_contract_and_real_mg2_replay(
    verification_project, tool, contract
):
    root, matrix, dependencies, base = verification_project
    head, changes = commit_verification_change(verification_project, tools=(tool,))
    proofs = verification_proofs(verification_project, head=head, changes=changes)
    assert len(proofs) == 1
    proof = proofs[0]
    assert proof["path"] == tool
    assert proof["base_revision"] == base and proof["head_revision"] == head
    assert proof["source_sha256"] == hashlib.sha256((root / tool).read_bytes()).hexdigest()
    assert proof["required_cpu_contracts"] == [contract]
    assert len(proof["immutable_runtime_tree_sha256"]) == 64
    assert int(proof["immutable_runtime_tree_sha256"], 16) >= 0
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert plan["status"] == "planned" and plan["selected_cases"] == [MG2_CASE]
    assert plan["cpu_only_changes"] == proofs
    assert plan["required_cpu_contracts"] == [contract]
    assert plan["required_checks"] == ["public-cpu", "inference-tensors", "real-weight-replays"]
    assert plan["reasons"][MG2_CASE] == [{"kind": "verified_tooling_pipeline_replay", "paths": [tool]}]
    assert not plan["uncovered_paths"] and not plan["graph_errors"]


def test_two_changed_verification_tools_require_both_complete_cpu_contracts(verification_project):
    root, matrix, dependencies, base = verification_project
    tools = (VERIFICATION_IMPACT, VERIFICATION_SUITE)
    head, changes = commit_verification_change(verification_project, tools=tools)
    proofs = verification_proofs(verification_project, head=head, changes=changes)
    assert {item["path"] for item in proofs} == set(tools)
    assert len({item["immutable_runtime_tree_sha256"] for item in proofs}) == 1
    assert all(item["required_cpu_contracts"] == VERIFICATION_CONTRACTS for item in proofs)
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert plan["selected_cases"] == [MG2_CASE]
    assert plan["required_cpu_contracts"] == VERIFICATION_CONTRACTS


def test_allowed_documentation_tests_workflow_and_make_edits_preserve_runtime_proof(verification_project):
    root, matrix, dependencies, base = verification_project
    head, changes = commit_verification_change(verification_project)
    original = verification_proofs(verification_project, head=head, changes=changes)[0]
    extra = {
        "docs/orchestration.md": "updated guide\n",
        ".github/workflows/ci.yml": "name: changed validation\n",
        "Makefile": "test:\n\tpython -m pytest -q\n",
        "tests/runtime/additional_contract.py": "# Additional checks\n",
    }
    head, changes = commit_verification_change(verification_project, extra=extra)
    proof = verification_proofs(verification_project, head=head, changes=changes)[0]
    assert proof["immutable_runtime_tree_sha256"] == original["immutable_runtime_tree_sha256"]
    assert proof["source_sha256"] == original["source_sha256"]
    assert proof["head_revision"] == head and proof["base_revision"] == base
    assert impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)["selected_cases"] == [
        MG2_CASE
    ]


@pytest.mark.parametrize(
    "path",
    [
        "worldfoundry/models/b.py",
        "MANIFEST.in",
        "requirements/model.txt",
        "envs/model.yaml",
        "assets/reference.bin",
        "thirdparty/vendor/source.py",
        "LICENSE",
    ],
)
def test_mixed_code_environment_packaging_or_asset_changes_retain_full_replay_selection(verification_project, path):
    head, changes = commit_verification_change(verification_project, extra={path: "# Changed production bytes\n"})
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("operation", ["mode", "delete", "rename", "new_asset", "symlink"])
def test_runtime_modes_membership_and_symlink_changes_cannot_use_cpu_proof(verification_project, operation):
    root, _, _, base = verification_project
    path = root / "assets/reference.bin"
    if operation == "mode":
        path.chmod(0o755)
    elif operation == "delete":
        path.unlink()
    elif operation == "rename":
        path.rename(root / "assets/renamed-reference.bin")
    elif operation == "new_asset":
        (root / "assets/new-reference.bin").write_text("additional input\n")
    else:
        path.unlink()
        path.symlink_to("../LICENSE")
    head, changes = commit_verification_change(verification_project)
    assert changes == impact.changed_paths(root, base, head)
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("definition", ["matrix", "dependencies"])
def test_committed_case_or_dependency_definition_change_cannot_be_exempted(verification_project, definition):
    _, matrix, dependencies, _ = verification_project
    path = matrix if definition == "matrix" else dependencies
    data = json.loads(path.read_text())
    if definition == "matrix":
        data["a"]["scope"] = "Changed replay definition"
    else:
        data["ignored_paths"].append("new-ignored-path/**")
    path.write_text(json.dumps(data))
    head, changes = commit_verification_change(verification_project)
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("path_kind", ["tool", "matrix", "dependencies", "contract"])
def test_uncommitted_inspected_tool_definition_or_contract_cannot_authorize_proof(verification_project, path_kind):
    root, matrix, dependencies, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    path = {
        "tool": root / VERIFICATION_SUITE,
        "matrix": matrix,
        "dependencies": dependencies,
        "contract": root / VERIFICATION_CONTRACTS[1],
    }[path_kind]
    if path_kind in {"matrix", "dependencies"}:
        path.write_text(json.dumps(json.loads(path.read_text()), indent=2) + "\n")
    else:
        path.write_text(path.read_text() + "# Edited after commit\n")
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("change", ["missing_tool", "missing_doc", "extra_path", "hidden_runtime"])
def test_incomplete_or_invented_changed_path_lists_cannot_authorize_proof(verification_project, change):
    extra = {"docs/orchestration.md": "updated guide\n"}
    if change == "hidden_runtime":
        extra["worldfoundry/models/b.py"] = "# Hidden production edit\n"
    head, changes = commit_verification_change(verification_project, extra=extra)
    if change == "missing_tool":
        supplied = [item for item in changes if item != VERIFICATION_SUITE]
    elif change == "missing_doc":
        supplied = [item for item in changes if item != "docs/orchestration.md"]
    elif change == "extra_path":
        supplied = [*changes, "docs/invented.md"]
    else:
        supplied = [item for item in changes if item != "worldfoundry/models/b.py"]
    assert verification_proofs(verification_project, head=head, changes=supplied) == []


@pytest.mark.parametrize("base", [None, "0" * 40, "missing-base-ref"])
def test_missing_or_unresolvable_history_cannot_turn_shared_tool_changes_into_cpu_only(verification_project, base):
    root, matrix, dependencies, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    assert impact.verification_tooling_proofs(root, matrix, dependencies, changes, base=base, head=head) == []
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert set(plan["selected_cases"]) == {"a", "b", MG2_CASE}
    assert not plan["cpu_only_changes"]


def test_replay_exporter_is_not_in_the_exact_verification_tool_allowance(verification_project):
    head, changes = commit_verification_change(verification_project, tools=("tests/manual/geometry_regression.py",))
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("path_kind", ["tool", "contract"])
@pytest.mark.parametrize("operation", ["missing", "symlink"])
def test_missing_or_symlinked_verification_implementation_cannot_be_exempted(
    verification_project, path_kind, operation
):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    path = root / (VERIFICATION_SUITE if path_kind == "tool" else VERIFICATION_CONTRACTS[1])
    path.unlink()
    if operation == "symlink":
        path.symlink_to(root / "LICENSE")
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_fixture_without_mg2_has_only_cpu_proof_and_claims_no_inference_pass(verification_project):
    root, matrix, dependencies, _ = verification_project
    cases, policy = json.loads(matrix.read_text()), json.loads(dependencies.read_text())
    cases.pop(MG2_CASE)
    policy["cases"].pop(MG2_CASE)
    matrix.write_text(json.dumps(cases))
    dependencies.write_text(json.dumps(policy))
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture without MG2")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(project)
    plan = impact.select_cases(matrix, dependencies, root, changes, base=project[3], head=head)
    assert plan["status"] == "no_inference_changes" and plan["selected_cases"] == []
    assert plan["required_cpu_contracts"] == [VERIFICATION_CONTRACTS[1]]
    assert plan["required_checks"] == ["public-cpu", "inference-tensors"]


@pytest.mark.parametrize("operation", ["modify", "delete", "mode"])
def test_dirty_production_checkout_cannot_use_an_immutable_git_tree_proof(verification_project, operation):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    source = root / "worldfoundry/models/b.py"
    if operation == "modify":
        source.write_text("# Uncommitted model implementation\n")
    elif operation == "delete":
        source.unlink()
    else:
        source.chmod(0o755)
    assert git(root, "rev-parse", "HEAD") == head
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("ignored", [False, True])
def test_untracked_python_shadow_including_ignored_files_cannot_use_verification_proof(verification_project, ignored):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    name = "ignored_shadow.py" if ignored else "untracked_shadow.py"
    shadow = root / "worldfoundry" / name
    shadow.write_text("# Runtime module absent from the recorded commit\n")
    status = git(root, "status", "--porcelain", "--", str(shadow.relative_to(root)))
    assert (not status) is ignored
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_untracked_interpreter_cache_is_not_confused_with_python_source_shadow(verification_project):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    cache = root / "worldfoundry/models/__pycache__"
    cache.mkdir()
    (cache / "a.cpython-311.pyc").write_bytes(b"generated interpreter cache")
    assert verification_proofs(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("excluded_root", ["docs", "tests", ".github"])
def test_runtime_symlink_to_changed_excluded_file_cannot_hide_runtime_bytes(verification_project, excluded_root):
    root, matrix, dependencies, _ = verification_project
    target = root / excluded_root / "runtime-input.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("original input consumed through a runtime alias\n")
    alias = root / "assets/aliased-input.txt"
    alias.symlink_to("../" + excluded_root + "/runtime-input.txt")
    git(root, "add", ".")
    git(root, "commit", "-qm", "runtime alias to excluded input")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(
        project, extra={str(target.relative_to(root)): "changed runtime input\n"}
    )
    assert alias.read_text() == "changed runtime input\n"
    assert_conservative_verification_fallback(project, head=head, changes=changes)


def test_runtime_symlink_to_an_immutable_tracked_license_is_validated(verification_project):
    root, matrix, dependencies, _ = verification_project
    alias = root / "thirdparty/vendor/LICENSE"
    alias.symlink_to("../../LICENSE")
    git(root, "add", ".")
    git(root, "commit", "-qm", "immutable runtime license alias")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(project)
    proofs = verification_proofs(project, head=head, changes=changes)
    assert len(proofs) == 1 and proofs[0]["path"] == VERIFICATION_SUITE
    assert impact.select_cases(matrix, dependencies, root, changes, base=project[3], head=head)["selected_cases"] == [
        MG2_CASE
    ]


def test_excluded_intermediate_symlink_cannot_redirect_an_unchanged_runtime_alias(verification_project):
    root, matrix, dependencies, _ = verification_project
    route = root / "docs/runtime-route.py"
    route.symlink_to("../worldfoundry/models/a.py")
    alias = root / "worldfoundry/runtime-alias.py"
    alias.symlink_to("../docs/runtime-route.py")
    git(root, "add", ".")
    git(root, "commit", "-qm", "runtime alias through excluded routing file")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    original = alias.read_bytes()
    route.unlink()
    route.symlink_to("../worldfoundry/models/b.py")
    head, changes = commit_verification_change(project)
    assert alias.read_bytes() != original
    assert git(root, "diff", project[3], head, "--", "worldfoundry") == ""
    assert_conservative_verification_fallback(project, head=head, changes=changes)


def test_immutable_symlink_chain_stays_inside_the_inspected_runtime_tree(verification_project):
    root, matrix, dependencies, _ = verification_project
    (root / "LICENSE_ALIAS").symlink_to("LICENSE")
    (root / "thirdparty/vendor/LICENSE").symlink_to("../../LICENSE_ALIAS")
    git(root, "add", ".")
    git(root, "commit", "-qm", "immutable runtime license chain")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(project)
    assert verification_proofs(project, head=head, changes=changes)


@pytest.mark.parametrize(
    "relative",
    [
        "torch.py",
        "numpy/__init__.py",
        "thirdparty/untracked_code.py",
        "shadow_extension.so",
        "shadow_types.pyi",
        "shadow_script.pyw",
        "shadow_extension.pyd",
        "ignored_shadow.py",
    ],
)
def test_untracked_importable_code_outside_worldfoundry_cannot_shadow_a_verified_runtime(
    verification_project, relative
):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    shadow = root / relative
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(b"untracked importable runtime shadow")
    if relative == "ignored_shadow.py":
        assert git(root, "status", "--porcelain", "--", relative) == ""
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_untracked_package_symlink_cannot_import_from_an_excluded_staging_directory(verification_project):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    staging = root / "tmp/alternate-numpy"
    staging.mkdir(parents=True)
    (staging / "__init__.py").write_text("# Package outside recorded production source\n")
    (root / "numpy").symlink_to("tmp/alternate-numpy", target_is_directory=True)
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_untracked_logs_and_excluded_staging_code_do_not_change_production_proof(verification_project):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    (root / "root.log").write_text("validation progress\n")
    staging = root / "tmp/temporary_validation.py"
    staging.parent.mkdir()
    staging.write_text("# Temporary orchestration outside runtime import roots\n")
    assert verification_proofs(verification_project, head=head, changes=changes)


def commit_additional_case(project, mutation=None):
    root, matrix, dependencies, base = project
    cases, policy = json.loads(matrix.read_text()), json.loads(dependencies.read_text())
    cases["extra"] = {"id": "extra", "target": "worldfoundry.models.b:Model"}
    policy["components"]["extra"] = ["worldfoundry/models/b.py"]
    policy["cases"]["extra"] = ["extra"]
    if mutation is not None:
        mutation(cases, policy)
    matrix.write_text(json.dumps(cases))
    dependencies.write_text(json.dumps(policy))
    (root / VERIFICATION_IMPACT).write_text("# Updated verification with added recipes\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "additional replay and verification contracts")
    head = git(root, "rev-parse", "HEAD")
    return head, impact.changed_paths(root, base, head)


def test_append_only_case_and_component_select_new_case_and_real_tooling_replay(verification_project):
    root, matrix, dependencies, base = verification_project
    head, changes = commit_additional_case(verification_project)
    assert impact.committed_additional_cases(root, matrix, dependencies, changes, base=base, head=head) == {"extra"}
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert set(plan["selected_cases"]) == {"extra", MG2_CASE}
    assert plan["status"] == "planned"
    assert plan["required_cpu_contracts"] == [VERIFICATION_CONTRACTS[0]]
    assert plan["cpu_only_changes"]
    assert any(reason["kind"] == "committed_additional_case_dependency" for reason in plan["reasons"]["extra"])


@pytest.mark.parametrize(
    "mutation", ["recipe", "component", "case_owner", "global", "json_type", "unreferenced_component"]
)
def test_case_additions_cannot_exempt_changes_to_existing_definitions(verification_project, mutation):
    root, matrix, dependencies, base = verification_project

    def mutate(cases, policy):
        if mutation == "recipe":
            cases["a"]["call"] = {"seed": 1730}
        elif mutation == "component":
            policy["components"]["a"] = ["worldfoundry/models/b.py"]
        elif mutation == "case_owner":
            policy["cases"]["a"] = ["b"]
        elif mutation == "global":
            policy["ignored_paths"].append("worldfoundry/**")
        elif mutation == "json_type":
            policy["schema_version"] = True
        else:
            policy["components"]["unused"] = ["worldfoundry/unexecuted.py"]

    head, changes = commit_additional_case(verification_project, mutate)
    assert impact.committed_additional_cases(root, matrix, dependencies, changes, base=base, head=head) is None
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert set(plan["selected_cases"]) == {"a", "b", "extra", MG2_CASE}
    assert not plan["cpu_only_changes"]


@pytest.mark.parametrize("dirty", ["matrix", "dependencies", "runtime"])
def test_case_addition_proof_never_exempts_dirty_definitions_or_runtime(verification_project, dirty):
    root, matrix, dependencies, base = verification_project
    head, changes = commit_additional_case(verification_project)
    if dirty == "runtime":
        (root / "worldfoundry/models/b.py").write_text("# Uncommitted runtime change\n")
    else:
        path = matrix if dirty == "matrix" else dependencies
        path.write_text(path.read_text() + "\n")
    assert not verification_proofs(verification_project, head=head, changes=changes)
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert set(plan["selected_cases"]) == {"a", "b", "extra", MG2_CASE}


def test_added_case_cannot_hide_independently_committed_runtime_changes(verification_project):
    root, matrix, dependencies, base = verification_project
    commit_additional_case(verification_project)
    (root / "worldfoundry/models/a.py").write_text("# Changed inference implementation\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "independent runtime change")
    head = git(root, "rev-parse", "HEAD")
    changes = impact.changed_paths(root, base, head)
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert set(plan["selected_cases"]) == {"a", "b", "extra", MG2_CASE}
    assert not plan["cpu_only_changes"]


def test_exact_environment_mapping_precedes_shared_metadata_rule(project: tuple[Path, Path, Path]) -> None:
    root, matrix, dependencies = project
    path = "worldfoundry/data/models/runtime/environments/video/a.yaml"
    policy = json.loads(dependencies.read_text())
    policy["shared_paths"].append("worldfoundry/data/models/**")
    policy["environment_variants"] = {path: ["a@a"]}
    dependencies.write_text(json.dumps(policy))
    plan = impact.select_cases(matrix, dependencies, root, [path])
    assert plan["status"] == "planned"
    assert plan["selected_cases"] == ["a"]
    assert plan["reasons"]["a"][0]["kind"] == "declared_environment"
    plan = impact.select_cases(matrix, dependencies, root, [path, "worldfoundry/core/attention/shared.py"])
    assert plan["selected_cases"] == ["a", "b"]
    policy["environment_variants"][path].append("missing@missing")
    dependencies.write_text(json.dumps(policy))
    plan = impact.select_cases(matrix, dependencies, root, [path])
    assert plan["status"] == "uncovered"
    assert plan["selected_cases"] == ["a", "b"]
    assert plan["uncovered_paths"] == [path]


@pytest.mark.parametrize(
    "mappings",
    [
        {"worldfoundry/data/models/runtime/environments/video/*.yaml": ["a@a"]},
        {"worldfoundry/data/models/runtime/environments/video/a.yaml": []},
        {"worldfoundry/data/models/runtime/environments/video/a.yaml": ["a"]},
    ],
)
def test_environment_ownership_requires_exact_paths_and_variants(mappings: dict[str, list[str]]) -> None:
    with pytest.raises(ValueError, match="Environment ownership"):
        impact.environment_variants({"environment_variants": mappings})


@pytest.mark.unit
def test_openai_clip_scoring_requires_cpu_contracts() -> None:
    module = "worldfoundry/base_models/perception_core/general_perception/openai_clip_runtime/clip"
    paths = {f"{module}/{name}.py" for name in ("clip", "model", "simple_tokenizer")}
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        TOOLS.parents[1],
        sorted(paths),
    )
    assert plan["status"] == "no_inference_changes", plan["uncovered_paths"]
    assert plan["selected_cases"] == []
    assert {row["path"] for row in plan["cpu_only_changes"]} == paths
    assert plan["required_cpu_contracts"] == ["tests/evaluation/execution/test_openai_clip_scoring.py"]
