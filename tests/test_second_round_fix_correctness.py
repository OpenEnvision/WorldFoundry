"""Second-round correctness fixes: pose, packaging, layering, pickle HTTP."""

from __future__ import annotations

import ast
import os
import re
import runpy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse(relpath: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"))


def test_human_pose_draw_mask_blends_resized_background() -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    source = (REPO_ROOT / "worldfoundry/studio/visualization/plugins/perception/human_pose.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "draw_mask"
    )
    namespace = {"np": np, "cv2": cv2, "load_image": lambda img, reverse=False: img}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "<draw_mask>", "exec"), namespace)

    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:, :3] = 255
    background = np.zeros((8, 8, 3), dtype=np.uint8)
    background[:] = (0, 255, 0)

    out = namespace["draw_mask"](img, mask, background=background, return_rgba=False)
    assert out.shape == (4, 6, 3)
    assert tuple(out[0, 0]) == (255, 0, 0)
    assert tuple(out[0, 5]) == (0, 255, 0)


def test_human_pose_unknown_stickwidth_raises_value_error() -> None:
    source = (
        REPO_ROOT / "worldfoundry/studio/visualization/plugins/perception/human_pose.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    bare_raises = [node for node in ast.walk(tree) if isinstance(node, ast.Raise) and node.exc is None]
    assert bare_raises == []
    assert "Unknown stickwidth_type" in source
    assert "alphaMerge" not in source


def test_workspace_app_uses_start_new_session_not_preexec_fn() -> None:
    source = (REPO_ROOT / "worldfoundry/studio/serving/workspace.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    popen_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
    ]
    assert popen_calls
    assert all(
        all(keyword.arg != "preexec_fn" for keyword in call.keywords)
        for call in popen_calls
    )
    assert any(
        any(keyword.arg == "start_new_session" for keyword in call.keywords)
        for call in popen_calls
    )


def test_reward_server_no_longer_unpickles_http_bodies() -> None:
    source = (
        REPO_ROOT
        / "worldfoundry/evaluation/tasks/execution/runners/worldolympiad"
        / "runtime/worldolympiad/3d_metrics/serve_reward_3d.py"
    ).read_text(encoding="utf-8")
    assert "pickle.loads" not in source
    assert "pickle.dumps" not in source
    assert "415" in source
    assert "REWARD_3D_ALLOW_NON_LOOPBACK" in source


def test_runtime_modules_do_not_import_evaluation_utils() -> None:
    runtime_dir = REPO_ROOT / "worldfoundry/runtime"
    offenders: list[str] = []
    for path in runtime_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "worldfoundry.evaluation.utils":
                offenders.append(path.name)
    assert offenders == []


def test_ray_setup_binds_runtime_before_ready_and_cleans_on_failure() -> None:
    source = (REPO_ROOT / "worldfoundry/training/distributed/ray_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    setup = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "setup")
    assigned_ray_before_get = False
    saw_get = False
    for node in ast.walk(setup):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "_ray" for target in node.targets
        ):
            assigned_ray_before_get = not saw_get
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            saw_get = True
    assert assigned_ray_before_get
    assert "self.shutdown()" in source
    assert "placement_ready_timeout_s" in source


def _load_pyproject() -> dict:
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_license_gated_packages_are_excluded_from_find_packages() -> None:
    pyproject = _load_pyproject()
    excludes = set(pyproject["tool"]["setuptools"]["packages"]["find"]["exclude"])
    required = {
        "worldfoundry.base_models.three_dimensions.general_3d.dust3r",
        "worldfoundry.base_models.three_dimensions.general_3d.dust3r.*",
        "worldfoundry.base_models.three_dimensions.general_3d.monst3r",
        "worldfoundry.base_models.three_dimensions.general_3d.mast3r",
        "worldfoundry.base_models.three_dimensions.point_clouds.gaussian_splatting",
        "worldfoundry.synthesis.visual_generation.hunyuan_world",
        "worldfoundry.synthesis.visual_generation.hunyuan_world.*",
    }
    missing = required - excludes
    assert missing == set()


def test_core_inference_does_not_import_pipelines() -> None:
    source = (REPO_ROOT / "worldfoundry/core/inference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("worldfoundry.pipelines"):
            raise AssertionError(f"core.inference still imports {node.module}")
    constants = (REPO_ROOT / "worldfoundry/pipelines/gen3c/constants.py").read_text(encoding="utf-8")
    assert "DEFAULT_GEN3C_PROMPT" in constants
    assert "worldfoundry.core.model_defaults" not in constants
    assert not (REPO_ROOT / "worldfoundry/core/model_defaults.py").exists()


def test_pipeline_invocation_contract_is_core_owned_and_compatible() -> None:
    from worldfoundry.core.contracts import PipelineInvocation as CoreInvocation
    from worldfoundry.evaluation.models.pipelines.invocation import (
        PipelineInvocation as EvaluationInvocation,
    )

    assert EvaluationInvocation is CoreInvocation
    for relpath in (
        "worldfoundry/pipelines/bernini/pipeline_bernini.py",
        "worldfoundry/pipelines/video_official/pipeline_official_video.py",
    ):
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "worldfoundry.evaluation" not in source
        assert "worldfoundry.core.contracts" in source


def test_runtime_geometry_is_core_owned_and_studio_path_is_compatible() -> None:
    from worldfoundry.core.geometry import depth_to_world_points as core_helper
    from worldfoundry.studio.visualization.core.geometry import (
        depth_to_world_points as studio_helper,
    )

    assert studio_helper is core_helper
    for relpath in (
        "worldfoundry/pipelines/cut3r/official_runtime.py",
        "worldfoundry/base_models/three_dimensions/three_d_four_d/runtime.py",
    ):
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "worldfoundry.studio" not in source
        assert "worldfoundry.core.geometry" in source


def test_embodied_dataset_roots_require_explicit_configuration(monkeypatch) -> None:
    from worldfoundry.evaluation.tasks.embodied.simulators.calvin.benchmark import (
        CALVINBenchmark,
    )
    from worldfoundry.evaluation.tasks.embodied.simulators.robocerebra.benchmark import (
        RoboCerebraBenchmark,
    )

    monkeypatch.delenv("WORLDFOUNDRY_CALVIN_DATASET_ROOT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_ROBOCEREBRA_ROOT", raising=False)
    with pytest.raises(ValueError, match="CALVIN dataset path is required"):
        CALVINBenchmark()
    with pytest.raises(ValueError, match="RoboCerebra root is required"):
        RoboCerebraBenchmark()

    calvin_root = REPO_ROOT / "tmp" / "calvin-data"
    robocerebra_root = REPO_ROOT / "tmp" / "robocerebra"
    monkeypatch.setenv("WORLDFOUNDRY_CALVIN_DATASET_ROOT", str(calvin_root))
    monkeypatch.setenv("WORLDFOUNDRY_ROBOCEREBRA_ROOT", str(robocerebra_root))
    assert CALVINBenchmark().dataset_path == str(calvin_root / "validation")
    assert RoboCerebraBenchmark().robocerebra_root == str(robocerebra_root)

    demo_source = (
        REPO_ROOT
        / "worldfoundry/evaluation/tasks/execution/runners/devil_dynamics/runtime/official"
        / "metrics_utils/standard_video_dataset.py"
    ).read_text(encoding="utf-8")
    assert "/home/LiaoMingxiang" not in demo_source
    assert 'add_argument("video_folder"' in demo_source


def test_worldolympiad_openrouter_key_priority(monkeypatch) -> None:
    module = runpy.run_path(
        str(
            REPO_ROOT
            / "worldfoundry/evaluation/tasks/execution/runners/worldolympiad/runtime"
            / "worldolympiad/model/openrouter.py"
        )
    )
    resolver = module["get_openrouter_api_key"]
    for name in ("WORLDFOUNDRY_OPENROUTER_API_KEY", "OPENROUTER_API_KEY", "api_key"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("api_key", "legacy")
    assert resolver() == "legacy"
    monkeypatch.setenv("OPENROUTER_API_KEY", "ecosystem")
    assert resolver() == "ecosystem"
    monkeypatch.setenv("WORLDFOUNDRY_OPENROUTER_API_KEY", "canonical")
    assert resolver() == "canonical"

    source = (
        REPO_ROOT
        / "worldfoundry/evaluation/tasks/execution/runners/worldolympiad/runtime/worldolympiad"
        / "model/openrouter.py"
    ).read_text(encoding="utf-8")
    assert "Bearer {os.getenv('api_key')}" not in source


def _finally_call_count(relpath: str, call_name: str) -> int:
    tree = _parse(relpath)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and node.finalbody
        and any(
            isinstance(child, ast.Call)
            and (
                (isinstance(child.func, ast.Attribute) and child.func.attr == call_name)
                or (isinstance(child.func, ast.Name) and child.func.id == call_name)
            )
            for statement in node.finalbody
            for child in ast.walk(statement)
        )
    )


def test_runner_temporary_media_is_cleaned_in_finally() -> None:
    wbench = (
        "worldfoundry/evaluation/tasks/execution/runners/wbench/runtime/wbench/src/metrics/vlm/"
        "vlm_evaluator.py"
    )
    memobench = (
        "worldfoundry/evaluation/tasks/execution/runners/memobench/runtime/memobench/evaluation/"
        "vqa/llm-vqa.py"
    )
    qwen = (
        "worldfoundry/evaluation/tasks/execution/runners/videoscore/runtime/videoscore/benchmark/"
        "mllm_tools/qwenVL_eval.py"
    )
    assert _finally_call_count(wbench, "remove") >= 2
    assert _finally_call_count(memobench, "unlink") >= 1
    assert _finally_call_count(qwen, "close") >= 1
    qwen_source = (REPO_ROOT / qwen).read_text(encoding="utf-8")
    assert "except FileNotFoundError" in qwen_source
    assert "merged_image_files = []" not in qwen_source.split("def __init__", 1)[0]


def test_artifact_timestamps_are_utc_and_elapsed_time_is_monotonic() -> None:
    from worldfoundry.core.io.file_utils import timestamp_file_name

    stamped = timestamp_file_name("scorecard.json")
    assert re.fullmatch(r"scorecard_\d{8}-\d{6}\.json", stamped)

    camera = (
        REPO_ROOT
        / "worldfoundry/evaluation/tasks/execution/runners/camerabench/camerabench_metrics.py"
    ).read_text(encoding="utf-8")
    assert camera.count("datetime.now(timezone.utc).isoformat()") == 3
    assert "datetime.now().isoformat()" not in camera

    shards = (
        REPO_ROOT / "worldfoundry/core/checkpoint/sharded_safetensors.py"
    ).read_text(encoding="utf-8")
    assert "perf_counter()" in shards
    assert "datetime.now()" not in shards


def test_ruff_excludes_evaluation_vendored_trees() -> None:
    pyproject = _load_pyproject()
    excludes = pyproject["tool"]["ruff"]["extend-exclude"]
    assert "worldfoundry/evaluation/tasks/execution/runners/*/runtime" in excludes
    assert "worldfoundry/evaluation/tasks/metrics/*/vendor" in excludes


def _comprehension_targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        return [generator.target for generator in node.generators]
    return []


def test_workspace_app_loop_vars_do_not_shadow_dataclasses_field() -> None:
    tree = _parse("worldfoundry/studio/serving/workspace.py")
    shadowed: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "field":
            shadowed.append(node.lineno)
        for target in _comprehension_targets(node):
            if isinstance(target, ast.Name) and target.id == "field":
                shadowed.append(node.lineno)
    assert shadowed == []


def test_robotics_plugin_drops_duplicate_require_package_imports() -> None:
    tree = _parse("worldfoundry/studio/visualization/plugins/robotics/robotics.py")
    require_imports = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "require_package" for alias in node.names)
    ]
    assert len(require_imports) == 1

    bound: dict[str, list[int]] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.asname or alias.name.split(".")[-1] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [alias.asname or alias.name for alias in node.names]
        for name in names:
            bound.setdefault(name, []).append(node.lineno)
    for name in ("require_package", "plt", "np", "dataclass"):
        assert len(bound.get(name, [])) <= 1, name


def test_gradio_patches_are_explicit_and_reversible() -> None:
    source = (REPO_ROOT / "worldfoundry/studio/ui/gradio_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "def install_gradio_patches" in source
    assert "def uninstall_gradio_patches" in source
    top_calls = [
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert "install_gradio_patches" not in top_calls
    assert not any(name.startswith("_install_") for name in top_calls)
    app_source = (REPO_ROOT / "worldfoundry/studio/ui/gradio_app.py").read_text(encoding="utf-8")
    app_tree = ast.parse(app_source)
    build_demo = next(
        node for node in app_tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_demo"
    )
    first_statement = build_demo.body[0]
    assert isinstance(first_statement, ast.Expr)
    assert isinstance(first_statement.value, ast.Call)
    assert isinstance(first_statement.value.func, ast.Name)
    assert first_statement.value.func.id == "install_gradio_patches"


def test_gradio_patches_uninstall_restores_url_ok() -> None:
    pytest.importorskip("gradio")
    from worldfoundry.studio.ui import gradio_runtime as grt
    import gradio.networking as gr_networking

    grt.install_gradio_patches()
    assert getattr(gr_networking, "_worldfoundry_proxy_safe_url_ok", False)
    patched = gr_networking.url_ok
    try:
        grt.uninstall_gradio_patches()
        assert not getattr(gr_networking, "_worldfoundry_proxy_safe_url_ok", False)
        assert gr_networking.url_ok is not patched
    finally:
        grt.install_gradio_patches()


def test_torchrun_control_group_create_and_shutdown_use_lifecycle_state() -> None:
    source = (REPO_ROOT / "worldfoundry/studio/inference/execution.py").read_text(encoding="utf-8")
    assert source.count("with _TORCHRUN_CONTROL_GROUP_CONDITION:") >= 3
    assert '_TORCHRUN_CONTROL_GROUP_STATE = "closing"' in source
    assert '_TORCHRUN_CONTROL_GROUP_STATE = "open"' in source
    assert "_TORCHRUN_CONTROL_GROUP_GENERATION += 1" in source
    assert "lambda: not _TORCHRUN_CONTROL_GROUP_CREATING" in source
    tree = ast.parse(source)
    names = {"_torchrun_control_group", "shutdown_torchrun_lingbot_fast_runtime"}
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert names <= set(found)


def test_env_registry_documents_legacy_trainer_and_wm_aliases() -> None:
    source = (REPO_ROOT / "worldfoundry/core/env_registry.py").read_text(encoding="utf-8")
    assert "LEGACY_ENV_ALIASES" in source
    assert "def getenv_registered" in source
    assert "TRAINER_TORCH_PROFILER_DIR" in source
    assert "WM_AUTO_CUDA_VISIBLE_DEVICES" in source

    from worldfoundry.core.env_registry import getenv_registered as core_getenv_registered
    from worldfoundry.runtime.env import getenv_registered as runtime_getenv_registered

    assert runtime_getenv_registered is core_getenv_registered
    runtime_source = (REPO_ROOT / "worldfoundry/runtime/env.py").read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_DETERMINISTIC" in runtime_source


def test_getenv_registered_falls_back_to_legacy_name() -> None:
    import warnings

    from worldfoundry.runtime.env import getenv_registered

    os_environ_key = "TRAINER_TORCH_PROFILER_DIR"
    canonical = "WORLDFOUNDRY_TRAINER_TORCH_PROFILER_DIR"
    previous = os.environ.pop(canonical, None)
    legacy_previous = os.environ.pop(os_environ_key, None)
    os.environ[os_environ_key] = "/tmp/wf-xc8-profiler"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getenv_registered(canonical)
        assert value == "/tmp/wf-xc8-profiler"
        assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    finally:
        os.environ.pop(os_environ_key, None)
        if previous is not None:
            os.environ[canonical] = previous
        if legacy_previous is not None:
            os.environ[os_environ_key] = legacy_previous


def test_seed_helpers_honour_worldfoundry_deterministic_env() -> None:
    source = (REPO_ROOT / "worldfoundry/core/utils/torch_utils.py").read_text(encoding="utf-8")
    assert "def apply_deterministic_from_env" in source
    assert 'WORLDFOUNDRY_DETERMINISTIC' in source
    inference = (REPO_ROOT / "worldfoundry/core/inference.py").read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_DETERMINISTIC" in inference
    assert "Failed to set float32 matmul precision" in inference


def test_training_sessions_use_set_seed_everywhere() -> None:
    session = (
        REPO_ROOT / "worldfoundry/training/engine/sessions/single_device.py"
    ).read_text(encoding="utf-8")
    sana = [
        (REPO_ROOT / "worldfoundry/training/engine/sana/scm_ladd.py").read_text(encoding="utf-8"),
        (REPO_ROOT / "worldfoundry/training/engine/sana/sid.py").read_text(encoding="utf-8"),
        (REPO_ROOT / "worldfoundry/training/engine/sana/sft.py").read_text(encoding="utf-8"),
    ]
    assert "set_seed_everywhere" in session
    assert "random.seed(" not in session
    for source in sana:
        assert "set_seed_everywhere" in source
        assert "random.seed(" not in source


def test_kernel_autotune_prefers_enabled_suffix() -> None:
    cache = (REPO_ROOT / "worldfoundry/core/kernels/autotune_cache.py").read_text(encoding="utf-8")
    registry = (REPO_ROOT / "worldfoundry/core/kernels/registry.py").read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_KERNEL_AUTOTUNE_CACHE_ENABLED" in cache
    assert "WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED" in registry


def test_deleted_utils_are_not_lazily_exported() -> None:
    source = (REPO_ROOT / "worldfoundry/core/utils/__init__.py").read_text(encoding="utf-8")
    for name in ("make_registry_metaclass", "encode_base64", "decode_base64"):
        assert name not in source


def test_diffusion_utils_uses_torch_amp_autocast() -> None:
    source = (REPO_ROOT / "worldfoundry/core/nn/diffusion_utils.py").read_text(encoding="utf-8")
    assert "torch.cuda.amp.autocast" not in source
    assert 'torch.amp.autocast' in source


def test_trainer_envs_prefer_worldfoundry_prefix(monkeypatch) -> None:
    monkeypatch.delenv("TRAINER_TARGET_DEVICE", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_TRAINER_TARGET_DEVICE", "cpu")
    import worldfoundry.core.distributed.sequence_parallel.envs as trainer_envs

    assert trainer_envs.TRAINER_TARGET_DEVICE == "cpu"


def test_metric_sync_uses_module_logger() -> None:
    source = (REPO_ROOT / "worldfoundry/core/distributed/metric_sync.py").read_text(
        encoding="utf-8"
    )
    assert "getLogger(__name__)" in source
    assert "getLogger()" not in source


def test_fsim_iw_ssim_uses_linalg_eigh() -> None:
    source = (
        REPO_ROOT
        / "worldfoundry/evaluation/tasks/metrics/fsim/vendor/piq/iw_ssim.py"
    ).read_text(encoding="utf-8")
    assert "torch.symeig" not in source
    assert "torch.linalg.eigh" in source


def test_trainer_logging_config_does_not_rewire_root() -> None:
    source = (
        REPO_ROOT / "worldfoundry/core/distributed/sequence_parallel/logger.py"
    ).read_text(encoding="utf-8")
    assert '"root":' not in source
    assert 'logging_config.pop("root"' in source


def test_scratch_directory_lives_under_cache_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(tmp_path))
    from worldfoundry.core.io.paths import scratch_directory

    created = scratch_directory("xc17_")
    assert created.is_dir()
    assert created.parent == tmp_path / "scratch"
    assert created.name.startswith("xc17_")


def test_xc17_callers_use_scratch_directory() -> None:
    files = [
        "worldfoundry/pipelines/lyra/pipeline_lyra1.py",
        "worldfoundry/pipelines/lyra/lyra_utils.py",
        "worldfoundry/pipelines/matrix_game/pipeline_matrix_game_3.py",
        "worldfoundry/core/io/video.py",
        "worldfoundry/evaluation/models/runtime/profile_synthesis.py",
        "worldfoundry/studio/inference/dispatch.py",
    ]
    for relpath in files:
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        if "conda_dispatch" in relpath:
            assert "getenv_registered" in source
            assert 'os.getenv("WM_AUTO_CUDA_VISIBLE_DEVICES"' not in source
        else:
            assert "scratch_directory" in source
            assert "tempfile.mkdtemp" not in source


def test_suite_and_studio_closures_bind_loop_variables() -> None:
    suite = (
        REPO_ROOT
        / "worldfoundry/evaluation/tasks/execution/orchestration/model_benchmark_suite.py"
    ).read_text(encoding="utf-8")
    studio = (REPO_ROOT / "worldfoundry/studio/inference/execution.py").read_text(encoding="utf-8")
    assert "def acquire_runner(plan=plan, runner_state=runner_state)" in suite
    studio_tree = ast.parse(studio)
    parents: dict[ast.AST, ast.AST] = {
        child: parent
        for parent in ast.walk(studio_tree)
        for child in ast.iter_child_nodes(parent)
    }
    closures = [
        node
        for node in ast.walk(studio_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "persisted_preview"
    ]
    assert len(closures) == 1
    ancestor = parents.get(closures[0])
    while ancestor is not None:
        assert not isinstance(ancestor, (ast.For, ast.AsyncFor))
        ancestor = parents.get(ancestor)
