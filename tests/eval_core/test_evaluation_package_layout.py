from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_uses_lmms_eval_style_top_level_domains() -> None:
    evaluation_root = REPO_ROOT / "worldfoundry/evals"
    tasks_root = evaluation_root / "tasks"
    models_root = evaluation_root / "models"

    assert tasks_root.is_dir()
    assert not (evaluation_root / "benchmarks").exists()

    top_level_packages = {
        path.name for path in evaluation_root.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    assert top_level_packages == {
        "api",
        "models",
        "reporting",
        "tasks",
    }
    assert (REPO_ROOT / "worldfoundry/cli").is_dir()
    assert (REPO_ROOT / "worldfoundry/mcp").is_dir()
    assert "runner" not in top_level_packages
    assert "cli" not in top_level_packages
    assert "mcp" not in top_level_packages
    assert "tui" not in top_level_packages
    assert "runtime" not in top_level_packages

    assert (evaluation_root / "public.py").is_file()
    for removed in ("runner.py", "framework.py", "__main__.py"):
        assert not (evaluation_root / removed).exists()
    assert (evaluation_root / "utils.py").is_file()
    assert (evaluation_root / "reporting" / "__init__.py").is_file()
    assert not (evaluation_root / "reporting.py").exists()
    assert not (evaluation_root / "runtime.py").exists()
    assert not (evaluation_root / "local_open_eval.py").exists()
    assert not (evaluation_root / "worldscore.py").exists()
    assert (REPO_ROOT / "worldfoundry/runtime/__init__.py").is_file()
    core_attention_root = REPO_ROOT / "worldfoundry/core/attention"
    core_nn_root = REPO_ROOT / "worldfoundry/core/nn"
    for core_attention_module in ("native.py", "rope.py", "kvcache.py", "cp.py"):
        assert (core_attention_root / core_attention_module).is_file()
    for retired_core_nn_module in ("attention.py", "rope.py"):
        assert not (core_nn_root / retired_core_nn_module).exists()
    for core_nn_module in ("inventory.py", "normalization.py", "patching.py", "transformer.py"):
        assert (core_nn_root / core_nn_module).is_file()
    assert not any(path.name == "qa" for path in (REPO_ROOT / "scripts").iterdir() if path.is_dir())
    assert (REPO_ROOT / "worldfoundry/cli/tui/launcher.py").is_file()
    assert (REPO_ROOT / "worldfoundry/cli/catalog.py").is_file()
    assert (tasks_root / "catalog" / "__init__.py").is_file()
    assert not (tasks_root / "catalog" / "benchmark_discovery.py").exists()
    assert not (tasks_root / "catalog" / "discovery.py").exists()
    assert not (tasks_root / "catalog" / "integrity.py").exists()
    assert not (tasks_root / "catalog" / "maturity.py").exists()
    assert not (tasks_root / "catalog" / "validation_matrix.py").exists()
    assert (tasks_root / "catalog" / "registry.py").is_file()
    assert (tasks_root / "catalog" / "schema.py").is_file()
    assert (tasks_root / "catalog" / "benchmark_catalog.py").is_file()
    assert (tasks_root / "catalog" / "specs.py").is_file()
    assert not (tasks_root / "catalog" / "suites.py").exists()
    assert not (tasks_root / "catalog" / "benchmark_runtime_profiles.py").exists()
    assert (tasks_root / "catalog" / "yaml.py").is_file()
    assert (tasks_root / "catalog" / "zoo_registry.py").is_file()
    assert (tasks_root / "datasets" / "__init__.py").is_file()
    assert (tasks_root / "datasets" / "manifest.py").is_file()
    assert (tasks_root / "datasets" / "manager.py").is_file()
    assert not (tasks_root / "datasets" / "readiness.py").exists()
    assert not (tasks_root / "datasets" / "materialization.py").exists()
    assert not (tasks_root / "zoo.py").exists()
    assert not (tasks_root / "zoo").exists()
    assert (tasks_root / "metrics" / "__init__.py").is_file()
    assert (tasks_root / "metrics" / "protocols.py").is_file()
    assert (tasks_root / "metrics" / "formulas.py").is_file()
    assert (tasks_root / "metrics" / "artifacts.py").is_file()
    assert (tasks_root / "metrics" / "bindings.py").is_file()
    assert (tasks_root / "metrics" / "builtins.py").is_file()
    assert (tasks_root / "metrics" / "evaluators.py").is_file()
    assert (tasks_root / "metrics" / "local_evaluators.py").is_file()
    assert (tasks_root / "metrics" / "registry.py").is_file()
    assert (tasks_root / "contracts" / "__init__.py").is_file()
    assert (tasks_root / "contracts" / "external.py").is_file()
    assert (tasks_root / "contracts" / "registry.py").is_file()
    assert not (tasks_root / "contracts" / "sources.py").exists()
    assert not (tasks_root / "native").exists()
    assert (tasks_root / "embodied" / "__init__.py").is_file()
    assert (tasks_root / "embodied" / "contracts.py").is_file()
    assert (tasks_root / "embodied" / "evaluate.py").is_file()
    assert (tasks_root / "embodied" / "materialize.py").is_file()
    assert (tasks_root / "embodied" / "metrics.py").is_file()
    assert (tasks_root / "embodied" / "normalizer.py").is_file()
    assert (tasks_root / "execution" / "__init__.py").is_file()
    assert (tasks_root / "execution" / "framework" / "__init__.py").is_file()
    assert (tasks_root / "execution" / "framework" / "official_runner.py").is_file()
    assert (tasks_root / "execution" / "framework" / "io.py").is_file()
    assert (tasks_root / "execution" / "framework" / "runner_registry.py").is_file()
    assert (tasks_root / "execution" / "orchestration" / "__init__.py").is_file()
    assert (tasks_root / "execution" / "orchestration" / "benchmark_runner.py").is_file()
    assert (tasks_root / "execution" / "orchestration" / "plan.py").is_file()
    assert (tasks_root / "execution" / "orchestration" / "model_benchmark.py").is_file()
    assert not (tasks_root / "execution" / "orchestration" / "manifest_cli.py").exists()
    assert not (tasks_root / "execution" / "orchestration" / "runtime_preflight.py").exists()
    assert not (tasks_root / "execution" / "manifest_cli.py").exists()
    assert not (tasks_root / "execution" / "runtime_preflight.py").exists()
    assert not (tasks_root / "execution" / "framework" / "dataset_readiness.py").exists()
    assert not (tasks_root / "execution" / "download_datasets.py").exists()
    assert not (tasks_root / "execution" / "materialize_benchmark_assets.py").exists()
    assert (tasks_root / "execution" / "runners").is_dir()
    assert (tasks_root / "official" / "__init__.py").is_file()
    assert not (tasks_root / "official" / "chronomagic_assets.py").exists()
    assert (tasks_root / "official" / "in_tree.py").is_file()
    assert (tasks_root / "official" / "normalizers.py").is_file()
    assert (tasks_root / "official" / "result_normalizer.py").is_file()
    assert (tasks_root / "official" / "physics_video.py").is_file()
    assert (tasks_root / "official" / "video_quality.py").is_file()
    assert not (tasks_root / "official" / "worldarena.py").exists()
    assert not (tasks_root / "official" / "worldscore.py").exists()
    assert not (tasks_root / "benchmark_lightweight_metrics.py").exists()
    for retired_task_shim in (
        "benchmark_chronomagic_assets.py",
        "benchmark_contract_registry.py",
        "benchmark_contract_types.py",
        "benchmark_contracts.py",
        "benchmark_integrity.py",
        "benchmark_in_tree_evaluators.py",
        "benchmark_maturity.py",
        "benchmark_normalizers.py",
        "benchmark_official_results_normalizer.py",
        "benchmark_physics_video_evaluator.py",
        "benchmark_registry.py",
        "benchmark_schema.py",
        "benchmark_specs.py",
        "benchmark_suites.py",
        "benchmark_video_quality_evaluator.py",
        "benchmark_worldarena_official.py",
        "benchmark_worldscore.py",
        "benchmark_metric_artifacts.py",
        "benchmark_metric_bindings.py",
        "benchmark_metric_evaluators.py",
        "benchmark_metric_local_evaluators.py",
        "catalog.py",
        "dataset_manifest.py",
        "dataset_manager.py",
        "dataset_materialization.py",
        "dataset_readiness.py",
        "datasets.py",
        "discovery.py",
        "registry.py",
        "run_embodied.py",
        "run_embodied_contracts.py",
        "run_embodied_evaluate.py",
        "run_embodied_materialize.py",
        "run_embodied_metrics.py",
        "run_embodied_normalizer.py",
        "run_metric_builtins.py",
        "run_metrics.py",
        "solaris_multiplayer.py",
        "zoo.py",
    ):
        assert not (tasks_root / retired_task_shim).exists()
    assert not (models_root / "zoo.py").exists()
    assert (models_root / "pipelines").is_dir()
    for package in ("catalog", "integrations", "pipelines", "runners", "runtime"):
        assert (models_root / package).is_dir()
    assert not (models_root / "compat").exists()
    for public_surface in ("catalog", "integrations", "pipelines", "runtime"):
        assert (models_root / public_surface / "__init__.py").is_file()
    for retired_model_module in (
        ("pipeline_", "loading.py"),
        ("pipeline_", "inference.py"),
        ("reference_", "runner.py"),
    ):
        assert not (models_root / "".join(retired_model_module)).exists()
    for removed_empty_module in (
        "pipeline_adapters.py",
        "pipeline_invocation.py",
        "pipeline_lifecycle.py",
        "pipeline_results.py",
        "pipeline_runner_loading.py",
        "pipelines.py",
        "model_integration.py",
        "integration.py",
        "runners/__init__.py",
        "runner_plugins.py",
        "runner_registry.py",
        "runtime_profiles.py",
        "zoo_schema.py",
    ):
        assert not (models_root / removed_empty_module).exists()
    assert not (models_root / "zoo").exists()

    nested_subpackages = sorted(
        str(path.relative_to(REPO_ROOT))
        for domain_root in (evaluation_root / "api", models_root, tasks_root)
        for path in domain_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )
    assert nested_subpackages == [
        "worldfoundry/zoo/catalog",
        "worldfoundry/evals/models/integrations",
        "worldfoundry/evals/models/pipelines",
        "worldfoundry/evals/models/runners",
        "worldfoundry/zoo/runtime",
        "worldfoundry/evals/tasks/catalog",
        "worldfoundry/evals/tasks/contracts",
        "worldfoundry/evals/tasks/datasets",
        "worldfoundry/evals/tasks/embodied",
        "worldfoundry/evals/tasks/execution",
        "worldfoundry/evals/tasks/metrics",
        "worldfoundry/evals/tasks/official",
        "worldfoundry/evals/tasks/release",
    ]


def test_scripts_qa_tree_is_not_packaged() -> None:
    assert not any(path.name == "qa" for path in (REPO_ROOT / "scripts").iterdir() if path.is_dir())


def test_dataset_helpers_are_split_by_responsibility() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evals/tasks"
    datasets_facade = tasks_root / "datasets" / "__init__.py"

    assert sum(1 for _ in datasets_facade.open(encoding="utf-8")) < 180
    for retired_dataset_module in (
        "dataset_manifest.py",
        "dataset_manager.py",
        "dataset_readiness.py",
        "dataset_materialization.py",
        "datasets.py",
    ):
        assert not (tasks_root / retired_dataset_module).exists()
    for marker in ("# manifest.py", "# manager.py", "# readiness.py", "# materialization.py"):
        assert marker not in datasets_facade.read_text(encoding="utf-8")


def test_zoo_schemas_use_shared_json_contract_base() -> None:
    model_schema = REPO_ROOT / "worldfoundry/zoo/catalog/schema.py"
    benchmark_schema = REPO_ROOT / "worldfoundry/evals/tasks/catalog/schema.py"

    for path in (model_schema, benchmark_schema):
        text = path.read_text(encoding="utf-8")
        assert "JsonSerializable = JsonContract" in text
        assert "def _to_plain(" not in text
        assert "def _json_dumps(" not in text
        assert "def _require_mapping(" not in text
        assert "class JsonSerializable" not in text


def test_embodied_contracts_use_shared_json_contract_base() -> None:
    contract_path = REPO_ROOT / "worldfoundry/evals/tasks/embodied/contracts.py"
    text = contract_path.read_text(encoding="utf-8")

    assert "JsonDataclass = JsonContract" in text
    assert "def _to_plain(" not in text
    assert "class JsonDataclass" not in text
    assert "hashlib" not in text
    assert "json.dumps" not in text


def test_embodied_task_modules_are_owned_by_embodied_package() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evals/tasks"

    import worldfoundry.evals.tasks.embodied as canonical
    import worldfoundry.evals.tasks.embodied.contracts as canonical_contracts
    import worldfoundry.evals.tasks.embodied.evaluate as canonical_evaluate
    import worldfoundry.evals.tasks.embodied.materialize as canonical_materialize
    import worldfoundry.evals.tasks.embodied.metrics as canonical_metrics
    import worldfoundry.evals.tasks.embodied.normalizer as canonical_normalizer

    for retired_module in (
        "run_embodied.py",
        "run_embodied_contracts.py",
        "run_embodied_evaluate.py",
        "run_embodied_materialize.py",
        "run_embodied_metrics.py",
        "run_embodied_normalizer.py",
    ):
        assert not (tasks_root / retired_module).exists()

    assert canonical_contracts.EmbodiedGenerationSpec is canonical.EmbodiedGenerationSpec
    assert canonical_evaluate.VlaVaWamRunRequest is canonical.VlaVaWamRunRequest
    assert canonical_materialize.materialize_vla_va_wam_requests is canonical.materialize_vla_va_wam_requests
    assert canonical_metrics.ResultFieldMetric is canonical.ResultFieldMetric
    assert canonical_normalizer.normalize_results is canonical.normalize_vla_va_wam_results


def test_external_benchmark_contract_registry_is_split_from_contract_data() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evals/tasks"
    contracts_module = tasks_root / "contracts" / "external.py"
    registry_module = tasks_root / "contracts" / "registry.py"

    contracts_text = contracts_module.read_text(encoding="utf-8")
    registry_text = registry_module.read_text(encoding="utf-8")

    assert not (tasks_root / "benchmark_contracts.py").exists()
    assert not (tasks_root / "benchmark_contract_registry.py").exists()
    assert not (tasks_root / "benchmark_contract_types.py").exists()
    assert not (tasks_root / "contracts" / "types.py").exists()
    assert not (tasks_root / "contracts" / "sources.py").exists()
    assert "class ExternalBenchmarkContract" in registry_text
    assert "class ExternalBenchmarkContractRegistry" in registry_text
    assert "ExternalBenchmarkContractRegistry(" in contracts_text
    assert "class ExternalBenchmarkContract" not in contracts_text
    assert "_CONTRACTS:" not in contracts_text
    assert 'known = ", ".join' not in contracts_text


def test_task_package_facade_is_lazy_and_bounded() -> None:
    tasks_init = REPO_ROOT / "worldfoundry/evals/tasks/__init__.py"
    text = tasks_init.read_text(encoding="utf-8")
    watched_modules = [
        "worldfoundry.evals.tasks.catalog.zoo_registry",
        "worldfoundry.evals.tasks.catalog.yaml",
        "worldfoundry.evals.tasks.catalog.registry",
        "worldfoundry.evals.tasks.execution.runners",
    ]
    script = (
        "import json, sys; "
        "import worldfoundry.evals.tasks; "
        f"watched = {watched_modules!r}; "
        "print(json.dumps([name for name in watched if name in sys.modules]))"
    )

    assert sum(1 for _ in tasks_init.open(encoding="utf-8")) < 90
    assert "from .catalog import" not in text
    assert "from .native_loader import" not in text
    assert "from .registry import" not in text
    assert "import_module" in text

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == []

    from worldfoundry.evals.tasks import load_benchmark_zoo_registry, load_task_registry_from_paths
    from worldfoundry.evals.tasks.catalog.registry import load_task_registry_from_paths as RegistryLoader
    from worldfoundry.evals.tasks.catalog.zoo_registry import load_benchmark_zoo_registry as ZooLoader

    assert load_benchmark_zoo_registry is ZooLoader
    assert load_task_registry_from_paths is RegistryLoader


def test_native_task_package_is_retired() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evals/tasks"

    assert not (tasks_root / "native").exists()
    for retired_native_shim in ("native_schema.py", "native_loader.py", "yaml.py"):
        assert not (tasks_root / retired_native_shim).exists()

    import importlib

    import pytest

    import worldfoundry.evals.tasks as facade
    import worldfoundry.evals.tasks.catalog.yaml as canonical_yaml

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("worldfoundry.evals.tasks.native")
    assert facade.load_catalog_yaml is canonical_yaml.load_catalog_yaml
    assert facade.load_yaml_mapping_with_extends is canonical_yaml.load_yaml_mapping_with_extends


def test_model_pipeline_package_keeps_path_a_surface_only() -> None:
    models_root = REPO_ROOT / "worldfoundry/evals/models"
    pipeline_root = models_root / "pipelines"
    pipelines_facade = pipeline_root / "__init__.py"

    pipelines_facade_text = pipelines_facade.read_text(encoding="utf-8")
    for marker in (
        "_LEGACY",
        "UNSUPPORTED_GENERIC_INFER",
        "load_*_pipeline",
        "infer_*_pipeline",
    ):
        assert marker not in pipelines_facade_text

    lazy_script = (
        "import json, sys; "
        "import worldfoundry.evals.models.pipelines; "
        "blocked = {'torch', 'diffusers'}; "
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in blocked)))"
    )
    lazy_result = subprocess.run(
        [sys.executable, "-c", lazy_script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(lazy_result.stdout) == []

    assert not (models_root / "compat").exists()
    for legacy_root_module in ("pipeline_runner.py", "resolver.py", "zoo.py"):
        assert not (models_root / legacy_root_module).exists()
    for module in (
        "lifecycle",
        "invocation",
        "results",
        "loading",
    ):
        assert (pipeline_root / f"{module}.py").is_file()
    for removed_module in (
        "discovery",
        "dispatch",
        "handlers",
        "components",
        "adapters",
        "media",
        "game_world",
        "named",
        "runtime_video",
        "variants",
        "world_video",
        "three" + "_dim",
        "loading" + "_maps",
    ):
        assert not (pipeline_root / f"{removed_module}.py").exists()
    for removed_module in (
        "pipeline_" + "loading.py",
        "pipeline_" + "inference.py",
        "reference_" + "runner.py",
    ):
        assert not (models_root / removed_module).exists()
    zoo_root = REPO_ROOT / "worldfoundry/zoo"
    for module in ("manifest", "policy", "registry", "schema", "zoo_registry"):
        assert (zoo_root / "catalog" / f"{module}.py").is_file()
    for module in ("profiles", "environments", "assets"):
        assert (zoo_root / "runtime" / f"{module}.py").is_file()
    for module in ("bindings", "aliases"):
        assert (zoo_root / f"{module}.py").is_file()
    assert not (models_root / "catalog").exists()
    assert not (models_root / "runtime").exists()
    for module in ("builtins", "pipeline", "registry", "resolver"):
        assert (models_root / "runners" / f"{module}.py").is_file()
    for removed_module in ("plugins", "reference"):
        assert not (models_root / "runners" / f"{removed_module}.py").exists()


def test_model_package_facade_is_lazy_and_bounded() -> None:
    models_init = REPO_ROOT / "worldfoundry/evals/models/__init__.py"
    text = models_init.read_text(encoding="utf-8")
    watched_modules = [
        "worldfoundry.zoo.catalog.registry",
        "worldfoundry.zoo.catalog.manifest",
        "worldfoundry.evals.models.integrations",
        "worldfoundry.evals.models.pipelines.lifecycle",
        "worldfoundry.evals.models.runners.builtins",
        "worldfoundry.evals.models.runners.resolver",
        "worldfoundry.evals.models.runners.registry",
    ]
    script = (
        "import json, sys; "
        "import worldfoundry.evals.models; "
        f"watched = {watched_modules!r}; "
        "print(json.dumps([name for name in watched if name in sys.modules]))"
    )

    assert "from .integrations import" not in text
    assert "from .pipeline_runner import" not in text
    assert "from .runtime_runners import" not in text
    assert "from .resolver import" not in text
    assert "import_module" in text

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == []

    from worldfoundry.evals.models.runners.pipeline import WorldFoundryPipelineRunner
    from worldfoundry.evals.models.runners.pipeline import WorldFoundryPipelineRunner as CanonicalPipelineRunner
    from worldfoundry.evals.models.runners.registry import ModelRunnerRegistry
    from worldfoundry.evals.models.runners.registry import ModelRunnerRegistry as CanonicalRegistry

    assert WorldFoundryPipelineRunner is CanonicalPipelineRunner
    assert ModelRunnerRegistry is CanonicalRegistry


def test_model_catalog_uses_canonical_pipeline_runner_target() -> None:
    catalog_root = REPO_ROOT / "worldfoundry/data/models/catalog"
    legacy_target = "worldfoundry.evals.models." + "pipeline_runner:WorldFoundryPipelineRunner"
    canonical_target = "worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner"

    runner_targets = [
        line.strip()
        for path in catalog_root.rglob("*.yaml")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("runner_target:")
    ]

    assert runner_targets
    assert not any(legacy_target in line for line in runner_targets)
    assert any(canonical_target in line for line in runner_targets)


def test_retired_model_root_reexport_shims_are_not_reintroduced() -> None:
    models_root = REPO_ROOT / "worldfoundry/evals/models"
    retired = {
        "catalog_registry.py",
        "manifests.py",
        "pipeline_runner.py",
        "pipeline_" + "loading.py",
        "pipeline_" + "inference.py",
        "reference_" + "runner.py",
        "resolver.py",
        "runtime_runners.py",
        "zoo.py",
        "zoo_manifests.py",
        "zoo_policy.py",
        "zoo_registry.py",
        "zoo_variant_selection.py",
    }

    assert not sorted(path.name for path in models_root.glob("*.py") if path.name in retired)


def test_low_value_evaluation_subpackages_are_not_reintroduced() -> None:
    evaluation_root = REPO_ROOT / "worldfoundry/evals"

    forbidden_paths = [
        evaluation_root / "api" / "registry",
        evaluation_root / "cli",
        evaluation_root / "local_open_eval.py",
        evaluation_root / "mcp",
        evaluation_root / "models" / "pipeline_dispatch",
        evaluation_root / "reporting.py",
        evaluation_root / "reporting_format.py",
        evaluation_root / "reporting_markdown.py",
        evaluation_root / "reporting_run_browser.py",
        evaluation_root / "reporting_run_comparison.py",
        evaluation_root / "reporting_run_index.py",
        evaluation_root / "reporting_run_manifest.py",
        evaluation_root / "reporting_run_report.py",
        evaluation_root / "reporting_run_summary.py",
        evaluation_root / "reporting_runtime_evidence.py",
        evaluation_root / "reporting_compare.py",
        evaluation_root / "runtime",
        evaluation_root / "runtime.py",
        evaluation_root / "runtime_assets.py",
        evaluation_root / "runtime_conda.py",
        evaluation_root / "runtime_env.py",
        evaluation_root / "runtime_jobs.py",
        evaluation_root / "runtime_probes.py",
        evaluation_root / "runner",
        evaluation_root / "scorecard.py",
        evaluation_root / "tui",
        evaluation_root / "worldscore.py",
        evaluation_root / "runner" / "embodied",
        evaluation_root / "runner" / "metrics",
        evaluation_root / "tasks" / "zoo",
        evaluation_root / "tasks" / "zoo.py",
        evaluation_root / "utils",
        evaluation_root / "models" / "zoo",
    ]

    assert [str(path.relative_to(REPO_ROOT)) for path in forbidden_paths if path.exists()] == []


def test_old_evaluation_import_paths_are_not_reintroduced() -> None:
    old_markers = [
        "worldfoundry.evals.benchmarks",
        "worldfoundry.evals.cli",
        "worldfoundry.evals.local_open_eval",
        "worldfoundry.evals.mcp",
        "worldfoundry.evals.worldscore",
        "worldfoundry/evals/benchmarks",
        "worldfoundry/evals/cli",
        "worldfoundry/evals/mcp",
        "from ..benchmarks",
        "from .benchmarks",
        "worldfoundry.evals.api.registry.aliases",
        "worldfoundry.evals.api.registry.core",
        "worldfoundry.evals.models.pipeline_dispatch",
        "worldfoundry.evals.reporting_format",
        "worldfoundry.evals.reporting_markdown",
        "worldfoundry.evals.reporting_run_summary",
        "worldfoundry.evals.reporting_runtime_evidence",
        "worldfoundry.evals.runtime",
        "worldfoundry.evals.runtime.assets",
        "worldfoundry.evals.runtime.env",
        "worldfoundry.evals.tui",
        "worldfoundry.evals.tasks.run_embodied.normalizer",
        "worldfoundry.evals.tasks.run_embodied.metrics",
        "worldfoundry.evals.tasks.run_metrics.registry",
        "worldfoundry.evals.tasks." + "zoo.",
        "worldfoundry.evals.models." + "zoo.",
        "worldfoundry.evals.utils.paths",
    ]
    violations = []
    for path in (REPO_ROOT / "worldfoundry/evals").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in old_markers:
            if marker in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {marker}")

    assert violations == []


def test_docs_do_not_advertise_removed_evaluation_subpackages() -> None:
    forbidden_markers = [
        "worldfoundry/evals/runner/",
        "evaluation/runner/",
        "worldfoundry/evals/benchmark_zoo",
        "evaluation/benchmark_zoo",
        "worldfoundry/evals/cli",
        "worldfoundry/evals/mcp",
        "evaluation/cli",
        "evaluation/mcp",
        "worldfoundry/evals/model_zoo",
        "evaluation/model_zoo",
        "evaluation/models_schema.py",
        "evaluation/models_registry.py",
        "evaluation/models_manifests.py",
        "worldfoundry/evals/reporting.py",
        "worldfoundry/evals/reporting_run_",
        "worldfoundry/evals/runtime.py",
        "worldfoundry/evals/runtime_",
        "worldfoundry/evals/scorecard.py",
        "evals/reporting.py",
        "evaluation/reporting_compare.py",
        "evaluation/reporting_run_",
        "evaluation/runtime.py",
        "evaluation/runtime_",
        "evaluation/scorecard.py",
        '<Folder name="runner"',
        "worldfoundry/evals/tasks/" + "zoo.py",
        "evals/tasks/" + "zoo.py",
    ]
    docs_roots = [
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs",
        REPO_ROOT / "docs" / "design",
    ]
    docs_files = [
        REPO_ROOT / "README.md",
    ]

    violations = []
    for root in docs_roots:
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".md", ".mdx"}:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden_markers:
                if marker in text:
                    violations.append(f"{path.relative_to(REPO_ROOT)}: {marker}")
    for path in docs_files:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            if marker in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {marker}")

    assert violations == []
