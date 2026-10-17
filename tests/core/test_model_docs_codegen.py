"""Public model documentation stays consistent with executable CLI contracts."""

from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path
from types import ModuleType

import pytest

from worldfoundry.cli.main import _build_parser
from worldfoundry.cli.model_run import load_model_run_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs/fumadocs"


@pytest.fixture(scope="module")
def recipes():
    return json.loads((DOCS / "lib/model-recipes-data.json").read_text())["recipes"]


@pytest.fixture
def generators(monkeypatch):
    scripts = DOCS / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import generate_model_home_prose as prose

    spec = importlib.util.spec_from_file_location("model_recipes_generator", scripts / "generate-model-recipes.py")
    assert spec is not None and spec.loader is not None
    commands = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(commands)
    return prose, commands


def test_generated_commands_parse_for_every_model_variant_and_task(recipes, generators):
    _, commands = generators
    failures = []
    checked = set()
    for recipe in recipes:
        for task in recipe["inferenceTasks"]:
            for model_id in task["variantIds"] or [recipe["id"]]:
                identity = (model_id, task["id"])
                if identity in checked:
                    continue
                checked.add(identity)
                try:
                    schema = load_model_run_schema(model_id, task_id=task["id"])
                    argv = shlex.split(commands.direct_run_command(model_id, task).replace("\\\n", ""))[1:]
                    _build_parser(schema).parse_args(argv)
                except (ValueError, SystemExit) as exc:
                    failures.append((identity, str(exc)))
    assert checked
    assert failures == []


def test_published_model_launch_commands_parse(recipes):
    failures = []
    for recipe in recipes:
        argv = shlex.split(recipe["commands"]["run"].replace("\\\n", ""))[1:]
        model_id = argv[1]
        task_id = argv[argv.index("--pipeline.task-profile") + 1]
        try:
            schema = load_model_run_schema(model_id, task_id=task_id)
            _build_parser(schema).parse_args(argv)
        except (ValueError, SystemExit) as exc:
            failures.append((recipe["id"], str(exc)))
    assert failures == []


def test_opening_uses_the_selected_variant_capability(recipes, generators):
    prose, _ = generators
    sana = next(recipe for recipe in recipes if recipe["id"] == "sana")
    assert "still image" in prose.family_opening(sana, "en")
    assert "文生视频" not in prose.family_opening(sana, "zh")
    assert "--pipeline.task-profile interactive-video" in prose.run_command_sentence(sana, "en")


def test_shared_profile_validation_does_not_verify_untested_variants(recipes, generators):
    prose, _ = generators
    sana = next(recipe for recipe in recipes if recipe["id"] == "sana")
    assert "— Verified" not in prose.variants_paragraph(sana, "en")
    assert "：已验证" not in prose.variants_paragraph(sana, "zh")
    shared = {
        "id": "shared-family",
        "variants": [
            {
                "id": name,
                "runtimeProfile": "shared",
                "runtimeStatus": "all_released_checkpoints_gpu_validated",
                "status": "integrated",
            }
            for name in ("first", "second")
        ],
    }
    assert prose.variants_paragraph(shared, "en").count("— Verified") == 2


def test_checkpoint_structure_validation_is_static(generators):
    generators
    from model_home_status import classify_token

    assert classify_token("native_checkpoint_structure_validated") == "static"
    assert classify_token("native_checkpoint_structure_verified") == "static"


def test_cli_profiles_are_separate_from_capability_names(recipes, generators):
    prose, _ = generators
    sana = next(recipe for recipe in recipes if recipe["id"] == "sana")
    assert "text-to-image" in prose.recorded_task_ids(sana)
    assert "interactive-video" not in prose.recorded_task_ids(sana)
    assert "interactive-video" in prose.task_profile_sentence(sana, "en")
    assert "`text-to-image`" not in prose.task_profile_sentence(sana, "en")


def test_default_launch_contract_uses_its_variant_artifacts(recipes, generators):
    prose, _ = generators
    wan = next(recipe for recipe in recipes if recipe["id"] == "wan2.2")
    contract = prose.contract_paragraph(wan, "en")
    assert "wan2.2-t2v-a14b.mp4" in contract
    assert "wan2.2-ti2v-5b-1280x704-121f.mp4" not in contract
    assert "required `image`" not in contract.lower()


def test_model_page_generators_share_the_same_article_template(recipes, generators):
    prose, _ = generators
    from model_page_mdx import render_model_page

    recipe = next(item for item in recipes if item["id"] == "hy-worldplay")
    for locale in ("en", "zh"):
        assert render_model_page(recipe, locale) == prose.render_page(recipe, locale, page_source_override="generated")


def test_documented_python_version_matches_the_selected_installer(generators):
    _, commands = generators
    unified = commands.load_yaml(commands.ENVIRONMENT_ROOT / "_unified.yaml")
    legacy = {
        "env_name": "worldfoundry-unified-cu128",
        "python": "3.10",
        "conda_packages": ["python=3.10", "pip"],
    }
    runtime = commands.runtime_data({}, None, None, "scope", legacy, None, None, unified)
    assert runtime["python"] == unified["python"]
    assert f"python={unified['python']}" in runtime["condaPackages"]

    dedicated = {**legacy, "env_name": "worldfoundry-dedicated-model"}
    runtime = commands.runtime_data({}, None, None, "dedicated-model", dedicated, None, None, unified)
    assert runtime["python"] == "3.10"
    assert "python=3.10" in runtime["condaPackages"]


@pytest.mark.parametrize("directory", ["ckpt", "ckpts"])
def test_documentation_normalizes_adjacent_checkpoint_defaults(
    generators: tuple[ModuleType, ModuleType], directory: str
) -> None:
    _, commands = generators
    checkpoint = commands.ROOT.parent / directory / "mt5-xxl"
    assert commands.portable_doc_value(str(checkpoint)) == f"{commands.CANONICAL_HOME}/checkpoints/mt5-xxl"


@pytest.mark.parametrize(
    ("model_id", "field", "checkpoint_name"),
    [
        ("lagernvs", "model_repo", "lagernvs_general_512"),
        ("open-sora-plan", "text_encoder_name_1", "mt5-xxl"),
    ],
)
def test_checkpoint_defaults_are_resolved_inside_the_documentation_scope(
    generators: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    field: str,
    checkpoint_name: str,
) -> None:
    from worldfoundry.core.io.paths import checkpoint_root_path
    from worldfoundry.studio.inference import catalog

    _, commands = generators
    monkeypatch.setattr(catalog, "_checkpoint_model_ref", lambda *names, **kwargs: "installed/model")
    monkeypatch.setattr(catalog, "_prefer_existing_model_ref", lambda *names: "installed/model")
    defaults = catalog.CURATED_OVERRIDES[model_id]["default_call_kwargs"]
    assert callable(defaults)
    assert defaults()[field] == "installed/model"
    with commands._canonical_catalog_defaults():
        assert defaults()[field] == str(checkpoint_root_path(checkpoint_name))
    assert defaults()[field] == "installed/model"


def test_documentation_defaults_ignore_existing_checkpoints(
    generators: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from worldfoundry.core.io.paths import checkpoint_root_path
    from worldfoundry.studio.inference import catalog

    _, commands = generators
    checkpoint = tmp_path / "installed-model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    missing = tmp_path / "declared-model"
    monkeypatch.setattr(catalog, "_cache_candidates", lambda *names: [str(checkpoint)])

    assert catalog._checkpoint_model_ref("model", fallback="owner/model") == str(checkpoint)
    assert catalog._prefer_existing_model_ref(str(missing), str(checkpoint)) == str(checkpoint)
    assert catalog._local_root_with_files([checkpoint], ["config.json"]) == str(checkpoint)

    with commands._canonical_catalog_defaults():
        assert catalog._checkpoint_model_ref("model", fallback="owner/model") == "owner/model"
        assert catalog._checkpoint_model_ref("model") == str(checkpoint_root_path("model"))
        assert catalog._prefer_existing_model_ref(str(missing), str(checkpoint)) == str(missing)
        assert catalog._local_root_with_files([checkpoint], ["config.json"]) == ""

    assert catalog._checkpoint_model_ref("model", fallback="owner/model") == str(checkpoint)
    assert catalog._prefer_existing_model_ref(str(missing), str(checkpoint)) == str(checkpoint)
    assert catalog._local_root_with_files([checkpoint], ["config.json"]) == str(checkpoint)
