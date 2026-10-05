from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from worldfoundry.evaluation.models.catalog import (
    CheckpointRef,
    DemoParitySpec,
    ModelSource,
    ModelVariantSpec,
    ModelZooEntry,
    is_in_tree_target,
    load_entries,
    model_zoo_entry_to_world_model_manifest,
    validate_in_tree_model_entry,
    validate_in_tree_model_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "worldfoundry"
DATA_ROOT = SOURCE_ROOT / "data"


def test_model_zoo_schema_uses_catalog_schema_as_canonical() -> None:
    import worldfoundry.evaluation.models.catalog.schema as catalog_schema

    legacy_path = SOURCE_ROOT / "evaluation" / "models" / "zoo_schema.py"

    assert catalog_schema.ModelZooEntry is ModelZooEntry
    assert catalog_schema.CheckpointRef is CheckpointRef
    assert catalog_schema.load_entries is load_entries
    assert not legacy_path.exists()


def test_model_zoo_entry_roundtrips_nested_dict_and_json() -> None:
    entry = ModelZooEntry(
        model_id="example-model",
        name="Example Model",
        tasks=("text-to-video", "image-to-video"),
        source=ModelSource(
            status="open_source",
            official_repo_url="https://example.invalid/repo",
            hf_repo_id="org/example",
            license="apache-2.0",
            requires_auth=False,
            notes=("source note",),
        ),
        checkpoint=CheckpointRef(
            hf_repo_id="org/example-checkpoint",
            revision="main",
            path="weights",
            filename="model.safetensors",
            estimated_size_gb=12.5,
            requires_auth=True,
        ),
        integration_status="integrated",
        install_profile="default",
        runner_target="worldfoundry.pipelines.example",
        pipeline_target="worldfoundry.pipelines.example.pipeline:ExamplePipeline",
        demo_parity=DemoParitySpec(
            status="verified",
            demo_command="conda run -n worldfoundry example",
            expected_artifacts=("tmp/example.mp4",),
        ),
        notes=("entry note",),
    )

    as_dict = entry.to_dict()
    assert as_dict["source"]["status"] == "open_source"
    assert as_dict["tasks"] == ["text-to-video", "image-to-video"]
    assert as_dict["checkpoint"]["estimated_size_gb"] == 12.5
    assert as_dict["pipeline_target"] == "worldfoundry.pipelines.example.pipeline:ExamplePipeline"
    assert as_dict["demo_parity"]["expected_artifacts"] == ["tmp/example.mp4"]

    from_dict = ModelZooEntry.from_dict(as_dict)
    assert from_dict == entry
    assert ModelZooEntry.from_json(entry.to_json()) == entry


def test_model_zoo_entry_accepts_flat_manifest_fields() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "model_id": "flat-example",
            "source_status": "api",
            "official_repo_url": "https://example.invalid/api",
            "hf_repo_id": "org/flat-example",
            "license": "unknown",
            "requires_auth": True,
            "estimated_size_gb": "2.75",
            "integration_status": "planned",
            "install_profile": "api",
            "runner_target": "worldfoundry.pipelines.flat",
            "pipeline_target": "worldfoundry.pipelines.flat:FlatPipeline",
            "demo_parity": {"status": "pending"},
            "demo_command": "conda run -n worldfoundry flat",
            "expected_artifacts": "tmp/flat.mp4",
            "notes": "flat note",
        }
    )

    assert entry.source.status == "api"
    assert entry.tasks == ()
    assert entry.official_repo_url == "https://example.invalid/api"
    assert entry.hf_repo_id == "org/flat-example"
    assert entry.license == "unknown"
    assert entry.requires_auth is True
    assert entry.estimated_size_gb == 2.75
    assert entry.integration_status == "planned"
    assert entry.install_profile == "api"
    assert entry.runner_target == "worldfoundry.pipelines.flat"
    assert entry.pipeline_target == "worldfoundry.pipelines.flat:FlatPipeline"
    assert entry.demo_parity.status == "pending"
    assert entry.demo_command == "conda run -n worldfoundry flat"
    assert entry.expected_artifacts == ("tmp/flat.mp4",)
    assert entry.notes == ("flat note",)


def test_model_zoo_entry_accepts_target_integration_fields() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "model_id": "target-model",
            "integration": {
                "status": "integrated",
                "runtime_profile": "target-runtime",
                "pipeline_binding": "target-binding",
                "runner": "worldfoundry.pipeline",
            },
        }
    )

    assert entry.integration_status == "integrated"
    assert entry.runtime_profile == "target-runtime"
    assert entry.pipeline_target is None
    assert entry.pipeline_binding == "target-binding"
    assert entry.runner_target == "worldfoundry.pipeline"


def test_model_zoo_entry_accepts_target_sources_and_checkpoint_mapping() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "schema_version": 2,
            "model_id": "target-model",
            "availability": {"source": "open_source", "license": "mit"},
            "sources": {
                "github": {"url": "https://github.com/example/target-model"},
                "huggingface": [{"repo_id": "org/target-model", "type": "model"}],
            },
            "checkpoints": {
                "primary": {
                    "repo_id": "org/target-model",
                    "revision": "abc123",
                    "gated": False,
                }
            },
        }
    )

    assert entry.source_status == "open_source"
    assert entry.official_repo_url == "https://github.com/example/target-model"
    assert entry.license == "mit"
    assert entry.hf_repo_ids == ("org/target-model",)
    assert entry.checkpoint.revision == "abc123"


def test_model_zoo_entry_normalizes_status_aliases_and_hf_urls() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "alias-example",
            "source_status": {
                "status": "confirmed_official_code_and_hf_data",
                "github": {"url": "https://github.com/example/alias-example"},
            },
            "official_sources": {
                "huggingface": ["https://huggingface.co/org/alias-example"],
            },
            "integration": {"status": "blocked_missing_runtime"},
            "demo_parity": {"status": "pending_checkpoint_and_demo"},
        }
    )

    assert entry.source.status == "open_source"
    assert entry.official_repo_url == "https://github.com/example/alias-example"
    assert entry.hf_repo_id == "org/alias-example"
    assert entry.integration_status == "blocked"
    assert entry.demo_parity.status == "pending"


def test_model_zoo_entry_does_not_treat_top_level_status_as_integration() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "ambiguous-status-model",
            "status": "integrated",
            "source_status": "confirmed_official_code",
        }
    )

    assert entry.source.status == "open_source"
    assert entry.integration_status == "planned"


def test_model_zoo_entry_preserves_multiple_checkpoint_refs() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "multi-checkpoint-model",
            "source_status": "confirmed_official_code",
            "checkpoint": {
                "repos": [
                    {"id": "org/model-a", "sha": "abc123", "license": "apache-2.0", "gated": False},
                    {"id": "org/model-b", "sha": "def456", "license": "mit", "gated": "auto"},
                ]
            },
        }
    )

    assert entry.hf_repo_id == "org/model-a"
    assert entry.hf_repo_ids == ("org/model-a", "org/model-b")
    assert entry.checkpoint_refs[0].revision == "abc123"
    assert entry.checkpoint_refs[1].license == "mit"
    assert entry.checkpoint_refs[1].requires_auth is True


def test_model_zoo_entry_skips_private_hf_sources_as_required_checkpoints() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "filtered-hf-sources",
            "official_sources": {
                "huggingface": [
                    {"repo_id": "org/public-model", "status": "confirmed"},
                    {"repo_id": "org/private-model", "status": "access_required_or_private"},
                ]
            },
        }
    )

    assert entry.hf_repo_ids == ("org/public-model",)
    assert [ref.hf_repo_id for ref in entry.checkpoint_refs] == ["org/public-model"]


def test_model_zoo_entry_preserves_task_contract() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "taskful-model",
            "tasks": ["text-to-video", "image-to-video"],
            "checkpoint": {"repos": [{"id": "org/taskful", "sha": "abc123"}]},
        }
    )

    assert entry.tasks == ("text-to-video", "image-to-video")
    assert entry.to_dict()["tasks"] == ["text-to-video", "image-to-video"]


def test_data_models_root_contains_only_model_scoped_yaml() -> None:
    data_root = DATA_ROOT
    models_root = data_root / "models"

    assert list(models_root.glob("*.yaml")) == []
    assert not (models_root / "bench_local_checkouts.yaml").exists()
    assert not (models_root / "runtime_cli_bench_roots.yaml").exists()
    assert not (models_root / "runtime_profiles" / "benchmark_zoo_official_eval.yaml").exists()
    assert not (data_root / "model_sources").exists()
    official_profiles = data_root / "benchmarks" / "runtime_profiles" / "official"
    assert official_profiles.is_dir()
    assert any(official_profiles.glob("*.yaml"))


def test_data_yaml_does_not_store_machine_local_absolute_paths() -> None:
    repo_root = REPO_ROOT
    data_root = DATA_ROOT
    forbidden_tokens = (
        "/mnt" + "/cpfs",
        "/mnt" + "/workspace",
        "/mnt" + "/world_foundational_model",
        "/home/",
        "/root/",
        "/share/project/bench",
        "/Users" + "/",
        "/private" + "/home",
        "yang" + "boxue",
    )
    hits = []

    for path in sorted((data_root / "models").rglob("*.yaml")) + sorted((data_root / "benchmarks").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden_tokens):
            hits.append(str(path.relative_to(repo_root)))

    assert hits == []


def test_model_zoo_entry_preserves_variant_level_contract() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "variant-model",
            "tasks": ["text-to-video", "image-to-video"],
            "provider": "hf_diffusers",
            "runtime_profile": "cuda-24gb",
            "min_vram_gb": 24,
            "variants": [
                {
                    "id": "variant-model-t2v",
                    "task": "text-to-video",
                    "provider": "official_repo",
                    "checkpoint_refs": [{"repo_id": "org/variant-t2v", "sha": "abc123"}],
                    "runner_target": "worldfoundry.pipelines.variant_t2v",
                    "runtime_profile": "t2v-smoke",
                    "min_vram_gb": 16,
                    "demo_parity": {"status": "pending_demo", "demo_command": ["python", "demo.py"]},
                    "runner_parity": {"status": "planned"},
                }
            ],
        }
    )

    assert entry.provider == "hf_diffusers"
    assert entry.runtime_profile == "cuda-24gb"
    assert entry.min_vram_gb == 24
    assert entry.variants[0].variant_id == "variant-model-t2v"
    assert entry.variants[0].task == "text-to-video"
    assert entry.variants[0].checkpoint_refs[0].hf_repo_id == "org/variant-t2v"
    assert entry.variants[0].demo_parity.status == "pending"
    assert entry.hf_repo_ids == ("org/variant-t2v",)


def test_model_zoo_entry_derives_source_variant_ids() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "source-variants",
            "variants": [
                {
                    "name": "LingBot-World-Base (Cam)",
                    "availability": "open_source",
                    "checkpoint": "robbyant/lingbot-world-base-cam",
                },
                {
                    "name": "LingBot-World-Fast",
                    "availability": "unknown",
                    "checkpoint": None,
                },
            ],
        }
    )

    assert entry.variants[0].variant_id == "lingbot-world-base-cam"
    assert entry.variants[0].checkpoint_refs[0].hf_repo_id == "robbyant/lingbot-world-base-cam"
    assert entry.variants[1].variant_id == "lingbot-world-fast"


def test_model_zoo_repo_manifests_all_load() -> None:
    catalog_root = DATA_ROOT / "models" / "catalog"
    manifest_paths = sorted(catalog_root.rglob("*.yaml"))

    loaded = {str(path.relative_to(catalog_root)): len(load_entries(path)) for path in manifest_paths}

    assert sum(count for path, count in loaded.items() if path.startswith("video/")) >= 10
    assert sum(count for path, count in loaded.items() if path.startswith("world_models/")) >= 10
    assert sum(count for path, count in loaded.items() if path.startswith("three_d_four_d/")) >= 10
    assert all(
        count == 1 or path == "hosted_api/video_world_apis.yaml"
        for path, count in loaded.items()
    )


def test_model_data_uses_per_model_yaml_manifests() -> None:
    models_root = DATA_ROOT / "models"
    catalog_root = models_root / "catalog"

    assert sorted(path.relative_to(catalog_root) for path in catalog_root.rglob("*.json")) == []
    assert all(
        len(load_entries(path)) == 1 or path.relative_to(catalog_root) == Path("hosted_api/video_world_apis.yaml")
        for path in catalog_root.rglob("*.yaml")
        if not path.name.startswith("_")
    )


def test_model_zoo_integrated_targets_are_in_tree() -> None:
    issues = validate_in_tree_model_registry(DATA_ROOT / "models" / "catalog")

    assert issues == ()


def test_model_zoo_policy_flags_external_runner_targets() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "external-runtime-model",
            "integration": {"status": "integrated"},
            "runner_target": "external_pkg.runner:Runner",
        }
    )

    issues = validate_in_tree_model_entry(entry)

    assert not is_in_tree_target("external_pkg.runner:Runner")
    assert issues[0].model_id == "external-runtime-model"
    assert issues[0].field == "runner_target"


def test_model_zoo_high_priority_variants_are_attached_to_correct_models() -> None:
    catalog_root = DATA_ROOT / "models" / "catalog"
    entries = {
        entry.model_id: entry
        for path in catalog_root.rglob("*.yaml")
        for entry in load_entries(path)
    }

    expected = {
        "hunyuanvideo": ["hunyuanvideo-t2v", "hunyuanvideo-i2v"],
        "wan2.2": ["wan2.2-t2v-a14b", "wan2.2-i2v-a14b", "wan2.2-ti2v-5b"],
        "hunyuanvideo-1.5": ["hunyuanvideo-1.5-t2v", "hunyuanvideo-1.5-i2v"],
        "ltx-2.x": [
            "ltx-2-t2v",
            "ltx-2-i2v",
            "ltx-2-v2v",
            "ltx-2.3-t2v",
            "ltx-2.3-i2v",
            "ltx-2.3-v2v",
        ],
        "mochi-1": ["mochi-1-preview-t2v"],
        "matrix-game-2": ["matrix-game-2-universal-action-validation"],
        "cosmos-predict-2.5": ["cosmos-predict-2.5-2b", "cosmos-predict-2.5-14b"],
        "cosmos-transfer-2.5": ["cosmos-transfer-2.5-2b"],
    }

    for model_id, variant_ids in expected.items():
        assert [variant.variant_id for variant in entries[model_id].variants] == variant_ids
        assert all(variant.checkpoint_refs for variant in entries[model_id].variants)


def test_model_zoo_entry_wires_runner_parity_command() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "runner-parity-model",
            "integration_status": "planned",
            "demo_parity": {
                "status": "verified",
                "demo_command": ["bash", "official_demo.sh"],
                "expected_artifacts": [{"path": "official.mp4", "video_probe": {"min_frames": 10}}],
            },
            "runner_parity": {
                "status": "verified",
                "demo_command": ["python", "worldfoundry_runner_demo.py"],
                "expected_artifacts": [
                    {"path": "worldfoundry.mp4", "video_probe": {"expected_frames": 81}}
                ],
            },
            "variants": [
                {
                    "id": "runner-parity-model-t2v",
                    "integration_status": "planned",
                    "runner_parity": {
                        "status": "verified",
                        "demo_command": ["python", "worldfoundry_runner_demo.py"],
                        "expected_artifacts": [
                            {"path": "worldfoundry.mp4", "video_probe": {"expected_frames": 81}}
                        ],
                    },
                }
            ],
        }
    )

    assert entry.demo_parity.status == "verified"
    assert entry.runner_parity.status == "verified"
    assert entry.runner_demo_command == ("python", "worldfoundry_runner_demo.py")
    assert entry.runner_expected_artifacts[0]["path"] == "worldfoundry.mp4"
    variant = entry.variants[0]
    assert variant.runner_parity.status == "verified"
    assert variant.runner_parity.demo_command == ("python", "worldfoundry_runner_demo.py")
    assert variant.runner_parity.expected_artifacts[0]["video_probe"]["expected_frames"] == 81


def test_model_zoo_entry_preserves_pending_runner_parity_notes() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "pending-runner-parity-model",
            "runner_parity": {
                "status": "planned",
                "blocked_reasons": ["official demo artifact not recorded"],
            },
            "variants": [{"id": "pending-runner-parity-model-t2v", "runner_parity": {"status": "pending"}}],
        }
    )

    assert entry.runner_parity.status == "pending"
    assert entry.runner_demo_command is None
    assert entry.runner_parity.notes == ("blocked: official demo artifact not recorded",)
    assert entry.variants[0].runner_parity.status == "pending"


def test_model_zoo_entry_converts_to_public_world_model_manifest() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "runner-model",
            "name": "Runner Model",
            "tasks": ["text-to-video"],
            "source_status": "confirmed_official_github",
            "official_repo_url": "https://github.com/example/runner-model",
            "checkpoint": {"repos": [{"id": "org/runner-model", "sha": "abc123"}]},
            "demo_parity": {
                "status": "pending_demo",
                "demo_command": ["python", "demo.py"],
                "expected_artifacts": ["demo.mp4"],
            },
        }
    )

    manifest = model_zoo_entry_to_world_model_manifest(entry)

    assert manifest.model_id == "runner-model"
    assert manifest.provider == "huggingface"
    assert manifest.capabilities == ("text-to-video",)
    assert manifest.supported_tasks == ("text-to-video",)
    assert manifest.required_artifacts == ("org/runner-model",)
    assert manifest.metadata["hf_repo_ids"] == ["org/runner-model"]
    assert manifest.metadata["demo_parity"]["demo_command"] == ["python", "demo.py"]


def test_model_zoo_entry_converts_variants_to_public_world_model_manifest() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "variant-manifest",
            "provider": "hf_diffusers",
            "variants": [
                {
                    "id": "variant-manifest-t2v",
                    "task": "text-to-video",
                    "integration_status": "integrated",
                    "runner_target": "tests.eval_core.contract_fixture:ContractFixtureRunner",
                    "pipeline_target": "worldfoundry.pipelines.variant:VariantPipeline",
                    "checkpoint_refs": [{"repo_id": "org/t2v", "sha": "abc123"}],
                    "runtime_profile": "short-video",
                },
                {
                    "id": "variant-manifest-i2v",
                    "task": "image-to-video",
                    "checkpoint_refs": [{"repo_id": "org/i2v", "sha": "def456"}],
                },
            ],
        }
    )

    manifest = model_zoo_entry_to_world_model_manifest(entry)

    assert manifest.provider == "hf_diffusers"
    assert manifest.capabilities == ("text-to-video", "image-to-video")
    assert manifest.output_artifacts == ("generated_video",)
    assert manifest.required_artifacts == ("org/t2v", "org/i2v")
    assert manifest.metadata["variant_ids"] == ["variant-manifest-t2v", "variant-manifest-i2v"]
    assert manifest.metadata["integrated_variant_ids"] == ["variant-manifest-t2v"]
    assert manifest.metadata["runnable_runner_variant_ids"] == ["variant-manifest-t2v"]
    assert manifest.metadata["default_variant_id"] == "variant-manifest-t2v"
    assert manifest.metadata["default_runner_target"] == "tests.eval_core.contract_fixture:ContractFixtureRunner"
    assert manifest.metadata["default_pipeline_target"] == "worldfoundry.pipelines.variant:VariantPipeline"
    assert manifest.metadata["default_runtime_profile"] == "short-video"
    assert manifest.metadata["default_integration_status"] == "integrated"
    assert manifest.metadata["default_verification_status"] == "not_applicable"
    assert manifest.metadata["runner_entry_kind"] == "runnable_runner"
    assert manifest.metadata["runnable_runner"] is True
    assert manifest.metadata["runner_parity"] == {
        "demo_command": None,
        "expected_artifacts": [],
        "notes": [],
        "status": "not_applicable",
    }
    assert manifest.metadata["integration"] == {
        "status": "planned",
        "verification_status": "not_applicable",
    }
    assert manifest.metadata["variants"][0]["runtime_profile"] == "short-video"
    assert manifest.metadata["variants"][0]["pipeline_target"] == "worldfoundry.pipelines.variant:VariantPipeline"
    assert manifest.metadata["variants"][0]["integration_status"] == "integrated"
    assert manifest.metadata["variants"][0]["runner_entry_kind"] == "runnable_runner"
    assert manifest.metadata["variants"][0]["runnable_runner"] is True
    assert manifest.metadata["variants"][1]["runner_entry_kind"] == "listed_only"


def test_model_zoo_entry_distinguishes_listed_candidate_and_runnable_runner_entries() -> None:
    listed = ModelZooEntry.from_dict({"id": "listed-model", "source_status": "confirmed_official_code"})
    candidate = ModelZooEntry.from_dict(
        {
            "id": "candidate-model",
            "integration_status": "planned",
            "runner_target": "tests.eval_core.contract_fixture:ContractFixtureRunner",
            "runner_parity": {"status": "pending"},
        }
    )
    runnable = ModelZooEntry.from_dict(
        {
            "id": "runnable-model",
            "integration_status": "integrated",
            "runner_target": "tests.eval_core.contract_fixture:ContractFixtureRunner",
            "runner_parity": {"status": "verified"},
        }
    )

    assert listed.runner_entry_kind == "listed_only"
    assert listed.is_runnable_runner_entry is False
    assert candidate.runner_entry_kind == "runner_candidate"
    assert candidate.verification_status == "pending"
    assert candidate.is_runnable_runner_entry is False
    assert runnable.runner_entry_kind == "runnable_runner"
    assert runnable.verification_status == "verified"
    assert runnable.is_runnable_runner_entry is True


def test_model_zoo_entry_loads_runner_parity() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "runner-parity-example",
            "source_status": "confirmed_official_code",
            "runner_parity": {
                "status": "planned",
                "demo_command": "conda run -n worldfoundry worldfoundry-runner-demo",
                "expected_artifacts": "tmp/runner.mp4",
                "blocked_reasons": ["official demo artifact not recorded"],
            },
        }
    )

    assert entry.runner_parity.status == "pending"
    assert entry.runner_demo_command == "conda run -n worldfoundry worldfoundry-runner-demo"
    assert entry.runner_expected_artifacts == ("tmp/runner.mp4",)
    assert entry.runner_parity.notes == ("blocked: official demo artifact not recorded",)


def test_model_zoo_entry_preserves_list_demo_command() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "list-command",
            "source_status": "confirmed_official_code",
            "demo_parity": {
                "status": "pending_demo",
                "demo_command": ["bash", "scripts/model_zoo/demo.sh"],
                "expected_artifacts": ["tmp/demo.mp4"],
            },
        }
    )

    assert entry.demo_command == ("bash", "scripts/model_zoo/demo.sh")
    assert entry.to_dict()["demo_parity"]["demo_command"] == ["bash", "scripts/model_zoo/demo.sh"]


def test_model_zoo_entry_preserves_artifact_contract_dict() -> None:
    entry = ModelZooEntry.from_dict(
        {
            "id": "artifact-contract",
            "source_status": "confirmed_official_code",
            "demo_parity": {
                "status": "pending_demo",
                "demo_command": ["bash", "demo.sh"],
                "expected_artifacts": [{"path": "tmp/demo.mp4", "sha256": "0" * 64}],
            },
        }
    )

    assert entry.expected_artifacts == ({"path": "tmp/demo.mp4", "sha256": "0" * 64},)
    assert entry.to_dict()["demo_parity"]["expected_artifacts"] == [{"path": "tmp/demo.mp4", "sha256": "0" * 64}]


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (ModelSource, {"status": "public"}, "ModelSource.status"),
        (ModelVariantSpec, {"variant_id": ""}, "ModelVariantSpec.variant_id"),
        (ModelVariantSpec, {"variant_id": "bad", "integration_status": "done"}, "integration_status"),
        (ModelZooEntry, {"model_id": "bad", "integration_status": "done"}, "integration_status"),
        (DemoParitySpec, {"status": "matching"}, "DemoParitySpec.status"),
    ],
)
def test_status_fields_validate_known_enum_strings(factory: object, kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)  # type: ignore[operator]


def test_model_zoo_schema_imports_are_stdlib_only() -> None:
    schema_path = SOURCE_ROOT / "evaluation" / "models" / "catalog" / "schema.py"
    allowed_modules = set(sys.stdlib_module_names) | {"__future__"}

    tree = ast.parse(schema_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            modules = {node.module.split(".", 1)[0]} if node.module else set()
        else:
            continue

        unexpected = modules - allowed_modules
        assert unexpected == set(), f"{schema_path} imports {unexpected}"
