from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.models.catalog import (
    ModelCatalogManifest,
    catalog_manifest_from_mapping,
    load_model_catalog_manifest,
    load_model_catalog_manifests,
    load_world_model_manifests_from_catalog,
)


def test_model_catalog_manifest_parses_target_schema(tmp_path: Path) -> None:
    path = tmp_path / "matrix-game-1.yaml"
    path.write_text(
        "\n".join(
            [
                "schema_version: 2",
                "model_id: matrix-game-1",
                "name: Matrix-Game 1.0",
                "family: Matrix-Game",
                "domain: world",
                "aliases:",
                "- matrix-game",
                "capabilities:",
                "  task_family: world_model",
                "  modalities:",
                "  - interactive-world-model",
                "  - action-control",
                "availability:",
                "  source: open_source",
                "  license: mit",
                "sources:",
                "  github:",
                "    url: https://github.com/SkyworkAI/Matrix-Game",
                "  huggingface:",
                "  - repo_id: Skywork/Matrix-Game",
                "    type: model",
                "checkpoints:",
                "  primary:",
                "    repo_id: Skywork/Matrix-Game",
                "    gated: false",
                "integration:",
                "  status: integrated",
                "  runtime_profile: matrix-game-1",
                "  pipeline_binding: matrix-game-1",
                "  runner: worldfoundry.pipeline",
                "evidence:",
                "  notes:",
                "  - verified fixture",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = load_model_catalog_manifest(path)
    public_manifest = manifest.to_world_model_manifest()

    assert isinstance(manifest, ModelCatalogManifest)
    assert manifest.model_id == "matrix-game-1"
    assert manifest.task_family == "world_model"
    assert manifest.modalities == ("interactive-world-model", "action-control")
    assert manifest.source_status == "open_source"
    assert public_manifest.model_id == "matrix-game-1"
    assert public_manifest.provider == "open_source"
    assert public_manifest.aliases == ("matrix-game",)
    assert public_manifest.supported_tasks == ("world_model",)
    assert "Skywork/Matrix-Game" in public_manifest.required_artifacts
    assert public_manifest.metadata["integration"]["pipeline_binding"] == "matrix-game-1"


def test_model_catalog_manifest_normalizes_legacy_model_zoo_shape() -> None:
    manifest = catalog_manifest_from_mapping(
        {
            "id": "legacy-video",
            "name": "Legacy Video",
            "task": "video_generation",
            "source_status": "confirmed_official_code",
            "official_repo_url": "https://github.com/example/legacy-video",
            "checkpoints": [{"repo_id": "example/legacy-video", "role": "primary"}],
            "integration": {"status": "planned"},
            "runtime_profile": "legacy-video",
            "pipeline_target": "pkg.module:Pipe",
            "runner_target": "pkg.runner:Runner",
        }
    )

    public_manifest = manifest.to_world_model_manifest()

    assert manifest.model_id == "legacy-video"
    assert manifest.source_status == "open_source"
    assert manifest.sources["github"]["url"] == "https://github.com/example/legacy-video"
    assert manifest.integration["pipeline_binding"] == "pkg.module:Pipe"
    assert public_manifest.capabilities == ("video_generation",)
    assert public_manifest.required_artifacts == ("example/legacy-video",)


def test_model_catalog_loader_reads_directory_and_exports_world_model_manifests(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    (root / "world_models").mkdir(parents=True)
    (root / "video").mkdir()
    (root / "world_models" / "alpha.yaml").write_text(
        "schema_version: 2\nmodel_id: alpha\nname: Alpha\ncapabilities:\n  task_family: world_model\n",
        encoding="utf-8",
    )
    (root / "video" / "beta.yaml").write_text(
        "id: beta\ntask: video_generation\nsource_status: open_source\n",
        encoding="utf-8",
    )

    manifests = load_model_catalog_manifests(root)
    public = load_world_model_manifests_from_catalog(root)

    assert [manifest.model_id for manifest in manifests] == ["alpha", "beta"]
    assert [manifest.model_id for manifest in public] == ["alpha", "beta"]
    assert public[0].supported_tasks == ("world_model",)
    assert public[1].provider == "open_source"


def test_model_catalog_loader_prefers_target_schema_over_legacy_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    (root / "world_models").mkdir(parents=True)
    (root / "world_models" / "alpha-legacy.yaml").write_text(
        "\n".join(
            [
                "id: alpha",
                "name: Alpha Legacy",
                "availability: open_source",
                "integration:",
                "  status: runtime_ported",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "world_models" / "alpha.yaml").write_text(
        "\n".join(
            [
                "schema_version: 2",
                "model_id: alpha",
                "name: Alpha Target",
                "capabilities:",
                "  task_family: world_model",
                "integration:",
                "  status: integrated",
                "  runtime_profile: alpha",
                "  pipeline_binding: alpha",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifests = load_model_catalog_manifests(root)

    assert [manifest.model_id for manifest in manifests] == ["alpha"]
    assert manifests[0].name == "Alpha Target"
    assert manifests[0].integration["pipeline_binding"] == "alpha"


def test_packaged_model_catalog_uses_target_matrix_game_manifest() -> None:
    manifests = [manifest for manifest in load_model_catalog_manifests() if manifest.model_id == "matrix-game-1"]

    assert len(manifests) == 1
    manifest = manifests[0]
    public_manifest = manifest.to_world_model_manifest()
    assert manifest.domain == "world"
    assert manifest.task_family == "world_model"
    assert manifest.integration["runtime_profile"] == "matrix-game-1"
    assert manifest.integration["pipeline_target"].endswith("pipeline_matrix_game_1:MatrixGame1Pipeline")
    assert public_manifest.metadata["integration"]["runner"] == "worldfoundry.pipeline"


def test_packaged_checkpoint_gated_vla_catalogs_use_canonical_integration_status() -> None:
    manifests = {manifest.model_id: manifest for manifest in load_model_catalog_manifests()}

    for model_id in ("ahawam", "giga-world-policy-0.5", "rise", "tinyvla"):
        assert manifests[model_id].integration["status"] == "integrated"
