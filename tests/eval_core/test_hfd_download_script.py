from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from worldfoundry.zoo.catalog import clear_model_zoo_registry_cache, load_model_zoo_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_embodied_asset_manifest_includes_gr00t_reasoner(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "setup" / "download_embodied_action_official_assets.py"
    spec = importlib.util.spec_from_file_location("download_embodied_action_official_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assets = module.default_assets(
        tmp_path / "hfd",
        tmp_path / "assets",
        tmp_path / "openpi",
        tmp_path / "repos",
    )
    by_asset = {asset.asset_id: asset for asset in assets if asset.model_id == "gr00t"}

    assert "nvidia/GR00T-N1.7-LIBERO" in by_asset
    assert "nvidia/Cosmos-Reason2-2B" in by_asset
    assert by_asset["nvidia/Cosmos-Reason2-2B"].revision == "9ce19a195e423419c349abfc86fd07178b230561"
    assert by_asset["nvidia/Cosmos-Reason2-2B"].metadata["access"] == "gated_nvidia_terms"


def test_embodied_asset_manifest_includes_giga_brain_0p1_checkpoint(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "setup" / "download_embodied_action_official_assets.py"
    spec = importlib.util.spec_from_file_location("download_embodied_action_official_assets_giga", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assets = module.default_assets(
        tmp_path / "hfd",
        tmp_path / "assets",
        tmp_path / "openpi",
        tmp_path / "repos",
    )
    by_asset = {asset.asset_id: asset for asset in assets if asset.model_id == "giga-brain-0"}

    assert "open-gigaai/GigaBrain-0-3.5B-Base" in by_asset
    assert "open-gigaai/GigaBrain-0.1-3.5B-Base" in by_asset
    assert by_asset["open-gigaai/GigaBrain-0.1-3.5B-Base"].metadata["demo"] == (
        "GigaBrain-0.1 official 3.5B base checkpoint"
    )


def test_embodied_asset_ready_rejects_metadata_only_gated_weights(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "setup" / "download_embodied_action_official_assets.py"
    spec = importlib.util.spec_from_file_location("download_embodied_action_official_assets_ready", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    local_dir = tmp_path / "hfd" / "gated--model"
    metadata_dir = local_dir / ".hfd"
    metadata_dir.mkdir(parents=True)
    (local_dir / "README.md").write_text("metadata is not a usable model\n", encoding="utf-8")
    (metadata_dir / "repo_metadata.json").write_text(
        json.dumps(
            {
                "siblings": [
                    {"rfilename": "README.md"},
                    {"rfilename": "model.safetensors"},
                ]
            }
        ),
        encoding="utf-8",
    )
    asset = module.Asset("blocked-model", "hf_model", "owner/gated-model", str(local_dir))

    assert module._asset_ready(asset) is False
    (local_dir / "model.safetensors").write_bytes(b"weights")
    assert module._asset_ready(asset) is True


def test_embodied_hf_download_plan_only_reports_planned_status(tmp_path: Path) -> None:
    path = REPO_ROOT / "scripts" / "setup" / "download_embodied_action_official_assets.py"
    spec = importlib.util.spec_from_file_location("download_embodied_action_official_assets_plan_only", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    local_dir = tmp_path / "hfd" / "owner--model"
    asset = module.Asset("blocked-model", "hf_model", "owner/model", str(local_dir))
    args = SimpleNamespace(plan_only=True, list=False, skip_existing=True, max_workers=1)

    result = module._download_hf(asset, args, {}, tmp_path / "logs")

    assert result["status"] == "planned"
    assert result["path_ready"] is False


def test_public_model_zoo_exposes_gated_foundation_dependencies() -> None:
    clear_model_zoo_registry_cache()
    registry = load_model_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog")

    giga_refs = {ref.hf_repo_id: ref for ref in registry.get("giga-brain-0").checkpoint_refs}
    assert giga_refs["google/paligemma-3b-pt-224"].requires_auth is True
    assert giga_refs["google/paligemma-3b-pt-224"].gated == "manual"

    gr00t_refs = {ref.hf_repo_id: ref for ref in registry.get("gr00t").checkpoint_refs}
    assert "nvidia/Cosmos-Reason2-2B" in gr00t_refs
    assert gr00t_refs["nvidia/Cosmos-Reason2-2B"].requires_auth is True
    assert gr00t_refs["nvidia/Cosmos-Reason2-2B"].gated == "auto"
    assert gr00t_refs["nvidia/Cosmos-Reason2-2B"].revision == "9ce19a195e423419c349abfc86fd07178b230561"
