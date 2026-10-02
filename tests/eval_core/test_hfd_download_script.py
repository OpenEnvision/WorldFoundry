from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from worldfoundry.evaluation.models.catalog import clear_model_zoo_registry_cache, load_model_zoo_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_download_hfd_models_exposes_gr00t_gated_reasoner(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download_hfd_models.sh"),
            "--download-root",
            str(tmp_path / "hfd"),
            "--list",
            "gr00t",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0
    assert "nvidia/GR00T-N1.7-LIBERO" in completed.stdout
    assert "nvidia/Cosmos-Reason2-2B" in completed.stdout
    manifest = tmp_path / "hfd" / "model_paths.tsv"
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "gr00t\tpolicy_checkpoint_path\tnvidia/GR00T-N1.7-LIBERO" in manifest_text
    assert "gr00t\treasoner_model_path\tnvidia/Cosmos-Reason2-2B" in manifest_text


def test_download_hfd_models_exposes_giga_brain_0p1_checkpoint(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download_hfd_models.sh"),
            "--download-root",
            str(tmp_path / "hfd"),
            "--list",
            "giga-brain-0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0
    assert "open-gigaai/GigaBrain-0-3.5B-Base" in completed.stdout
    assert "open-gigaai/GigaBrain-0.1-3.5B-Base" in completed.stdout
    manifest = tmp_path / "hfd" / "model_paths.tsv"
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "giga-brain-0\tpretrained_model_path\topen-gigaai/GigaBrain-0-3.5B-Base" in manifest_text
    assert "giga-brain-0\tpretrained_model_0p1_path\topen-gigaai/GigaBrain-0.1-3.5B-Base" in manifest_text


def test_download_hfd_models_exposes_show_o_runtime_dependencies(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "download_hfd_models.sh"),
            "--download-root",
            str(tmp_path / "hfd"),
            "--list",
            "show-o",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0
    manifest = tmp_path / "hfd" / "model_paths.tsv"
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "show-o\tpretrained_model_path\tshowlab/show-o" in manifest_text
    assert "show-o\tvq_model_path\tshowlab/magvitv2" in manifest_text
    assert "show-o\tllm_model_path\tmicrosoft/phi-1_5" in manifest_text


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
