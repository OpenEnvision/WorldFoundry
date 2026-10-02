from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "model_zoo" / "download_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("worldfoundry_download_checkpoints", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
download_checkpoints = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_checkpoints
SPEC.loader.exec_module(download_checkpoints)


def test_manifest_and_command_preserve_recipe_specific_filename(tmp_path: Path) -> None:
    manifest = download_checkpoints.ModelManifest(
        model_id="memory-recipe",
        path=tmp_path / "memory-recipe.yaml",
        data={
            "id": "memory-recipe",
            "checkpoint_refs": [
                {
                    "repo_id": "Example/Memory",
                    "filename": "context_k1/epoch-0.safetensors",
                },
                {"repo_id": "Example/Backbone", "revision": "a" * 40},
            ],
        },
    )

    assert manifest.hf_repo_files == {
        "Example/Memory": ["context_k1/epoch-0.safetensors"]
    }
    command = download_checkpoints.build_download_command(
        "Example/Memory",
        tmp_path,
        downloader=["hf", "download"],
        filenames=manifest.hf_repo_files["Example/Memory"],
        max_workers=1,
    )
    assert command[:4] == [
        "hf",
        "download",
        "Example/Memory",
        "context_k1/epoch-0.safetensors",
    ]


def test_local_check_requires_the_model_specific_file(tmp_path: Path) -> None:
    direct_repo = tmp_path / "Example--Memory"
    required = direct_repo / "context_k1" / "epoch-0.safetensors"
    required.parent.mkdir(parents=True)
    required.write_bytes(b"checkpoint")

    ready = download_checkpoints.check_local_checkpoint(
        "Example/Memory",
        tmp_path,
        required_files=["context_k1/epoch-0.safetensors"],
    )
    assert ready["ready"] is True
    assert ready["direct_hfd_missing_required_files"] == []

    missing = download_checkpoints.check_local_checkpoint(
        "Example/Memory",
        tmp_path,
        required_files=["context_k20/epoch-0.safetensors"],
    )
    assert missing["ready"] is False
    assert missing["direct_hfd_missing_required_files"] == [
        "context_k20/epoch-0.safetensors"
    ]


def test_hf_filename_rejects_path_traversal() -> None:
    assert download_checkpoints._normalize_hf_filename("../secret.safetensors") is None
