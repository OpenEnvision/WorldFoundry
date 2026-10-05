from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import hashlib
from types import SimpleNamespace

import pytest

from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.materialize import NativeCheckpointResolver
from worldfoundry.base_models.diffusion_model.recipes.sana import sana_recipe
from worldfoundry.base_models.diffusion_model.recipes.wan import (
    WAN21_T2V_1P3B_FILE_SHA256,
    WAN21_T2V_1P3B_FILE_SIZE_BYTES,
    WAN21_T2V_1P3B_REVISION,
    wan21_t2v_1p3b_recipe,
)
from worldfoundry.core.io.paths import resolve_local_hf_model_path


def test_checkpoint_integrity_audit_accepts_exact_local_file(tmp_path) -> None:
    payload = b"immutable training checkpoint"
    checkpoint_file = tmp_path / "model.safetensors"
    checkpoint_file.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    spec = CheckpointSpec(
        source=tmp_path,
        files=(checkpoint_file.name,),
        file_sha256={checkpoint_file.name: digest},
        file_size_bytes={checkpoint_file.name: len(payload)},
    )

    materialized = NativeCheckpointResolver().materialize(spec)

    assert materialized.paths == (checkpoint_file,)


def test_checkpoint_integrity_audit_rejects_tampering(tmp_path) -> None:
    checkpoint_file = tmp_path / "model.safetensors"
    checkpoint_file.write_bytes(b"changed")
    spec = CheckpointSpec(
        source=tmp_path,
        files=(checkpoint_file.name,),
        file_sha256={checkpoint_file.name: hashlib.sha256(b"expected").hexdigest()},
        file_size_bytes={checkpoint_file.name: len(b"changed")},
    )

    with pytest.raises(ValueError, match="SHA-256 audit failed"):
        NativeCheckpointResolver().materialize(spec)


def test_checkpoint_integrity_metadata_must_use_safe_resource_paths() -> None:
    with pytest.raises(ValueError, match="unsafe relative paths"):
        CheckpointSpec(
            source="unused",
            files=("model.safetensors",),
            resource_sha256={"../config.json": "0" * 64},
        )


def test_checkpoint_integrity_audits_required_sidecar_resources(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    weights.write_bytes(b"weights")
    config.write_bytes(b"configuration")
    spec = CheckpointSpec(
        source=tmp_path,
        files=(weights.name,),
        resource_sha256={config.name: hashlib.sha256(b"configuration").hexdigest()},
        resource_size_bytes={config.name: len(b"configuration")},
    )

    NativeCheckpointResolver().materialize(spec)
    config.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size audit failed"):
        NativeCheckpointResolver().materialize(spec)


def test_local_hugging_face_resolution_honors_the_requested_snapshot_revision(
    tmp_path,
) -> None:
    repository = tmp_path / "models--owner--model"
    first = repository / "snapshots" / ("a" * 40)
    second = repository / "snapshots" / ("b" * 40)
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "config.json").write_text("first", encoding="utf-8")
    (second / "config.json").write_text("second", encoding="utf-8")
    refs = repository / "refs"
    refs.mkdir()
    (refs / "main").write_text("b" * 40, encoding="utf-8")
    env = {
        "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "checkpoints"),
        "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
    }

    resolved = resolve_local_hf_model_path(
        repository,
        required_files=("config.json",),
        revision="a" * 40,
        env=env,
    )

    assert resolved == first.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_local_hf_model_path(
            repository,
            required_files=("config.json",),
            revision="c" * 40,
            env=env,
        )


def test_local_hugging_face_resolution_scans_the_default_hub_cache(tmp_path) -> None:
    revision = "c" * 40
    cache_root = tmp_path / "xdg" / "huggingface" / "hub"
    snapshot = (
        cache_root
        / "models--owner--model"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("configuration", encoding="utf-8")

    resolved = resolve_local_hf_model_path(
        "owner/model",
        required_files=("config.json",),
        revision=revision,
        env={
            "XDG_CACHE_HOME": str(tmp_path / "xdg"),
            "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "checkpoints"),
            "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
        },
    )

    assert resolved == snapshot.resolve()


def test_local_hugging_face_resolution_scans_plural_checkpoint_sibling(tmp_path) -> None:
    repository = tmp_path / "ckpts" / "owner--model"
    repository.mkdir(parents=True)
    (repository / "config.json").write_text("configuration", encoding="utf-8")
    (repository / "model.safetensors").write_bytes(b"weights")

    resolved = resolve_local_hf_model_path(
        "owner/model",
        required_files=("config.json", "model.safetensors"),
        env={
            "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "ckpt"),
            "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
        },
    )

    assert resolved == repository.resolve()


def test_local_hugging_face_resolution_ignores_stale_download_cache_partials(tmp_path) -> None:
    repository = tmp_path / "owner--model"
    repository.mkdir()
    (repository / "config.json").write_text("configuration", encoding="utf-8")
    cache = repository / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "stale-weight.incomplete").write_bytes(b"old partial download")
    env = {
        "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "checkpoints"),
        "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
    }

    resolved = resolve_local_hf_model_path(
        repository,
        required_files=("config.json",),
        env=env,
    )

    assert resolved == repository.resolve()

    (repository / "active-weight.incomplete").write_bytes(b"active partial download")
    with pytest.raises(FileNotFoundError):
        resolve_local_hf_model_path(
            repository,
            required_files=("config.json",),
            env=env,
        )


def test_local_hugging_face_resolution_matches_hub_cache_env_precedence(
    tmp_path,
) -> None:
    revision = "d" * 40

    def snapshot_at(hub_root, marker: str):
        snapshot = (
            hub_root
            / "models--owner--model"
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(marker, encoding="utf-8")
        return snapshot

    xdg_snapshot = snapshot_at(
        tmp_path / "xdg" / "huggingface" / "hub",
        "xdg",
    )
    home_snapshot = snapshot_at(tmp_path / "hf-home" / "hub", "home")
    legacy_snapshot = snapshot_at(tmp_path / "legacy-hub", "legacy")
    explicit_snapshot = snapshot_at(tmp_path / "explicit-hub", "explicit")
    base_env = {
        "XDG_CACHE_HOME": str(tmp_path / "xdg"),
        "HF_HOME": str(tmp_path / "hf-home"),
        "HUGGINGFACE_HUB_CACHE": str(tmp_path / "legacy-hub"),
        "HF_HUB_CACHE": str(tmp_path / "explicit-hub"),
        "WORLDFOUNDRY_CKPT_DIR": str(tmp_path / "checkpoints"),
        "WORLDFOUNDRY_HFD_ROOT": str(tmp_path / "hfd"),
    }

    def resolve(env):
        return resolve_local_hf_model_path(
            "owner/model",
            required_files=("config.json",),
            revision=revision,
            env=env,
        )

    assert resolve(base_env) == explicit_snapshot.resolve()
    without_explicit = {name: value for name, value in base_env.items() if name != "HF_HUB_CACHE"}
    assert resolve(without_explicit) == legacy_snapshot.resolve()
    without_legacy = {
        name: value
        for name, value in without_explicit.items()
        if name != "HUGGINGFACE_HUB_CACHE"
    }
    assert resolve(without_legacy) == home_snapshot.resolve()
    without_home = {name: value for name, value in without_legacy.items() if name != "HF_HOME"}
    assert resolve(without_home) == xdg_snapshot.resolve()

    home_default_snapshot = snapshot_at(
        tmp_path / "injected-home" / ".cache" / "huggingface" / "hub",
        "home-default",
    )
    without_xdg = {
        name: value
        for name, value in without_home.items()
        if name != "XDG_CACHE_HOME"
    }
    without_xdg["HOME"] = str(tmp_path / "injected-home")
    assert resolve(without_xdg) == home_default_snapshot.resolve()


def test_sana_recipe_binds_the_checkpoint() -> None:
    checkpoint = sana_recipe("sana-600m-512px").checkpoints["dit"]

    assert checkpoint.revision == "a189ee0678ba4b806c2091396fa07dedf8946913"
    assert checkpoint.file_sha256 == {
        "checkpoints/Sana_600M_512px_MultiLing.pth": (
            "cbb880000c9c2594f00e16fff43be67733f0e2e75a4391e4897ffdbe74a19881"
        )
    }
    assert checkpoint.file_size_bytes == {"checkpoints/Sana_600M_512px_MultiLing.pth": 2371119642}


def test_sana_sprint_recipes_pin_independent_student_and_teacher_assets() -> None:
    expected = {
        "sana-sprint-600m-1024px": {
            "student_revision": "268b7e32816200db48e0b0a8939f32397ae6889f",
            "student_file": "checkpoints/Sana_Sprint_0.6B_1024px.pth",
            "student_sha256": "fd1f4497ba127cac3da5cc6832306cc8b554c187ce6f86ab9b7e0064d0f1b503",
            "student_size": 2381713754,
            "teacher_revision": "141411fcb8360dc1e6ae91e7c7ed69371adb058c",
            "teacher_file": "checkpoints/Sana_Sprint_0.6B_1024px_teacher.pth",
            "teacher_sha256": "1afe15602c9a60d32510aed0a83e3d75b9c289aa41d202f6444e1892fc14cc1b",
            "teacher_size": 2375214166,
        },
        "sana-sprint-1600m-1024px": {
            "student_revision": "9f6866d6962d5d91d213f113e162cb18bffe741c",
            "student_file": "checkpoints/Sana_Sprint_1.6B_1024px.pth",
            "student_sha256": "180d3c8cfb1e85e907c7f02f431900343d5768ee86a2f20a5af56f9e3b7a2862",
            "student_size": 6453060642,
            "teacher_revision": "f111cf761e05a1ad460cd5fb528c1a107327b459",
            "teacher_file": "checkpoints/Sana_Sprint_1.6B_1024px_teacher.pth",
            "teacher_sha256": "9829ee64eb6ae3cb372e0f7bd2b62372fe6f106ddf0bafcafa99356247d6a548",
            "teacher_size": 6430670590,
        },
    }
    for model_id, values in expected.items():
        recipe = sana_recipe(model_id)
        student = recipe.checkpoints["dit"]
        teacher = recipe.checkpoints["teacher"]
        assert student.revision == values["student_revision"]
        assert student.files == (values["student_file"],)
        assert student.file_sha256 == {values["student_file"]: values["student_sha256"]}
        assert student.file_size_bytes == {values["student_file"]: values["student_size"]}
        assert teacher.revision == values["teacher_revision"]
        assert teacher.files == (values["teacher_file"],)
        assert teacher.file_sha256 == {values["teacher_file"]: values["teacher_sha256"]}
        assert teacher.file_size_bytes == {values["teacher_file"]: values["teacher_size"]}


def test_wan_recipe_binds_every_large_asset_and_tokenizer_binary() -> None:
    recipe = wan21_t2v_1p3b_recipe()

    assert all(checkpoint.revision == WAN21_T2V_1P3B_REVISION for checkpoint in recipe.checkpoints.values())
    assert recipe.checkpoints["dit"].file_sha256 == {
        "diffusion_pytorch_model.safetensors": WAN21_T2V_1P3B_FILE_SHA256["diffusion_pytorch_model.safetensors"]
    }
    assert recipe.checkpoints["text-encoder"].file_sha256 == {
        "models_t5_umt5-xxl-enc-bf16.pth": WAN21_T2V_1P3B_FILE_SHA256["models_t5_umt5-xxl-enc-bf16.pth"]
    }
    assert recipe.checkpoints["vae"].file_sha256 == {"Wan2.1_VAE.pth": WAN21_T2V_1P3B_FILE_SHA256["Wan2.1_VAE.pth"]}
    assert recipe.checkpoints["tokenizer"].file_sha256 == {
        name: WAN21_T2V_1P3B_FILE_SHA256[name]
        for name in (
            "google/umt5-xxl/spiece.model",
            "google/umt5-xxl/tokenizer.json",
        )
    }
    for checkpoint in recipe.checkpoints.values():
        assert checkpoint.file_size_bytes == {name: WAN21_T2V_1P3B_FILE_SIZE_BYTES[name] for name in checkpoint.files}


def test_local_checkpoint_override_preserves_the_content_audit(tmp_path) -> None:
    expected = b"audited override"
    local_file = tmp_path / "renamed.pth"
    local_file.write_bytes(expected)
    remote_name = "checkpoints/original.pth"
    default = CheckpointSpec(
        repo_id="owner/model",
        files=(remote_name,),
        file_sha256={remote_name: hashlib.sha256(expected).hexdigest()},
        file_size_bytes={remote_name: len(expected)},
    )
    recipe = SimpleNamespace(model_id="audited-model", checkpoints={"dit": default})

    overridden = NativeDiffusionAssembler._checkpoints(
        recipe,
        {"dit": str(local_file)},
    )["dit"]
    materialized = NativeCheckpointResolver().materialize(overridden)

    assert materialized.paths == (local_file,)
    assert overridden.file_sha256 == {local_file.name: hashlib.sha256(expected).hexdigest()}
