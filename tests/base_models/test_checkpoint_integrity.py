from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ftfy")

from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.materialize import NativeCheckpointResolver
from worldfoundry.base_models.diffusion_model.recipes.sana import sana_recipe
from worldfoundry.base_models.diffusion_model.recipes.wan import (
    WAN21_T2V_1P3B_REVISION,
    wan21_t2v_1p3b_recipe,
)
from worldfoundry.core.io.paths import resolve_local_hf_model_path


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
    snapshot = cache_root / "models--owner--model" / "snapshots" / revision
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
        snapshot = hub_root / "models--owner--model" / "snapshots" / revision
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
    without_legacy = {name: value for name, value in without_explicit.items() if name != "HUGGINGFACE_HUB_CACHE"}
    assert resolve(without_legacy) == home_snapshot.resolve()
    without_home = {name: value for name, value in without_legacy.items() if name != "HF_HOME"}
    assert resolve(without_home) == xdg_snapshot.resolve()

    home_default_snapshot = snapshot_at(
        tmp_path / "injected-home" / ".cache" / "huggingface" / "hub",
        "home-default",
    )
    without_xdg = {name: value for name, value in without_home.items() if name != "XDG_CACHE_HOME"}
    without_xdg["HOME"] = str(tmp_path / "injected-home")
    assert resolve(without_xdg) == home_default_snapshot.resolve()


def test_local_checkpoint_resolution_does_not_read_weights(tmp_path, monkeypatch) -> None:
    checkpoint_file = tmp_path / "model.safetensors"
    checkpoint_file.write_bytes(b"local model weights")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("checkpoint resolution read the weight payload")

    monkeypatch.setattr(Path, "open", unexpected_read)
    materialized = NativeCheckpointResolver().materialize(
        CheckpointSpec(source=tmp_path, files=(checkpoint_file.name,))
    )

    assert materialized.paths == (checkpoint_file,)


def test_local_checkpoint_resolution_requires_declared_files(tmp_path) -> None:
    spec = CheckpointSpec(source=tmp_path, files=("missing.safetensors",))

    with pytest.raises(FileNotFoundError, match="checkpoint files do not exist"):
        NativeCheckpointResolver().materialize(spec)


def test_required_configuration_is_resolved_without_loading_as_weights(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    weights.write_bytes(b"weights")
    spec = CheckpointSpec(source=tmp_path, files=(weights.name,), resources=(config.name,))

    with pytest.raises(FileNotFoundError, match="checkpoint resources do not exist"):
        NativeCheckpointResolver().materialize(spec)

    config.write_text('{"width": 8}', encoding="utf-8")
    materialized = NativeCheckpointResolver().materialize(spec)

    assert materialized.paths == (weights,)


def test_hub_resolution_requires_model_configuration(tmp_path, monkeypatch) -> None:
    from worldfoundry.core.io import paths

    def cached_model(repo_id, *, required_files, revision):
        assert required_files == ("model.safetensors", "config.json")
        return tmp_path

    monkeypatch.setattr(paths, "resolve_local_hf_model_path", cached_model)
    spec = CheckpointSpec(repo_id="owner/model", files=("model.safetensors",), resources=("config.json",))

    assert NativeCheckpointResolver().materialize(spec).paths == (tmp_path / "model.safetensors",)


def test_local_checkpoint_override_accepts_replacement_weights(tmp_path) -> None:
    local_file = tmp_path / "lightvaew2_1.pth"
    local_file.write_bytes(b"student checkpoint")
    recipe = wan21_t2v_1p3b_recipe()

    overridden = NativeDiffusionAssembler.resolve_checkpoints(recipe, {"vae": str(local_file)})["vae"]
    materialized = NativeCheckpointResolver().materialize(overridden)

    assert materialized.paths == (local_file,)


def test_directory_override_preserves_checkpoint_layout(tmp_path) -> None:
    local_file = tmp_path / "checkpoints" / "model.pth"
    local_file.parent.mkdir()
    local_file.write_bytes(b"replacement model")
    default = CheckpointSpec(repo_id="owner/model", files=("checkpoints/model.pth",))
    recipe = SimpleNamespace(model_id="local-model", checkpoints={"dit": default})

    overridden = NativeDiffusionAssembler.resolve_checkpoints(recipe, {"dit": str(tmp_path)})["dit"]

    assert NativeCheckpointResolver().materialize(overridden).paths == (local_file,)


def test_sana_sprint_recipes_pin_independent_student_and_teacher_sources() -> None:
    expected = {
        "sana-sprint-600m-1024px": (
            "268b7e32816200db48e0b0a8939f32397ae6889f",
            "checkpoints/Sana_Sprint_0.6B_1024px.pth",
            "141411fcb8360dc1e6ae91e7c7ed69371adb058c",
            "checkpoints/Sana_Sprint_0.6B_1024px_teacher.pth",
        ),
        "sana-sprint-1600m-1024px": (
            "9f6866d6962d5d91d213f113e162cb18bffe741c",
            "checkpoints/Sana_Sprint_1.6B_1024px.pth",
            "f111cf761e05a1ad460cd5fb528c1a107327b459",
            "checkpoints/Sana_Sprint_1.6B_1024px_teacher.pth",
        ),
    }
    for model_id, (student_revision, student_file, teacher_revision, teacher_file) in expected.items():
        recipe = sana_recipe(model_id)
        assert recipe.checkpoints["dit"].revision == student_revision
        assert recipe.checkpoints["dit"].files == (student_file,)
        assert recipe.checkpoints["teacher"].revision == teacher_revision
        assert recipe.checkpoints["teacher"].files == (teacher_file,)


def test_wan_recipe_pins_component_sources() -> None:
    recipe = wan21_t2v_1p3b_recipe()

    assert all(checkpoint.revision == WAN21_T2V_1P3B_REVISION for checkpoint in recipe.checkpoints.values())
    assert recipe.checkpoints["vae"].files == ("Wan2.1_VAE.pth",)
