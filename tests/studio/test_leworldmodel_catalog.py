from __future__ import annotations

from argparse import Namespace

from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.world_model.le_wm import worldfoundry_runtime as runtime


def test_leworldmodel_catalog_prefers_local_smoke_dataset(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "ckpts" / "leworldmodel" / "pusht_worldfoundry_smoke.h5"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"hdf5 fixture")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    entry = find_entry("leworldmodel")

    assert entry.display_name == "LeWorldModel"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == ""
    assert entry.default_load_kwargs == {
        "model_id": "leworldmodel",
        "config_name": "pusht",
        "policy": "random",
        "cache_dir": str(dataset.parent),
        "dataset_name": str(dataset),
        "num_eval": 1,
        "eval_budget": 25,
        "goal_offset_steps": 5,
        "seed": 42,
    }
    assert entry.default_call_kwargs["plan_only"] is False


def test_leworldmodel_inference_contract_is_portable_and_executes_by_default() -> None:
    spec = get_model_inference_spec("leworldmodel")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert variant.checkpoints == ()
    assert variant.load_kwargs["cache_dir"] == "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel"
    assert variant.load_kwargs["dataset_name"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert variant.load_kwargs["config_dir"].startswith("${WORLDFOUNDRY_DATA_ROOT}/")
    assert task.default_call_kwargs["plan_only"] is False


def test_leworldmodel_runtime_resolves_portable_dataset_path(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "checkpoints" / "leworldmodel" / "pusht_worldfoundry_smoke.h5"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"hdf5 fixture")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    options = {
        "config_name": "pusht",
        "policy": "random",
        "cache_dir": "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel",
        "dataset_name": "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel/pusht_worldfoundry_smoke.h5",
    }

    assert runtime.BLOCKED_REASON == ""
    assert runtime.missing_requirements(
        options=options,
        runtime_root=runtime.RUNTIME_DIR,
        entrypoint=runtime.OFFICIAL_ENTRYPOINT,
        profile=None,
    ) == []

    command = runtime.build_command(
        {
            "python": "python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_path": str(tmp_path / "leworldmodel_result.json"),
            "output_dir": str(tmp_path),
            "device": "cpu",
            "options": options,
        }
    )
    assert str(dataset) in command
    assert "${WORLDFOUNDRY_CKPT_DIR}" not in " ".join(command)


def test_leworldmodel_eval_config_composes_with_dynamic_cache_dir(tmp_path, monkeypatch) -> None:
    from worldfoundry.synthesis.visual_generation.world_model.le_wm.infer import _load_config

    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "checkpoints"))
    args = Namespace(
        config_dir=str(runtime.CONFIG_DIR),
        config_name="pusht",
        policy="random",
        cache_dir="${WORLDFOUNDRY_CKPT_DIR}/leworldmodel",
        seed=42,
        dataset_name="${WORLDFOUNDRY_CKPT_DIR}/leworldmodel/pusht_worldfoundry_smoke.h5",
        num_eval=1,
        eval_budget=25,
        goal_offset_steps=5,
        img_size=None,
    )

    cfg = _load_config(args)

    assert cfg.policy == "random"
    assert cfg.cache_dir == str(tmp_path / "checkpoints" / "leworldmodel")
    assert cfg.eval.dataset_name == str(
        tmp_path / "checkpoints" / "leworldmodel" / "pusht_worldfoundry_smoke.h5"
    )
    assert cfg.eval.num_eval == 1
    assert cfg.eval.eval_budget == 25
    assert cfg.eval.goal_offset_steps == 5
