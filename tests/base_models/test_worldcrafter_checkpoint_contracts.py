from __future__ import annotations

import json

import pytest

from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.fast.contract import (
    load_dmd_inference_contract,
    resolve_dmd_inference_trace,
)
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.fast.timestep_grid import (
    DmdSchedulerConfig,
)


def _dmd_document() -> dict:
    stages = [
        [
            {
                "model_timestep": first,
                "current_sigma": 1.0,
                "next_sigma": 0.5,
            },
            {
                "model_timestep": second,
                "current_sigma": 0.5,
                "next_sigma": 0.0,
            },
        ]
        for first, second in (
            (998.5342, 833.9636),
            (742.8216, 547.1926),
            (385.4137, 253.9905),
        )
    ]
    return {
        "schema_version": 3,
        "latent_shape": [16, 9, 48, 80],
        "student": {
            "condition_dtype": "float32",
            "rollout_steps_per_stage": [2, 2, 2],
            "amplify_empty_history": False,
            "normal_stages": stages,
            "empty_history_stages": stages,
            "scheduler_provenance": DmdSchedulerConfig(
                num_train_timesteps=1000,
                stages=3,
                stage_range=(0.0, 1 / 3, 2 / 3, 1.0),
                gamma=1 / 3,
                shift=1.0,
                version="v1",
            ).to_dict(),
        },
    }


def test_dmd_contract_compares_schedule_values_and_ignores_metadata(tmp_path) -> None:
    sidecar = tmp_path / "dmd_timestep_contract.json"
    document = _dmd_document()
    sidecar.write_text(json.dumps(document))
    original = load_dmd_inference_contract(tmp_path)

    document.update(fingerprint="stale", description="local metadata")
    sidecar.write_text(json.dumps(document))
    updated = load_dmd_inference_contract(tmp_path)

    assert updated == original
    trace = resolve_dmd_inference_trace(
        updated,
        latent_shape=(16, 9, 48, 80),
        num_stages=3,
        history_tensors=[],
    )
    assert trace.empty_history is True
    assert trace.num_steps == 6


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_timestep", 900.0, "Locked .* schedule changed"),
        ("next_sigma", 0.7, "Broken x0 re-noise transition"),
    ],
)
def test_dmd_contract_keeps_numerical_validation(tmp_path, field, value, match) -> None:
    document = _dmd_document()
    document["student"]["normal_stages"][0][0][field] = value
    sidecar = tmp_path / "dmd_timestep_contract.json"
    sidecar.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=match):
        load_dmd_inference_contract(tmp_path)


def test_repencoder_metadata_load_does_not_scan_model_bytes(tmp_path, monkeypatch) -> None:
    pytest.importorskip("safetensors")
    from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.repencoder.checkpoint import (
        read_checkpoint_metadata,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.repencoder.config import (
        RepEncoderConfig,
    )

    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"weights")
    (tmp_path / "config.json").write_text(json.dumps(RepEncoderConfig().to_dict()))
    manifest = {"format": "worldcrafter_repencoder_v1", "model_sha256": "stale"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        type(model_path),
        "read_bytes",
        lambda _path: pytest.fail("metadata loading scanned model bytes"),
    )

    path, config, metadata = read_checkpoint_metadata(tmp_path)

    assert path == model_path
    assert config == RepEncoderConfig()
    assert metadata == manifest
