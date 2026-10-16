from __future__ import annotations

from unittest.mock import Mock

import pytest

from worldfoundry.synthesis.action_generation.molmobot import pi0_policy, runtime


@pytest.fixture
def pi0_assets(tmp_path, monkeypatch):
    checkpoint = tmp_path / "local-pi0-checkpoint"
    config = checkpoint / "assets" / "train_config.pkl"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"local training config")
    openpi_data = tmp_path / "openpi"
    tokenizer = openpi_data / "big_vision" / "paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"local tokenizer")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(openpi_data))
    monkeypatch.setattr(runtime, "_RUNTIME_CACHE", {})
    return checkpoint, config, tokenizer


def test_loads_local_pi0_assets_without_fixed_digests(pi0_assets, monkeypatch):
    checkpoint, _, _ = pi0_assets
    policy = Mock()
    factory = Mock(return_value=policy)
    monkeypatch.setattr(pi0_policy, "Pi0JointPositionPolicy", factory)

    assert runtime._pi0_runtime_for(str(checkpoint), "cuda:0", {}) is policy

    factory.assert_called_once_with(
        checkpoint_dir=str(checkpoint),
        device="cuda:0",
        camera_keys=("exo_camera_1", "wrist_camera"),
    )
    policy.prepare_model.assert_called_once_with()


@pytest.mark.parametrize("missing_asset", ["config", "tokenizer"])
def test_missing_pi0_assets_report_the_required_file(pi0_assets, missing_asset):
    checkpoint, config, tokenizer = pi0_assets
    missing = config if missing_asset == "config" else tokenizer
    missing.unlink()

    with pytest.raises(FileNotFoundError, match=missing.name.replace(".", r"\.")):
        runtime._pi0_runtime_for(str(checkpoint), "cuda", {})


def test_pi0_still_requires_cuda(pi0_assets):
    checkpoint, _, _ = pi0_assets

    with pytest.raises(ValueError, match="requires a CUDA device"):
        runtime._pi0_runtime_for(str(checkpoint), "cpu", {})


def test_pi0_model_loading_error_is_preserved(pi0_assets, monkeypatch):
    checkpoint, _, _ = pi0_assets
    policy = Mock()
    policy.prepare_model.side_effect = ValueError("invalid checkpoint weights")
    monkeypatch.setattr(pi0_policy, "Pi0JointPositionPolicy", Mock(return_value=policy))

    with pytest.raises(ValueError, match="invalid checkpoint weights"):
        runtime._pi0_runtime_for(str(checkpoint), "cuda", {})

    assert not runtime._RUNTIME_CACHE
