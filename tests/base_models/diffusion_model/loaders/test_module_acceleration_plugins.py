"""Load real checkpoints through the declared DiT capability and audit scope."""

import pytest

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.core.model_loading.policy import RuntimePolicy

_CONFIG = {
    "dim": 12,
    "in_dim": 2,
    "ffn_dim": 24,
    "out_dim": 2,
    "text_dim": 8,
    "freq_dim": 4,
    "eps": 1e-6,
    "patch_size": (1, 1, 1),
    "num_heads": 1,
    "num_layers": 2,
    "has_image_input": False,
    "require_vae_embedding": False,
    "require_clip_embedding": False,
}


@pytest.mark.parametrize("supports_plugins", [True, False])
def test_shared_policy_only_installs_on_declared_target(tmp_path, supports_plugins):
    safetensors = pytest.importorskip("safetensors.torch")
    model = WanModel(**_CONFIG).eval()
    safetensors.save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=WanModel, config=_CONFIG, supports_acceleration_plugins=supports_plugins),
        CheckpointSpec(source=str(tmp_path), files=("model.safetensors",)),
        RuntimePolicy(options={"accelerations": {"easycache": {"threshold": 0.1}}}),
    )
    assert tuple(loaded.state_dict()) == tuple(model.state_dict())
    assert hasattr(loaded, "_worldfoundry_accelerations") is supports_plugins
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    if supports_plugins:
        assert snapshot.quality_tier == "approximate"
        assert snapshot.effective["accelerations"]["execution_verified"] is False
        loaded._worldfoundry_accelerations.uninstall()
        assert not hasattr(loaded, "_worldfoundry_easycache_config")
    else:
        assert "accelerations" not in snapshot.effective
