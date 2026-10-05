from importlib import import_module
from types import SimpleNamespace

import pytest


SAMPLER_MODULES = (
    "worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim",
    "worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim_multiplecond",
    "worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim_vid2world",
    "worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim_multiplecond_vid2world",
)


@pytest.mark.parametrize("module_name", SAMPLER_MODULES)
def test_ddim_buffers_follow_model_device(monkeypatch, module_name):
    module = import_module(module_name)

    class FakeTensor:
        def __init__(self):
            self.device = "cpu"
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            self.device = device
            return self

    fake_torch = SimpleNamespace(Tensor=FakeTensor, device=lambda value: str(value))
    monkeypatch.setattr(module, "torch", fake_torch)
    sampler = module.DDIMSampler.__new__(module.DDIMSampler)
    sampler.model = SimpleNamespace(device="cuda:3")
    tensor = FakeTensor()

    sampler.register_buffer("schedule", tensor)

    assert tensor.moves == ["cuda:3"]
    assert sampler.schedule is tensor
