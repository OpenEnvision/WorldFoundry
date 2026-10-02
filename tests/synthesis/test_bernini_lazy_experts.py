from types import SimpleNamespace

import torch

from worldfoundry.synthesis.visual_generation.bernini.inference.models import (
    wan_diffusion,
)


class _Expert(torch.nn.Module):
    def __init__(self, subfolder):
        super().__init__()
        self.subfolder = subfolder


def test_lazy_experts_load_directly_on_device_and_replace_each_other(monkeypatch):
    calls = []
    monkeypatch.setattr(
        wan_diffusion.WanTransformer3DModel,
        "load_config",
        lambda *args, **kwargs: {"in_channels": 16, "text_dim": 4096},
    )

    def fake_from_pretrained(*args, subfolder, **kwargs):
        calls.append((subfolder, kwargs))
        return _Expert(subfolder)

    monkeypatch.setattr(
        wan_diffusion.WanTransformer3DModel,
        "from_pretrained",
        fake_from_pretrained,
    )
    config = SimpleNamespace(
        switch_dit_boundary=0.875,
        wan22_base="/checkpoint",
        transformer_config_path=None,
        transformer_2_config_path=None,
        use_src_id_rotary_emb=True,
        lazy_expert_loading=True,
        expert_device="cuda",
        skip_transformer_1=False,
        skip_transformer_2=False,
        use_unipc=False,
        shift=3.0,
    )

    model = wan_diffusion.GEN_Wanx22(config)
    assert model.transformer is None
    assert model.transformer_2 is None

    model._activate_initial_expert("cuda")
    assert model.transformer.subfolder == "transformer"
    assert calls[0][1]["device_map"] == {"": "cuda"}

    model._switch_to_low_expert("cuda")
    assert model.transformer is None
    assert model.transformer_2.subfolder == "transformer_2"
    assert calls[1][1]["device_map"] == {"": "cuda"}

    model.release_lazy_experts()
    assert model.transformer is None
    assert model.transformer_2 is None
