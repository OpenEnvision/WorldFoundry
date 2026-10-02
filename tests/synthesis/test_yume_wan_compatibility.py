from __future__ import annotations

import pytest
import torch

# This test module imports worldfoundry code that requires the optional
# "easydict" dependency at import time; skip when it is unavailable.
pytest.importorskip("easydict")

from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan21 import (
    WAN_CONFIGS,
    i2v_14B,
)
from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume_1p5.modules.model import (
    Yume1p5WanModel,
)
from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume_1p5 import (
    worldfoundry_runtime as yume15_runtime,
)
from worldfoundry.synthesis.visual_generation.yume._facade import YumeFacadeSynthesis


def test_wan21_package_exports_upstream_config_map() -> None:
    assert WAN_CONFIGS["i2v-14B"] is i2v_14B
    assert {"t2v-1.3B", "t2v-14B", "i2v-14B"} <= set(WAN_CONFIGS)


def test_yume15_native_model_accepts_diffusers_style_config() -> None:
    model = Yume1p5WanModel.from_config(
        {
            "_class_name": "WanModel",
            "_diffusers_version": "0.33.0",
            "model_type": "ti2v",
            "patch_size": (1, 2, 2),
            "text_len": 8,
            "in_dim": 4,
            "dim": 12,
            "ffn_dim": 24,
            "freq_dim": 4,
            "text_dim": 8,
            "out_dim": 4,
            "num_heads": 1,
            "num_layers": 0,
        }
    )

    assert model.config["model_type"] == "ti2v"
    assert model.dim == 12
    assert model.device.type == "cpu"


def test_yume_facade_forwards_runtime_memory_options() -> None:
    calls = []

    class DummyRuntime:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.append(kwargs)
            return object()

    class DummySynthesis(YumeFacadeSynthesis):
        @classmethod
        def _runtime_cls(cls):
            return DummyRuntime

    DummySynthesis.from_pretrained(
        pretrained_model_path="checkpoint",
        device="cpu",
        weight_dtype=torch.bfloat16,
        fsdp=False,
        t5_cpu=True,
    )

    assert calls[0]["t5_cpu"] is True


def test_yume15_loader_destroys_owned_process_group_after_load_failure(
    monkeypatch, tmp_path
) -> None:
    distributed_state = {"initialized": False, "destroy_calls": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "is_initialized",
        lambda: distributed_state["initialized"],
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)

    def init_process_group(*, backend):
        assert backend == "gloo"
        distributed_state["initialized"] = True

    def destroy_process_group():
        distributed_state["destroy_calls"] += 1
        distributed_state["initialized"] = False

    def fail_to_load(**kwargs):
        raise RuntimeError("checkpoint load failed")

    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy_process_group)
    monkeypatch.setattr(yume15_runtime, "Yume1p5TI2V", fail_to_load)
    monkeypatch.setenv("WORLD_SIZE", "1")

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        yume15_runtime.Yume1p5Runtime.from_pretrained(
            pretrained_model_path=str(tmp_path),
            device=torch.device("cpu"),
            weight_dtype=torch.bfloat16,
            fsdp=False,
        )

    assert distributed_state == {"initialized": False, "destroy_calls": 1}
