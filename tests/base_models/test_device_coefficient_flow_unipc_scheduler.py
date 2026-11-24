# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Device-coefficient UniPC shares native math and preserves Diffusers serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers.scheduling_utils import SchedulerOutput

from worldfoundry.base_models.diffusion_model.schedulers._solver_support import SchedulerConfig
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler as NativeUniPC
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc_compat import FlowUniPCMultistepScheduler
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc_device import (
    DeviceCoefficientFlowUniPCMultistepScheduler,
)

pytestmark = pytest.mark.unit


def test_device_unipc_inherits_native_predictor_and_corrector() -> None:
    for scheduler in (DeviceCoefficientFlowUniPCMultistepScheduler, FlowUniPCMultistepScheduler):
        assert scheduler.multistep_uni_p_bh_update is NativeUniPC.multistep_uni_p_bh_update
        assert scheduler.multistep_uni_c_bh_update is NativeUniPC.multistep_uni_c_bh_update
        assert scheduler.set_timesteps is NativeUniPC.set_timesteps
    assert isinstance(NativeUniPC().config, SchedulerConfig)
    assert isinstance(FlowUniPCMultistepScheduler().config, FrozenDict)


def test_unipc_coefficient_device_policy_is_separate_from_update_math() -> None:
    native = NativeUniPC()
    variant = DeviceCoefficientFlowUniPCMultistepScheduler()
    h = torch.tensor(0.5)
    requested = torch.device("meta")
    assert native._coefficient_ratios([h], h, requested).device.type == "cpu"
    assert native._coefficient_rhs([h], requested).device.type == "cpu"
    assert variant._coefficient_ratios([h], h, requested).device == requested
    assert variant._coefficient_rhs([h], requested).device == requested


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_device_unipc_cpu_trajectory_matches_native(order: int, dtype: torch.dtype) -> None:
    torch.manual_seed(17)
    schedulers = [cls(solver_order=order, shift=1) for cls in (NativeUniPC, FlowUniPCMultistepScheduler)]
    samples = [torch.randn(2, 3, 2, 2).to(dtype)]
    samples.append(samples[0].clone())
    for scheduler in schedulers:
        scheduler.set_timesteps(7, shift=3)
    for index in range(7):
        for position, scheduler in enumerate(schedulers):
            sample = samples[position]
            timestep = scheduler.timesteps[index]
            prediction = torch.tanh(sample * 0.1 + timestep.float() / 1000).to(dtype)
            samples[position] = scheduler.step(prediction, timestep, sample)[0]
        torch.testing.assert_close(samples[1], samples[0], rtol=0, atol=0)


def test_device_unipc_diffusers_save_load_and_output_contract(tmp_path: Path) -> None:
    config = {"_class_name": "FlowUniPCMultistepScheduler", "num_train_timesteps": 1000, "shift": 1}
    (tmp_path / "scheduler_config.json").write_text(json.dumps(config))
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(tmp_path)
    scheduler.save_pretrained(tmp_path / "saved")
    saved_config = json.loads((tmp_path / "saved/scheduler_config.json").read_text())
    assert saved_config["_class_name"] == "FlowUniPCMultistepScheduler"
    restored = FlowUniPCMultistepScheduler.from_pretrained(tmp_path / "saved")
    assert restored.config["shift"] == scheduler.config["shift"]
    assert restored.config["solver_order"] == scheduler.config["solver_order"]
    restored.set_timesteps(3)
    value = torch.randn(1, 2, 2, 2)
    output = restored.step(torch.zeros_like(value), restored.timesteps[0], value)
    assert isinstance(output, SchedulerOutput)
    assert output[0] is output["prev_sample"]
    tuple_output = restored.step(torch.zeros_like(value), restored.timesteps[1], output.prev_sample, return_dict=False)
    assert isinstance(tuple_output, tuple)
    assert len(tuple_output) == 1
