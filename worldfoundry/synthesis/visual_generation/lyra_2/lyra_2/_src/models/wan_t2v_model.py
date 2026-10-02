# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import collections
import os
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import attrs
import torch
from einops import rearrange
from torch import Tensor
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.tensor import distribute_tensor
from torch.nn.modules.module import _IncompatibleKeys

try:
    from worldfoundry.core.distributed.model_parallel.megatron_compat import parallel_state
except ModuleNotFoundError:

    class _ParallelStateFallback:
        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def get_context_parallel_group():
            return None

    parallel_state = _ParallelStateFallback()

try:
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.tuners_utils import BaseTunerLayer, _find_minimal_target_modules
except ImportError:
    print("peft is not installed, Lora is not supported")
    LoraConfig = None
    inject_adapter_in_model = None
    _find_minimal_target_modules = None
    BaseTunerLayer = None

from lyra_2._src.models.utils import (
    _convert_musubi_wan_lora_to_non_diffusers_wan,
    _convert_non_diffusers_wan_lora_to_diffusers,
)
from lyra_2._src.modules.conditioner import DataType, T2VCondition
from lyra_2._src.utils.resolution import VIDEO_RES_SIZE_INFO

from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from worldfoundry.core.configuration.lazy_config import LazyDict
from worldfoundry.core.configuration.lazy_config import instantiate as lazy_instantiate
from worldfoundry.core.distributed import broadcast_dtensor_model_states
from worldfoundry.core.distributed.model_parallel.context import (
    broadcast,
    broadcast_split_tensor,
    cat_outputs_cp,
)
from worldfoundry.core.distributed.sharding.fsdp_runtime import hsdp_device_mesh
from worldfoundry.core.distributed.runtime.logging import log
from worldfoundry.core.model_loading import InferenceModel, load_state_dict, non_strict_load_model
from worldfoundry.core.observability.time import CudaSyncTimer as sync_timer
from worldfoundry.core.utils import count_parameters as count_params
from worldfoundry.core.execution import inference_runtime as misc

IS_PREPROCESSED_KEY = "is_preprocessed"
NUM_EMBEDDING_PADDING_TOKENS = 512


@attrs.define(slots=False)
class I4LoraConfig:
    enabled: bool = False
    pretrained_lora_path: str = ""
    lora_rank: int = -1
    adapter_name: str = "default"
    lora_target_modules: list[str] = []
    init_lora_weights: str = "kaiming"


@attrs.define(slots=False)
class T2VModelConfig:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None

    fsdp_shard_size: int = 1
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "image"  # key to fetch input image from data_batch
    input_caption_key: str = "ai_caption"  # Key used to fetch input captions
    use_torch_compile: bool = False
    lora_config: I4LoraConfig = I4LoraConfig()
    use_mp_policy_fsdp: bool = False
    keep_original_net_dtype: bool = False

    state_ch: int = 16  # for latent model, ref to the latent channel number
    state_t: int = 8  # for latent model, ref to the latent number of frames
    resolution: str = "512"

    shift: int = 5
    use_dynamic_shift: bool = False


class WANDiffusionModel(InferenceModel):
    """
    Diffusion model.
    """

    def __init__(self, config: T2VModelConfig):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        log.warning(f"DiffusionModel: precision {self.precision}")

        # 1. set data keys and data information
        # self.sigma_data = config.sigma_data
        self.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        # Lazily-initialized flow-matching scheduler for DMD distillation (4-step) inference.
        self.dmd_scheduler = None

        # 3. tokenizer
        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer: Any = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == self.config.state_ch, (
                f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"
            )

        # 5. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=config.fsdp_shard_size,
            )
        else:
            self.fsdp_device_mesh = None

        # 6. diffusion neural networks part
        if "lora_config" in config:
            if config.lora_config.enabled:
                self.config.net.postpone_checkpoint = True
        self.set_up_model()

        self.net.requires_grad_(False)
        self.net.eval()

    def maybe_inject_lora_to_net(self, net, lora_config=None, skip_inject=False, skip_load=False):
        if lora_config is None:
            lora_config = self.config.lora_config
        if lora_config.enabled:
            if lora_config.pretrained_lora_path:
                self.load_lora_weights(
                    lora_path=lora_config.pretrained_lora_path,
                    adapter_name=lora_config.adapter_name,
                    training_mode=True,
                    skip_inject=skip_inject,
                    skip_load=skip_load,
                    model=net,
                )
            elif not skip_inject:
                self.add_lora_to_model(
                    adapter_name=lora_config.adapter_name,
                    lora_rank=lora_config.lora_rank,
                    lora_target_modules=lora_config.lora_target_modules,
                    init_lora_weights=lora_config.init_lora_weights,
                    training_mode=True,
                    model=net,
                )

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def build_net(self):
        config = self.config
        # NOTE: (ruiyuang) always use meta device, no need to use cpu
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with sync_timer("net instantiate"):
                with torch.device(init_device):
                    net = lazy_instantiate(config.net)

            if "lora_config" in config:
                self.maybe_inject_lora_to_net(net, skip_load=True)
            self._param_count = count_params(net, verbose=False)

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh)
                net = fully_shard(net, mesh=self.fsdp_device_mesh, reshard_after_forward=True)

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                net.init_weights()

            if self.fsdp_device_mesh:
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
            if "lora_config" in config:
                self.maybe_inject_lora_to_net(net, skip_inject=True)
        return net

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )
            self.net = self.build_net()
            self._param_count = count_params(self.net, verbose=False)
        torch.cuda.empty_cache()

    def prepare_runtime(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            if torch.__version__ < "2.3":
                log.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container, or the WorldFoundry unified CUDA environment."
                )
            # Increasing cache size. It's required because of the model size and dynamic input shapes resulting in
            # multiple different triton kernels. For 28 TransformerBlocks, the cache limit of 256 should be enough for
            # up to 9 different input shapes, as 28*9 < 256. If you have more Blocks or input shapes, and you observe
            # graph breaks at each Block (detectable with torch._dynamo.explain) or warnings about
            # exceeding cache limit, you may want to increase this size.
            # Starting with 24.05 Pytorch container, the default value is 256 anyway.
            # You can read more about it in the comments in Pytorch source code under path torch/_dynamo/cache_size.py.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape. It incurs higher compilation costs
            # at initial iterations, but can result in more specialized and efficient kernels.
            # dynamic=True currently throws errors in pytorch 2.3.
            self.net = torch.compile(self.net, dynamic=False, disable=not self.config.use_torch_compile)

    def is_lora_model(self, model):
        for _, module in model.named_modules():
            if isinstance(module, BaseTunerLayer):
                return True
        return False

    def load_lora_weights(
        self,
        lora_path,
        adapter_name=None,
        lora_alpha=None,
        training_mode=False,
        skip_inject=False,
        skip_load=False,
        model=None,
    ):
        if adapter_name is None:
            adapter_name = os.path.basename(lora_path).replace(".", "--")
        if model is None:
            model = self.net

        for _, module in model.named_modules():
            if isinstance(module, BaseTunerLayer) and not skip_inject and not skip_load:
                # on `skip_inject` or `skip_load`, we allow override the existing LoRA with same name.
                if adapter_name in module.active_adapter:
                    log.info(f"LoRA {adapter_name} already loaded, skip loading")
                    return adapter_name
                else:
                    log.warning(f"LoRA {module.active_adapter} loaded, loading {adapter_name} will override it!")
                break

        state_dict = load_state_dict(lora_path, torch_dtype=self.precision)
        if any(k.startswith("lora_unet_") for k in state_dict):
            state_dict = _convert_musubi_wan_lora_to_non_diffusers_wan(state_dict)
        # remove the prefix of the state dict
        state_dict = {k.replace("diffusion_model.", ""): v for k, v in state_dict.items()}
        # strip leading "net." prefix: load_lora_weights injects into `self.net`,
        # so the LoRA keys must be relative to `self.net` (not the full model).
        state_dict = {k[len("net.") :] if k.startswith("net.") else k: v for k, v in state_dict.items()}

        if not skip_inject:
            target_modules = list({name.split(".lora")[0] for name in state_dict.keys()})
            for key, val in state_dict.items():
                if ("lora_down" in key or "lora_A" in key) and val.ndim > 1:
                    rank = val.shape[0]
                    break
            else:
                raise ValueError("Rank is not found in the state dict")
            if lora_alpha is None:
                lora_alpha = rank
            lora_bias = any("diff_b" in k for k in state_dict)

            named_modules = model.named_modules()
            key_list = [key for key, _ in named_modules]
            names_no_target = [
                name
                for name in key_list
                if not any((name == suffix) or name.endswith("." + suffix) for suffix in target_modules)
            ]
            new_target_modules = _find_minimal_target_modules(target_modules, names_no_target)

            log.info(
                f"Injecting LoRA as from {lora_path}, rank: {rank}, lora_alpha: {lora_alpha}, lora_bias: {lora_bias}, target_modules: {new_target_modules}"
            )
            self.add_lora_to_model(
                adapter_name=adapter_name,
                lora_rank=rank,
                lora_alpha=lora_alpha,
                lora_bias=lora_bias,
                lora_target_modules=new_target_modules,
                training_mode=training_mode,
                model=model,
            )
            log.info(f"Injected LoRA weights as from {lora_path}")
        else:
            log.info(f"LoRA {adapter_name} skip injecting on this call.")
        if not skip_load:
            self.load_weights_to_lora(
                pretrained_lora_state_dict=state_dict,
                state_dict_converter=partial(_convert_non_diffusers_wan_lora_to_diffusers, adapter_name=adapter_name),
                model=model,
            )
            log.info(f"Loaded LoRA weights from {lora_path}")
        else:
            log.info(f"LoRA {adapter_name} skip loading on this call.")
        return adapter_name

    def add_lora_to_model(
        self,
        adapter_name="default",
        lora_rank=4,
        lora_alpha=None,
        lora_bias=False,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        training_mode=True,
        init_lora_weights="kaiming",
        model=None,
    ):
        if model is None:
            model = self.net

        # Add LoRA to UNet
        if init_lora_weights == "kaiming":
            init_lora_weights = True
        if lora_alpha is None:
            lora_alpha = lora_rank

        if isinstance(lora_target_modules, str):
            lora_target_modules = lora_target_modules.split(",")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules,
            lora_bias=lora_bias,
        )

        # this op is inplace
        inject_adapter_in_model(lora_config, model, adapter_name=adapter_name)

        # count trainable and total parameters
        count_trainable_params = 0
        count_total_params = 0
        if training_mode:
            for param in model.parameters():
                if param.requires_grad:
                    # # Upcast LoRA parameters into fp32
                    # param.data = param.to(torch.float32)
                    param.data = param.data.to(self.precision)
                    count_trainable_params += param.numel()
                count_total_params += param.numel()
        log.info(
            f"Trainable parameters after adding LoRA: {count_trainable_params:,} / Total parameters: {count_total_params:,}"
        )

    def load_weights_to_lora(
        self,
        pretrained_lora_path=None,
        pretrained_lora_state_dict=None,
        state_dict_converter=None,
        model=None,
    ):
        assert pretrained_lora_path is None or pretrained_lora_state_dict is None, (
            "Only one of pretrained_lora_path or pretrained_lora_state_dict should be provided"
        )

        if model is None:
            model = self.net
        # Lora pretrained lora weights
        if pretrained_lora_path is not None or pretrained_lora_state_dict is not None:
            if pretrained_lora_path is not None and pretrained_lora_state_dict is None:
                state_dict = load_state_dict(pretrained_lora_path, torch_dtype=self.precision)
                pretrained_lora_state_dict = state_dict
            if state_dict_converter is not None:
                pretrained_lora_state_dict = state_dict_converter(pretrained_lora_state_dict)
            if self.fsdp_device_mesh:
                _state_dict = get_model_state_dict(model)
                missing_keys = []
                unexpected_keys = []
                for k in _state_dict.keys():
                    if "_extra_state" in k:
                        pass
                    if k in pretrained_lora_state_dict:
                        # set local tensor to DTensor
                        _state_dict[k] = distribute_tensor(
                            pretrained_lora_state_dict.pop(k),
                            _state_dict[k].device_mesh,
                            _state_dict[k].placements,
                        )
                    else:
                        missing_keys.append(k)
                unexpected_keys = list(pretrained_lora_state_dict.keys())
                log.info(set_model_state_dict(model, _state_dict, options=StateDictOptions(strict=True)))
            else:
                missing_keys, unexpected_keys = model.load_state_dict(pretrained_lora_state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            if any("k_img" in k for k in unexpected_keys):
                total_unexpected_keys = len(unexpected_keys)
                unexpected_keys = [k for k in unexpected_keys if "k_img" not in k]
                unexpected_keys = [k for k in unexpected_keys if "v_img" not in k]
                unexpected_keys = [k for k in unexpected_keys if "img_emb.proj" not in k]
                ignore_keys = total_unexpected_keys - len(unexpected_keys)
                log.critical(f"You may loading a I2V LoRA into T2V model. Ignore {ignore_keys} unexpected keys.")
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            log.info(
                f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
            )
            if num_unexpected_keys > 0:
                log.critical(f"Unexpected keys: {unexpected_keys}")

    def set_weights_and_activate_adapters(self, adapter_names, weights=None):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        if weights is None:
            weights = [1.0] * len(adapter_names)
        assert len(adapter_names) == len(weights), "adapter_names and weights should have the same length"

        def get_module_weight(weight_for_adapter, module_name):
            if not isinstance(weight_for_adapter, dict):
                # If weight_for_adapter is a single number, always return it.
                return weight_for_adapter

            for layer_name, weight_ in weight_for_adapter.items():
                if layer_name in module_name:
                    return weight_

            raise ValueError(
                "weight_for_adapter should be a single number or a dict containing the layer "
                f"name, got {weight_for_adapter} for {module_name}"
            )

        for module_name, module in self.net.named_modules():
            if isinstance(module, BaseTunerLayer):
                # For backward compatibility with previous PEFT versions, set multiple active adapters
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_names)
                else:
                    module.active_adapter = adapter_names

                # Set the scaling weight for each adapter for this module
                for adapter_name, weight in zip(adapter_names, weights):
                    module.set_scale(adapter_name, get_module_weight(weight, module_name))
        log.info(f"Set weights: {weights}, and activate adapters: {adapter_names}")

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            if x0_B_C_T_H_W is not None:
                x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    # ------------------------ Sampling ------------------------

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        _, x0, _ = self.get_data_and_condition(data_batch)  # we need always process the data batch first.
        is_image_batch = self.is_image_batch(data_batch)

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            if guidance == 1.0:
                cond_v = self.denoise(noise_x, timestep, condition)
                noise_pred = cond_v
            elif guidance == 0.0:
                uncond_v = self.denoise(noise_x, timestep, uncondition)
                noise_pred = uncond_v
            else:
                cond_v = self.denoise(noise_x, timestep, condition)
                uncond_v = self.denoise(noise_x, timestep, uncondition)
                noise_pred = uncond_v + guidance * (cond_v - uncond_v)
            return noise_pred

        return x0_fn

    @sync_timer("WANDiffusionModel: generate_samples_from_batch")
    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)

        timesteps = self.sample_scheduler.timesteps

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        latents = noise

        if self.net.is_context_parallel_enabled:
            latents = broadcast_split_tensor(latents, seq_dim=2, process_group=self.get_context_parallel_group())

        with sync_timer(f"WANDiffusionModel: generate_samples_from_batch: {num_steps} diffusion_steps"):
            for _, t in enumerate(timesteps):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                velocity_field_pred = x0_fn(latent_model_input, timestep.unsqueeze(0))  # velocity field
                temp_x0 = self.sample_scheduler.step(
                    velocity_field_pred.unsqueeze(0), t, latents, return_dict=False, generator=seed_g
                )[0]
                latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        return latents

    @torch.no_grad()
    def forward(self, xt, t, condition: T2VCondition):
        pass

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor], return_latent_state: bool = True
    ) -> Tuple[Tensor, Tensor, T2VCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        if return_latent_state:
            latent_state = self.encode(raw_state).contiguous().float()
        else:
            latent_state = None

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        return raw_state, latent_state, condition

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            _flag = data_batch.get(IS_PREPROCESSED_KEY, False)
            if isinstance(_flag, torch.Tensor):
                try:
                    _flag = bool(_flag.bool().all().item())
                except Exception:
                    _flag = False
            else:
                _flag = bool(_flag)

            if _flag:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all((data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            _flag = data_batch.get(IS_PREPROCESSED_KEY, False)
            if isinstance(_flag, torch.Tensor):
                try:
                    _flag = bool(_flag.bool().all().item())
                except Exception:
                    _flag = False
            else:
                _flag = bool(_flag)

            if _flag:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> Dict[str, Any]:
        return self.net.state_dict(prefix="net.")

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False, pretrain_copy: bool = False
    ) -> _IncompatibleKeys:
        """Load regular inference weights from a consolidated checkpoint."""
        if pretrain_copy:
            model_state = state_dict
        else:
            model_state = collections.OrderedDict()
            for key, value in state_dict.items():
                if key.startswith("net."):
                    model_state[key.removeprefix("net.")] = value
                elif not key.startswith("net_ema."):
                    model_state[key] = value

        if strict:
            return self.net.load_state_dict(model_state, strict=True, assign=assign)

        log.warning("load model in non-strict mode")
        return non_strict_load_model(self.net, model_state)

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from the inference caller and is treated as video data.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_data_key should be present in the data_batch."
        )
        return is_image

    def denoise(self, xt_B_C_T_H_W: torch.Tensor, timestep: torch.Tensor, condition: T2VCondition) -> Tensor:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            timestep (torch.Tensor): The timestep level.
            condition (T2VCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=(xt_B_C_T_H_W).to(**self.tensor_kwargs),
            timesteps_B_T=timestep,
            **condition.to_dict(),
        ).float()

        return net_output_B_C_T_H_W

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, T_latent_seq_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        if T_latent_seq_lens is not None:
            latent_list = torch.split(latent, T_latent_seq_lens.tolist(), dim=2)
            decoded_list = [self.tokenizer.decode(latent) for latent in latent_list]
            return torch.cat(decoded_list, dim=2)
        else:
            return self.tokenizer.decode(latent)

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return height // self.tokenizer.spatial_compression_factor, width // self.tokenizer.spatial_compression_factor

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t


NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"
