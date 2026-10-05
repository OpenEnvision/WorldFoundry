"""Shared staged inference runner for research pipelines with custom inputs.

This is a parallel research path, not the native recipe → assembler → strategy
→ runner chain. StagedDiffusionPipeline owns preprocessing, VRAM/LoRA helpers,
and CFG-guided model_fn; InferenceStageGraph / InferenceStageRunner sequence
declarative stages. Native recipes should still be preferred for standard models.

Must not: mutate a process-wide DiffSynth registry; discover checkpoints without
an explicit ``model_pool_factory``.
"""

from PIL import Image
import torch
import numpy as np
from einops import rearrange, repeat, reduce
from typing import Union
from worldfoundry.core.execution.device import get_device_name, get_device_type, is_npu_available, parse_device_type
from worldfoundry.core.model_loading import GeneralLoRALoader, ModelConfig, load_state_dict
from worldfoundry.core.vram import AutoTorchModule, AutoWrappedLinear


class InferenceStage:
    """One declarative preprocessing or postprocessing stage.

    Fields:
        seperate_cfg: When True, InferenceStageRunner runs process() once for
            positive and (if cfg_scale != 1) once for negative.
        take_over: When True, process() receives the full shared/posi/nega maps
            and must return the updated triple.
        input_params / output_params: Shared-state keys this stage reads / writes.
        input_params_posi / input_params_nega: Branch-local remaps for CFG.
        onload_model_names: Models this stage needs on device; used by the graph split.
    """

    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None,
    ):
        """Init.

        Args:
            seperate_cfg: The seperate cfg.
            take_over: The take over.
            input_params: The input params.
            output_params: The output params.
            input_params_posi: The input params posi.
            input_params_nega: The input params nega.
            onload_model_names: The onload model names.
        """
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        """Union of shared / positive / negative input keys, used to build the stage graph."""
        params = []
        if self.input_params is not None:
            for param in self.input_params:
                params.append(param)
        if self.input_params_posi is not None:
            for _, param in self.input_params_posi.items():
                params.append(param)
        if self.input_params_nega is not None:
            for _, param in self.input_params_nega.items():
                params.append(param)
        params = sorted(list(set(params)))
        return params

    def fetch_output_params(self):
        """Keys this stage writes; later stages that read them become graph dependents."""
        params = []
        if self.output_params is not None:
            for param in self.output_params:
                params.append(param)
        return params

    def process(self, pipe, **kwargs) -> dict:
        """Run the stage. Override in subclasses. Default is a no-op empty dict."""
        return {}

    def post_process(self, pipe, **kwargs) -> dict:
        """Post process.

        Args:
            pipe: The pipe.

        Returns:
            The return value.
        """
        return {}


class StagedDiffusionPipeline(torch.nn.Module):
    """Composable inference shell for work-specific staged diffusion flows.

    It owns no model registry.  A work that still needs hash-based checkpoint
    discovery must provide ``model_pool_factory`` explicitly; native recipes
    should be preferred for standard models.
    """

    # Specialized pipelines can expose a narrow checkpoint registry without
    # mutating the process-wide DiffSynth registry.
    model_pool_factory = None
    model_pool_configs = None

    def __init__(
        self,
        device=get_device_type(),
        torch_dtype=torch.float16,
        height_division_factor=64,
        width_division_factor=64,
        time_division_factor=None,
        time_division_remainder=None,
    ):
        """Init.

        Args:
            device: The device.
            torch_dtype: The torch dtype.
            height_division_factor: The height division factor.
            width_division_factor: The width division factor.
            time_division_factor: The time division factor.
            time_division_remainder: The time division remainder.
        """
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        self.device_type = parse_device_type(device)
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # VRAM management
        self.vram_management_enabled = False
        self.cpu_offload = False
        self.model_names = []
        # Pipeline Unit Runner
        self.unit_runner = InferenceStageRunner()
        # LoRA Loader
        self.lora_loader = GeneralLoRALoader

    def to(self, *args, **kwargs):
        """To."""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        """Check resize height width.

        Args:
            height: The height.
            width: The width.
            num_frames: The num frames.
        """
        # Shape check
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            )
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    num_frames + self.time_division_factor - 1
                ) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(
                    f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}."
                )
            return height, width, num_frames

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        """Preprocess image.

        Args:
            image: The image.
            torch_dtype: The torch dtype.
            device: The device.
            pattern: The pattern.
            min_value: The min value.
            max_value: The max value.
        """
        target_dtype = torch_dtype or self.torch_dtype
        target_device = device or self.device
        if isinstance(image, torch.Tensor):
            # Tensor inputs use the same [0, 1] convention as preprocess_video.
            # Accept the two layouts emitted by native runtimes without routing a
            # CUDA tensor through NumPy.
            image = image.to(dtype=target_dtype, device=target_device)
            image = image * (max_value - min_value) + min_value
            if image.ndim == 4:
                source_pattern = "B C H W"
            elif image.ndim == 3 and image.shape[0] in (1, 3, 4):
                source_pattern = "C H W"
            elif image.ndim == 3:
                source_pattern = "H W C"
            else:
                raise ValueError(
                    f"Tensor image must have CHW, HWC, or BCHW layout, got shape {tuple(image.shape)}"
                )
            if source_pattern != pattern:
                if "B" in pattern and "B" not in source_pattern:
                    image = rearrange(image, f"{source_pattern} -> {pattern}", B=1)
                else:
                    image = rearrange(image, f"{source_pattern} -> {pattern}")
            return image

        # Transform a PIL.Image or ndarray in [0, 255] to torch.Tensor.
        image = torch.as_tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=target_dtype, device=target_device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
        normalize=1.0,
    ):
        """Preprocess video.

        Args:
            video: The video.
            torch_dtype: The torch dtype.
            device: The device.
            pattern: The pattern.
            min_value: The min value.
            max_value: The max value.
            normalize: The normalize.
        """
        if isinstance(video, torch.Tensor) or (
            isinstance(video, (list, tuple)) and video and isinstance(video[0], torch.Tensor)
        ):
            if not isinstance(video, torch.Tensor):
                video = torch.stack(list(video), dim=0)
            if video.ndim == 4:
                # A frame sequence is T C H W; add the batch dimension.
                video = video.unsqueeze(0)
            elif video.ndim != 5:
                raise ValueError(
                    f"Tensor video must have TCHW or BTCHW layout, got shape {tuple(video.shape)}"
                )
            video = video / normalize
            video = video.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
            video = video * (max_value - min_value) + min_value
            if pattern != "B T C H W":
                video = rearrange(video, f"B T C H W -> {pattern}")
        else:
            # Transform a list of PIL.Image to torch.Tensor
            video = [
                self.preprocess_image(
                    image,
                    torch_dtype=torch_dtype,
                    device=device,
                    min_value=min_value,
                    max_value=max_value,
                )
                for image in video
            ]
            video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def preprocess_images(self, images):
        """Preprocess images.

        Args:
            images: The images.
        """
        return [self.preprocess_image(image) for image in images]

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        """Vae output to image.

        Args:
            vae_output: The vae output.
            pattern: The pattern.
            min_value: The min value.
            max_value: The max value.
        """
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        """Vae output to video.

        Args:
            vae_output: The vae output.
            pattern: The pattern.
            min_value: The min value.
            max_value: The max value.
        """
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [
            self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value)
            for image in vae_output
        ]
        return video

    def load_models_to_device(self, model_names=()):
        """Load models to device.

        Args:
            model_names: The model names.
        """
        if not (self.vram_management_enabled or self.cpu_offload):
            return
        model_names = set(model_names or ())
        child_map = dict(self.named_children())
        managed_names = self.model_names or tuple(child_map)
        for name in managed_names:
            model = child_map.get(name, getattr(self, name, None))
            if model is None or name in model_names:
                continue
            if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                if hasattr(model, "offload"):
                    model.offload()
                else:
                    for module in model.modules():
                        if hasattr(module, "offload"):
                            module.offload()
            elif self.cpu_offload:
                model.cpu()
        device_module = getattr(torch, self.device_type, None)
        if device_module is not None and hasattr(device_module, "empty_cache"):
            device_module.empty_cache()
        for name in managed_names:
            if name not in model_names:
                continue
            model = child_map.get(name, getattr(self, name, None))
            if model is None:
                continue
            if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                if hasattr(model, "onload"):
                    model.onload()
                else:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
            elif self.cpu_offload:
                model.to(self.device)

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device=None,
        rand_torch_dtype=torch.float32,
        device=None,
        torch_dtype=None,
        dtype=None,
    ):
        """Generate noise.

        Args:
            shape: The shape.
            seed: The seed.
            rand_device: The rand device.
            rand_torch_dtype: The rand torch dtype.
            device: The device.
            torch_dtype: The torch dtype.
            dtype: The dtype.
        """
        # Initialize Gaussian noise
        if dtype is not None:
            torch_dtype = dtype
            rand_torch_dtype = dtype
        rand_device = rand_device or device or "cpu"
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def get_vram(self):
        """Get vram."""
        device = self.device if not is_npu_available() else get_device_name()
        return getattr(torch, self.device_type).mem_get_info(device)[1] / (1024**3)

    def enable_cpu_offload(self):
        """Enable cpu offload."""
        self.cpu_offload = True
        self.vram_management_enabled = True

    def get_module(self, model, name):
        """Get module.

        Args:
            model: The model.
            name: The name.
        """
        if "." in name:
            name, suffix = name[: name.index(".")], name[name.index(".") + 1 :]
            if name.isdigit():
                return self.get_module(model[int(name)], suffix)
            else:
                return self.get_module(getattr(model, name), suffix)
        else:
            return getattr(model, name)

    def freeze_except(self, model_names):
        """Freeze except.

        Args:
            model_names: The model names.
        """
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(
                    f"No {name} models in the pipeline. We cannot enable training on the model. If this occurs during the data processing stage, it is normal."
                )
                continue
            module.train()
            module.requires_grad_(True)

    def blend_with_mask(self, base, addition, mask):
        """Blend with mask.

        Args:
            base: The base.
            addition: The addition.
            mask: The mask.
        """
        return base * (1 - mask) + addition * mask

    def step(self, scheduler, latents, progress_id, noise_pred, input_latents=None, inpaint_mask=None, **kwargs):
        """Step.

        Args:
            scheduler: The scheduler.
            latents: The latents.
            progress_id: The progress id.
            noise_pred: The noise pred.
            input_latents: The input latents.
            inpaint_mask: The inpaint mask.
        """
        timestep = scheduler.timesteps[progress_id]
        if inpaint_mask is not None:
            noise_pred_expected = scheduler.return_to_timestep(scheduler.timesteps[progress_id], latents, input_latents)
            noise_pred = self.blend_with_mask(noise_pred_expected, noise_pred, inpaint_mask)
        latents_next = scheduler.step(noise_pred, timestep, latents)
        return latents_next

    def merge_latents(self, value, latents, masks, scales, blur_kernel_size=33, blur_sigma=10.0):
        """Merge latents.

        Args:
            value: The value.
            latents: The latents.
            masks: The masks.
            scales: The scales.
            blur_kernel_size: The blur kernel size.
            blur_sigma: The blur sigma.
        """
        if len(latents) == 0:
            return value
        from torchvision.transforms import GaussianBlur

        blur = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        height, width = value.shape[-2:]
        weight = torch.ones_like(value)
        for latent, mask, scale in zip(latents, masks, scales):
            mask = self.preprocess_image(mask.resize((width, height))).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, latent.shape[1], 1, 1).to(dtype=latent.dtype, device=latent.device)
            mask = blur(mask)
            value += latent * mask * scale
            weight += mask * scale
        value /= weight
        return value

    def control_noise_via_local_prompts(
        self,
        prompt_emb_global,
        prompt_emb_locals,
        masks,
        mask_scales,
        inference_callback,
        special_kwargs=None,
        special_local_kwargs_list=None,
    ):
        """Control noise via local prompts.

        Args:
            prompt_emb_global: The prompt emb global.
            prompt_emb_locals: The prompt emb locals.
            masks: The masks.
            mask_scales: The mask scales.
            inference_callback: The inference callback.
            special_kwargs: The special kwargs.
            special_local_kwargs_list: The special local kwargs list.
        """
        if special_kwargs is None:
            noise_pred_global = inference_callback(prompt_emb_global)
        else:
            noise_pred_global = inference_callback(prompt_emb_global, special_kwargs)
        if special_local_kwargs_list is None:
            noise_pred_locals = [inference_callback(prompt_emb_local) for prompt_emb_local in prompt_emb_locals]
        else:
            noise_pred_locals = [
                inference_callback(prompt_emb_local, special_kwargs)
                for prompt_emb_local, special_kwargs in zip(prompt_emb_locals, special_local_kwargs_list)
            ]
        return self.merge_latents(noise_pred_global, noise_pred_locals, masks, mask_scales)

    def extend_prompt(self, prompt, local_prompts, masks, mask_scales):
        """Extend prompt.

        Args:
            prompt: The prompt.
            local_prompts: The local prompts.
            masks: The masks.
            mask_scales: The mask scales.
        """
        local_prompts = local_prompts or []
        masks = masks or []
        mask_scales = mask_scales or []
        extended_prompt_dict = self.prompter.extend_prompt(prompt)
        prompt = extended_prompt_dict.get("prompt", prompt)
        local_prompts += extended_prompt_dict.get("prompts", [])
        masks += extended_prompt_dict.get("masks", [])
        mask_scales += [100.0] * len(extended_prompt_dict.get("masks", []))
        return prompt, local_prompts, masks, mask_scales

    def split_pipeline_units(self, model_names: list[str]):
        """Split pipeline units.

        Args:
            model_names: The model names.
        """
        return InferenceStageGraph().split_pipeline_units(self.units, model_names)

    def flush_vram_management_device(self, device):
        """Flush vram management device.

        Args:
            device: The device.
        """
        for module in self.modules():
            if isinstance(module, AutoTorchModule):
                module.offload_device = device
                module.onload_device = device
                module.preparing_device = device
                module.computation_device = device

    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=None,
        state_dict=None,
        verbose=1,
    ):
        """Load lora.

        Args:
            module: The module.
            lora_config: The lora config.
            alpha: The alpha.
            hotload: The hotload.
            state_dict: The state dict.
            verbose: The verbose.
        """
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        lora_loader = self.lora_loader(torch_dtype=self.torch_dtype, device=self.device)
        lora = lora_loader.convert_state_dict(lora)
        if hotload is None:
            hotload = hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")
        if hotload:
            if not (hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")):
                raise ValueError("VRAM Management is not enabled. LoRA hotloading is not supported.")
            updated_num = 0
            for _, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    name = module.name
                    lora_a_name = f"{name}.lora_A.weight"
                    lora_b_name = f"{name}.lora_B.weight"
                    if lora_a_name in lora and lora_b_name in lora:
                        updated_num += 1
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
            if verbose >= 1:
                print(
                    f"{updated_num} tensors are patched by LoRA. You can use `pipe.clear_lora()` to clear all LoRA layers."
                )
        else:
            lora_loader.fuse_lora_to_base_model(module, lora, alpha=alpha)

    def clear_lora(self, verbose=1):
        """Clear lora.

        Args:
            verbose: The verbose.
        """
        cleared_num = 0
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear):
                if hasattr(module, "lora_A_weights"):
                    if len(module.lora_A_weights) > 0:
                        cleared_num += 1
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()
        if verbose >= 1:
            print(f"{cleared_num} LoRA layers are cleared.")

    def download_and_load_models(self, model_configs: list[ModelConfig] = [], vram_limit: float = None):
        """Download and load models.

        Args:
            model_configs: The model configs.
            vram_limit: The vram limit.
        """
        if self.model_pool_factory is None:
            raise RuntimeError(
                "staged pipeline checkpoint discovery requires an explicit model_pool_factory"
            )
        model_pool = self.model_pool_factory(model_configs=self.model_pool_configs)
        for model_config in model_configs:
            model_config.download_if_necessary()
            vram_config = model_config.vram_config()
            vram_config["computation_dtype"] = vram_config["computation_dtype"] or self.torch_dtype
            vram_config["computation_device"] = vram_config["computation_device"] or self.device
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                vram_limit=vram_limit,
                clear_parameters=model_config.clear_parameters,
                state_dict=model_config.state_dict,
            )
        return model_pool

    def check_vram_management_state(self):
        """Check vram management state."""
        vram_management_enabled = False
        for module in self.children():
            if hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled"):
                vram_management_enabled = True
        return vram_management_enabled

    def cfg_guided_model_fn(self, model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others):
        """Classic CFG: ε_neg + cfg_scale * (ε_pos - ε_neg). Tuple outputs are guided per-element.

        When cfg_scale == 1.0 the negative pass is skipped. Optional positive_only_lora
        is fused only for the positive branch, then cleared.
        """
        if inputs_shared.get("positive_only_lora", None) is not None:
            self.clear_lora(verbose=0)
            self.load_lora(self.dit, state_dict=inputs_shared["positive_only_lora"], verbose=0)
        noise_pred_posi = model_fn(**inputs_posi, **inputs_shared, **inputs_others)
        if cfg_scale != 1.0:
            if inputs_shared.get("positive_only_lora", None) is not None:
                self.clear_lora(verbose=0)
            noise_pred_nega = model_fn(**inputs_nega, **inputs_shared, **inputs_others)
            if isinstance(noise_pred_posi, tuple):
                # Separately handling different output types of latents, eg. video and audio latents.
                noise_pred = tuple(
                    n_nega + cfg_scale * (n_posi - n_nega) for n_posi, n_nega in zip(noise_pred_posi, noise_pred_nega)
                )
            else:
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        return noise_pred


class InferenceStageGraph:
    """Dependency graph for staged inference inputs and outputs.

    Used to split units into model-related vs unrelated subgraphs so VRAM
    onload can keep only the stages that touch named models.
    """

    def __init__(self):
        """Init."""
        pass

    def build_edges(self, units: list[InferenceStage]):
        """Build edges.

        Args:
            units: The units.
        """
        # Establish dependencies between units
        # to search for subsequent related computation units.
        last_compute_unit_id = {}
        edges = []
        for unit_id, unit in enumerate(units):
            for input_param in unit.fetch_input_params():
                if input_param in last_compute_unit_id:
                    edges.append((last_compute_unit_id[input_param], unit_id))
            for output_param in unit.fetch_output_params():
                last_compute_unit_id[output_param] = unit_id
        return edges

    def build_chains(self, units: list[InferenceStage]):
        """Build chains.

        Args:
            units: The units.
        """
        # Establish updating chains for each variable
        # to track their computation process.
        params = sum([unit.fetch_input_params() + unit.fetch_output_params() for unit in units], [])
        params = sorted(list(set(params)))
        chains = {param: [] for param in params}
        for unit_id, unit in enumerate(units):
            for param in unit.fetch_output_params():
                chains[param].append(unit_id)
        return chains

    def search_direct_unit_ids(self, units: list[InferenceStage], model_names: list[str]):
        """Search direct unit ids.

        Args:
            units: The units.
            model_names: The model names.
        """
        # Search for units that directly participate in the model's computation.
        related_unit_ids = []
        for unit_id, unit in enumerate(units):
            for model_name in model_names:
                if unit.onload_model_names is not None and model_name in unit.onload_model_names:
                    related_unit_ids.append(unit_id)
                    break
        return related_unit_ids

    def search_related_unit_ids(self, edges, start_unit_ids, direction="target"):
        """Search related unit ids.

        Args:
            edges: The edges.
            start_unit_ids: The start unit ids.
            direction: The direction.
        """
        # Search for subsequent related computation units.
        related_unit_ids = [unit_id for unit_id in start_unit_ids]
        while True:
            neighbors = []
            for source, target in edges:
                if direction == "target" and source in related_unit_ids and target not in related_unit_ids:
                    neighbors.append(target)
                elif direction == "source" and source not in related_unit_ids and target in related_unit_ids:
                    neighbors.append(source)
            neighbors = sorted(list(set(neighbors)))
            if len(neighbors) == 0:
                break
            else:
                related_unit_ids.extend(neighbors)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids

    def search_updating_unit_ids(self, units: list[InferenceStage], chains, related_unit_ids):
        """Search updating unit ids.

        Args:
            units: The units.
            chains: The chains.
            related_unit_ids: The related unit ids.
        """
        # If the input parameters of this subgraph are updated outside the subgraph,
        # search for the units where these updates occur.
        first_compute_unit_id = {}
        for unit_id in related_unit_ids:
            for param in units[unit_id].fetch_input_params():
                if param not in first_compute_unit_id:
                    first_compute_unit_id[param] = unit_id
        updating_unit_ids = []
        for param in first_compute_unit_id:
            unit_id = first_compute_unit_id[param]
            chain = chains[param]
            if unit_id in chain and chain.index(unit_id) != len(chain) - 1:
                for unit_id_ in chain[chain.index(unit_id) + 1 :]:
                    if unit_id_ not in related_unit_ids:
                        updating_unit_ids.append(unit_id_)
        related_unit_ids.extend(updating_unit_ids)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids

    def split_pipeline_units(self, units: list[InferenceStage], model_names: list[str]):
        """Split pipeline units.

        Args:
            units: The units.
            model_names: The model names.
        """
        # Split the computation graph,
        # separating all model-related computations.
        related_unit_ids = self.search_direct_unit_ids(units, model_names)
        edges = self.build_edges(units)
        chains = self.build_chains(units)
        while True:
            num_related_unit_ids = len(related_unit_ids)
            related_unit_ids = self.search_related_unit_ids(edges, related_unit_ids, "target")
            related_unit_ids = self.search_updating_unit_ids(units, chains, related_unit_ids)
            if len(related_unit_ids) == num_related_unit_ids:
                break
            else:
                num_related_unit_ids = len(related_unit_ids)
        related_units = [units[i] for i in related_unit_ids]
        unrelated_units = [units[i] for i in range(len(units)) if i not in related_unit_ids]
        return related_units, unrelated_units


class InferenceStageRunner:
    """Execute an :class:`InferenceStage` against shared/CFG state maps."""

    def __init__(self):
        """Init."""
        pass

    def __call__(
        self,
        unit: InferenceStage,
        pipe: StagedDiffusionPipeline,
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
    ) -> tuple[dict, dict]:
        """Dispatch one stage against shared / CFG state maps.

        take_over: the stage replaces this method. seperate_cfg: run process()
        on positive, then on negative only if cfg_scale != 1 (else reuse the
        positive outputs). Otherwise write into inputs_shared only.
        """
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(
                pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega
            )
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


__all__ = [
    "InferenceStage",
    "InferenceStageGraph",
    "InferenceStageRunner",
    "StagedDiffusionPipeline",
]
