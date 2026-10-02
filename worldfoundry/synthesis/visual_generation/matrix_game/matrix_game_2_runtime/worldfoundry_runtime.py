from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path
from worldfoundry.core.media.artifacts import process_game_control_video as process_video
from worldfoundry.core.utils.tensors.torch import set_seed_everywhere as set_seed
from worldfoundry.evaluation.utils import worldfoundry_data_path

MATRIX_GAME_2_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "matrix_game_2")
MODE_CHECKPOINTS = {
    "universal": ("base_distilled_model", "base_distill.safetensors"),
    "gta_drive": ("gta_distilled_model", "gta_keyboard2dim.safetensors"),
    "templerun": ("templerun_distilled_model", "templerun_7dim_onlykey.safetensors"),
}
MODE_CONFIGS = {
    "universal": "inference_yaml/inference_universal.yaml",
    "gta_drive": "inference_yaml/inference_gta_drive.yaml",
    "templerun": "inference_yaml/inference_templerun.yaml",
}


def _hf_snapshot_dirs(root: Path) -> list[Path]:
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(path for path in snapshots.iterdir() if path.is_dir())


def _matrix_game2_roots(primary: str | Path | None = None) -> list[Path]:
    raw: list[Path] = []
    if primary not in {None, ""}:
        raw.append(Path(str(primary)).expanduser())
    raw.extend(
        [
            checkpoint_root_path("Matrix-Game-2.0"),
            hfd_root_path("Skywork--Matrix-Game-2.0"),
            hfd_root_path("custom--Matrix-Game-2.0"),
            checkpoint_root_path("huggingface", "hub", "models--Skywork--Matrix-Game-2.0"),
            checkpoint_root_path("hf_home", "hub", "models--Skywork--Matrix-Game-2.0"),
        ]
    )

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in raw:
        if root.is_file():
            root = root.parent
        for candidate in [root, *_hf_snapshot_dirs(root)]:
            key = str(candidate)
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)
    return candidates


def _matrix_game2_layout_complete(root: Path, mode: str) -> bool:
    rel_dir, filename = MODE_CHECKPOINTS[mode]
    required = [
        root / "Wan2.1_VAE.pth",
        root / "xlm-roberta-large",
        root / rel_dir / filename,
    ]
    return all(path.exists() for path in required)


def _resolve_model_root(path_value: str | Path, mode: str) -> str:
    for candidate in _matrix_game2_roots(path_value):
        if _matrix_game2_layout_complete(candidate, mode):
            return str(candidate.resolve())
    for candidate in _matrix_game2_roots(path_value):
        if candidate.is_dir():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "Matrix-Game-2 requires a local checkpoint directory. "
        f"Checked: {[str(path) for path in _matrix_game2_roots(path_value)]}"
    )


def _enable_torch_compile() -> bool:
    return os.environ.get("WORLDFOUNDRY_ENABLE_TORCH_COMPILE", "").lower() in {"1", "true", "yes"}


_MATRIX_GAME2_OPTIMIZATION_KEYS = {
    "fuse_qkv", "qkv_strategy", "qkv_split_threshold", "quantization", "compile",
    "compile_backend", "compile_mode", "compile_dynamic", "compile_fullgraph", "compile_options",
    "attention", "attention_backend", "offload", "offload_mode",
    "vae_channels_last", "vae_channels_last_3d",
}
_MATRIX_GAME2_UNSUPPORTED_OPTIONS = {
    "adacache", "approximate_attention", "blocktaylorseer", "cfg_gate_fraction", "cfg_gate_step",
    "cfg_parallel", "cfg_parallel_degree", "cuda_graph", "custom", "device_map", "dit_weight_dtype",
    "feature_cache", "fused_residual_adaln", "fused_rope", "inplace_residual", "magcache",
    "rope_precision", "rms_norm_precision", "sequence_parallel", "sp_degree", "static_cross_kv",
    "taylorseer", "teacache", "teacache_thresh", "vae_decode_autocast",
    "vae_parallel", "vae_parallel_degree", "vae_preview_decoder_path", "vae_spatial_tiling",
    "vae_temporal_chunk_size", "vae_tile_size", "vae_tile_stride", "vae_tiled_decode", "vae_weight_dtype",
}


def _matrix_game2_runtime_policy(kwargs, *, device, dtype):
    """Validate public optimization requests before touching configs or weights."""
    from worldfoundry.base_models.diffusion_model.loaders.module import _compile_policy_from_runtime
    from worldfoundry.base_models.diffusion_model.optimizations.policy import (
        parse_attention_backend,
        parse_offload_policy,
        parse_quantization_policy,
    )
    from worldfoundry.core.model_loading.policy import AttentionBackend, OffloadMode, RuntimePolicy

    raw_options = kwargs.get("runtime_options")
    if raw_options is None:
        options = {}
    elif isinstance(raw_options, Mapping):
        options = dict(raw_options)
    else:
        raise TypeError("Matrix-Game-2 runtime_options must be a mapping")
    known = _MATRIX_GAME2_OPTIMIZATION_KEYS | _MATRIX_GAME2_UNSUPPORTED_OPTIONS
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"unsupported Matrix-Game-2 runtime_options: {sorted(unknown)}")
    for key in known:
        if key in kwargs and kwargs[key] is not None:
            options[key] = kwargs[key]
    for key in _MATRIX_GAME2_UNSUPPORTED_OPTIONS:
        value = options.get(key)
        disabled = value is None or value is False or value == 0
        if isinstance(value, str):
            disabled = value.strip().lower() in {"", "none", "false", "0"}
        if not disabled:
            raise ValueError(f"Matrix-Game-2 does not support requested optimization {key!r}")

    for key in ("fuse_qkv", "compile", "vae_channels_last", "vae_channels_last_3d"):
        value = options.get(key, False)
        if not isinstance(value, bool):
            raise TypeError(f"Matrix-Game-2 {key} must be a bool")
    strategy = str(options.get("qkv_strategy", "packed")).strip().lower()
    if strategy not in {"auto", "packed", "split"}:
        raise ValueError("Matrix-Game-2 qkv_strategy must be 'auto', 'packed', or 'split'")
    threshold = options.get("qkv_split_threshold", 8192)
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
        raise ValueError("Matrix-Game-2 qkv_split_threshold must be a positive integer")
    options.update(qkv_strategy=strategy, qkv_split_threshold=threshold)

    for key in ("attention", "attention_backend"):
        if key in options and parse_attention_backend(options[key], owner="Matrix-Game-2") is not AttentionBackend.AUTO:
            raise ValueError(f"Matrix-Game-2 does not support requested {key}; its causal attention owns the provider")
    for key in ("offload", "offload_mode"):
        if key in options and options[key] is not None:
            offload = parse_offload_policy(options[key], owner="Matrix-Game-2")
            if offload.mode is not OffloadMode.NONE:
                raise ValueError(f"Matrix-Game-2 does not support requested {key}")
    policy = RuntimePolicy(
        device=device or "cpu", dtype=dtype,
        quantization=parse_quantization_policy(options.get("quantization"), owner="Matrix-Game-2"),
        compile=options.get("compile", False), options=options,
    )
    # Validate backend/mode/options even before lazy compiler installation.
    _compile_policy_from_runtime(policy)
    return policy


def _apply_matrix_game2_optimizations(model, policy):
    """Apply build transforms to the restored causal DiT, retaining its type."""
    from worldfoundry.base_models.diffusion_model.loaders.module import _compile_policy_from_runtime
    from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections
    from worldfoundry.core.attention.backends.dispatch import attention_compile_receipt_scope
    from worldfoundry.core.execution.compile_cache import compile_callable_cached
    from worldfoundry.core.model_loading.optimize import AppliedOptimizations, apply_quantization_policy

    applied = AppliedOptimizations()
    model._worldfoundry_applied_optimizations = applied
    requested_fusion = policy.options.get("fuse_qkv", False)
    strategy = policy.options["qkv_strategy"]
    threshold = policy.options["qkv_split_threshold"]
    fused = fuse_qkv_projections(model, strategy=strategy, split_threshold=threshold) if requested_fusion else 0
    applied.record_fusion(requested=requested_fusion, fused_blocks=fused, strategy=strategy, split_threshold=threshold)
    applied.record_quantization(apply_quantization_policy(model, policy.quantization))
    applied.record_compile(requested=False)
    if policy.compile:
        compile_policy, compile_options = _compile_policy_from_runtime(policy)
        eager_forward = model.forward
        compiled_forward = compile_callable_cached(
            eager_forward, policy=compile_policy, options=compile_options, namespace="matrix-game-2-dit-forward",
        )
        installed = compiled_forward is not eager_forward
        compile_runtime = {
            "wrapper_installed": installed, "calls": 0, "failures": 0, "last_error": None,
            "request_calls": 0, "request_failures": 0, "request_last_error": None,
            "attention_provider_graph_traces": {},
        }
        if installed:
            def audited_compiled_forward(*args, **kwargs):
                try:
                    with attention_compile_receipt_scope(compile_runtime["attention_provider_graph_traces"]):
                        result = compiled_forward(*args, **kwargs)
                except Exception as error:
                    compile_runtime["failures"] += 1
                    compile_runtime["request_failures"] += 1
                    message = f"{type(error).__name__}: {error}"
                    compile_runtime["last_error"] = compile_runtime["request_last_error"] = message
                    raise
                compile_runtime["calls"] += 1
                compile_runtime["request_calls"] += 1
                return result

            model.forward = audited_compiled_forward
        model._worldfoundry_compile_runtime = compile_runtime
        model._worldfoundry_compile_config = {
            "backend": compile_policy.backend, "mode": compile_policy.mode,
            "fullgraph": compile_policy.fullgraph, "dynamic": compile_policy.dynamic, "options": compile_options,
        }
        applied.record_compile(requested=True, compiled=installed, details=model._worldfoundry_compile_config)
    return applied


class MatrixGame2Runtime:
    def __init__(
        self,
        pipeline,
        vae,
        weight_dtype: torch.dtype | None = None,
        mode: str = "universal",
        device: str = "cuda",
    ):
        """
        the mode including "gta_drive", "templerun", "universal"
        """
        self.pipeline = pipeline
        self.vae = vae
        self.weight_dtype = weight_dtype or torch.bfloat16
        self.device = device
        self.mode = mode

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        mode: str = "universal",
        device=None,
        weight_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "MatrixGame2Runtime":
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        if mode not in ["universal", "gta_drive", "templerun"]:
            raise NotImplementedError("mode should be one of ['universal', 'gta_drive', 'templerun']")
        weight_dtype = weight_dtype or torch.bfloat16
        runtime_policy = _matrix_game2_runtime_policy(kwargs, device=device, dtype=weight_dtype)
        config_path = str(MATRIX_GAME_2_CONFIG_ROOT / MODE_CONFIGS[mode])

        config = OmegaConf.load(config_path)
        model_config = config["model_kwargs"]["model_config"]
        model_config_path = (
            os.fspath(model_config)
            if os.path.isabs(os.fspath(model_config))
            else os.fspath(MATRIX_GAME_2_CONFIG_ROOT / os.fspath(model_config))
        )
        if os.path.isdir(model_config_path):
            model_config_path = os.path.join(model_config_path, "config.yaml")
        if model_config_path.endswith((".yaml", ".yml")):
            config["model_kwargs"]["model_config"] = OmegaConf.to_container(
                OmegaConf.load(model_config_path),
                resolve=True,
            )
        else:
            config["model_kwargs"]["model_config"] = model_config_path

        model_root = _resolve_model_root(pretrained_model_path, mode)

        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.extension_modules.wanx_vae.wanx_vae import (
            get_wanx_vae_wrapper,
        )
        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline import (
            CausalInferencePipeline,
        )
        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.vae_runtime.vae_block3 import (
            VAEDecoderWrapper,
        )
        from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.wan_wrapper import (
            WanDiffusionWrapper,
        )

        generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(
            os.path.join(model_root, "Wan2.1_VAE.pth"),
            map_location="cpu",
            weights_only=True,
        )
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if "decoder." in key or "conv2" in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        vae_layout = None
        if runtime_policy.options.get("vae_channels_last", False) or runtime_policy.options.get("vae_channels_last_3d", False):
            from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

            vae_layout = convert_convolution_weight_layouts(
                current_vae_decoder,
                conv2d=runtime_policy.options.get("vae_channels_last", False),
                conv3d=runtime_policy.options.get("vae_channels_last_3d", False),
            )
            current_vae_decoder._worldfoundry_convolution_layout = vae_layout
        if _enable_torch_compile():
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        pipeline = CausalInferencePipeline(config, generator=generator, vae_decoder=current_vae_decoder)

        resolved_checkpoint_path = cls._resolve_checkpoint_path(
            model_root=model_root,
            mode=mode,
            checkpoint_path=checkpoint_path,
        )
        print(f"Loading Pretrained Model from {resolved_checkpoint_path}...")
        state_dict = load_file(resolved_checkpoint_path)
        pipeline.generator.load_state_dict(state_dict)

        # Place components separately: casting the whole pipeline to BF16 and
        # then the decoder back to FP16 irreversibly rounds its loaded weights.
        pipeline = pipeline.to(device=device)
        pipeline.generator.to(dtype=weight_dtype)
        applied = _apply_matrix_game2_optimizations(pipeline.generator.model, runtime_policy)
        pipeline.generator._worldfoundry_applied_optimizations = applied
        applied.requested["vae_channels_last"] = runtime_policy.options.get("vae_channels_last", False)
        applied.requested["vae_channels_last_3d"] = runtime_policy.options.get("vae_channels_last_3d", False)
        if vae_layout is not None:
            pipeline._worldfoundry_vae_convolution_layout = vae_layout
            applied.effective["vae_convolution_layout"] = {
                name: getattr(vae_layout, name)
                for name in ("conv2d_total", "conv2d_converted", "conv3d_total", "conv3d_converted")
            }

        vae = get_wanx_vae_wrapper(model_root, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, weight_dtype)

        runtime = cls(pipeline=pipeline, vae=vae, weight_dtype=weight_dtype, mode=mode, device=device)
        runtime._worldfoundry_applied_optimizations = applied
        return runtime

    @staticmethod
    def _resolve_checkpoint_path(model_root: str, mode: str, checkpoint_path: str | None = None) -> str:
        if checkpoint_path is None:
            rel_dir, filename = MODE_CHECKPOINTS[mode]
            resolved = os.path.join(model_root, rel_dir, filename)
        elif os.path.isabs(checkpoint_path):
            resolved = checkpoint_path
        else:
            resolved = os.path.join(model_root, checkpoint_path)

        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Matrix-Game-2 checkpoint not found for mode='{mode}': {resolved}")
        return resolved

    @torch.no_grad()
    def predict(
        self,
        cond_concat,
        visual_context,
        operator_condition,
        num_output_frames,
        operation_visualization=True,
        seed=None,
    ):
        if seed is not None:
            # MG2's denoising loop also samples fresh noise inside the pipeline,
            # so we need to seed the global RNG chain, not only the initial tensor.
            set_seed(int(seed))
        sampled_noise = torch.randn(
            [1, 16, num_output_frames, cond_concat.size(-2), cond_concat.size(-1)],
            device=self.device,
            dtype=self.weight_dtype,
        )

        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype),
        }
        if "mouse_condition" in operator_condition:
            mouse_condition = operator_condition["mouse_condition"].unsqueeze(0).to(
                device=self.device,
                dtype=self.weight_dtype,
            )
            conditional_dict["mouse_cond"] = mouse_condition
        if "keyboard_condition" not in operator_condition:
            raise ValueError("keyboard_condition must be provided in operator_condition")
        keyboard_condition = operator_condition["keyboard_condition"].unsqueeze(0).to(
            device=self.device,
            dtype=self.weight_dtype,
        )
        conditional_dict["keyboard_cond"] = keyboard_condition

        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                mode=self.mode,
                profile=False,
            )

        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)

        mouse_icon = None
        if self.mode != "templerun":
            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy(),
            )
        else:
            config = keyboard_condition[0].float().cpu().numpy()
        output_video = process_video(
            video.astype(np.uint8),
            config,
            mouse_icon,
            mouse_scale=0.1,
            process_icon=operation_visualization,
            mode=self.mode,
        )
        return output_video


__all__ = [
    "MATRIX_GAME_2_CONFIG_ROOT",
    "MODE_CHECKPOINTS",
    "MODE_CONFIGS",
    "MatrixGame2Runtime",
    "process_video",
]
