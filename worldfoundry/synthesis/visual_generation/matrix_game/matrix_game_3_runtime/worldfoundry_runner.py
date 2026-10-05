from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
import torch
import torch.distributed as dist


def patch_torch_dynamo_config_aliases():
    """Bridge renamed torch._dynamo config keys used by Matrix-Game-3 upstream code."""
    import torch._dynamo

    alias_map = {
        "recompile_limit": "cache_size_limit",
        "accumulated_recompile_limit": "accumulated_cache_size_limit",
    }
    config_module = torch._dynamo.config
    config_class = type(config_module)

    if getattr(config_class, "_worldfoundry_native_alias_patch_installed", False):
        return

    if all(alias in getattr(config_module, "_config", {}) for alias in alias_map):
        return

    original_setattr = config_class.__setattr__
    original_getattr = config_class.__getattr__

    def compat_setattr(self, name, value):
        target = alias_map.get(name)
        if target and target in self._config and name not in self._config:
            setattr(self, target, value)
            return
        return original_setattr(self, name, value)

    def compat_getattr(self, name):
        target = alias_map.get(name)
        if target and target in self._config and name not in self._config:
            return getattr(self, target)
        return original_getattr(self, name)

    config_class.__setattr__ = compat_setattr
    config_class.__getattr__ = compat_getattr
    config_class._worldfoundry_native_alias_patch_installed = True


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix-Game-3 subprocess runner with custom actions.")
    parser.add_argument("--runtime_root", required=True, type=str)
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--actions_json", required=True, type=str)
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--save_name", required=True, type=str)
    parser.add_argument("--size", default="704*1280", type=str)
    parser.add_argument("--fps", default=17, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_iterations", default=1, type=int)
    parser.add_argument("--num_inference_steps", default=3, type=int)
    parser.add_argument("--sample_shift", default=None, type=float)
    parser.add_argument("--sample_guide_scale", default=5.0, type=float)
    parser.add_argument("--ulysses_size", default=1, type=int)
    parser.add_argument("--vae_type", default="wan", type=str)
    parser.add_argument("--lightvae_pruning_rate", default=None, type=float)
    parser.add_argument("--fa_version", default=None, type=str, choices=["0", "2", "3"])
    parser.add_argument("--visualize_ops", action="store_true")
    parser.add_argument("--use_base_model", action="store_true")
    parser.add_argument("--use_int8", action="store_true")
    parser.add_argument("--verify_quant", action="store_true")
    parser.add_argument("--use_async_vae", action="store_true")
    parser.add_argument("--async_vae_warmup_iters", default=0, type=int)
    parser.add_argument("--compile_vae", action="store_true")
    parser.add_argument("--t5_fsdp", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--dit_fsdp", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")
    return parser.parse_args()


def _init_logging(rank: int):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def _prepare_distributed(runtime_args, cfg):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    _init_logging(rank)

    if runtime_args.ulysses_size <= 1:
        if runtime_args.t5_fsdp or runtime_args.dit_fsdp:
            logging.info(
                "Single GPU detected (ulysses_size=%s). Automatically disabling FSDP.",
                runtime_args.ulysses_size,
            )
            runtime_args.t5_fsdp = False
            runtime_args.dit_fsdp = False

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
    else:
        if runtime_args.t5_fsdp or runtime_args.dit_fsdp:
            raise RuntimeError("t5_fsdp and dit_fsdp are not supported in non-distributed environments.")
        if runtime_args.ulysses_size > 1:
            raise RuntimeError("sequence parallel is not supported in non-distributed environments.")

    if runtime_args.ulysses_size > 1:
        if runtime_args.ulysses_size != world_size:
            raise RuntimeError("The number of ulysses_size should be equal to the world size.")
        from worldfoundry.core.distributed.sequence_parallel.ops import init_distributed_group

        init_distributed_group()

    if runtime_args.ulysses_size > 1 and cfg.num_heads % runtime_args.ulysses_size != 0 and rank == 0:
        logging.warning(
            "`cfg.num_heads=%s` is not divisible by `ulysses_size=%s`. "
            "Ulysses attention will pad heads internally and trim back after communication.",
            cfg.num_heads,
            runtime_args.ulysses_size,
        )

    return rank, world_size, local_rank


def normalize_to_neg_one_to_one(x):
    return 2.0 * x - 1.0


def build_runtime_args(args):
    return SimpleNamespace(
        ckpt_dir=args.ckpt_dir,
        size=args.size,
        save_name=args.save_name,
        seed=args.seed,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
        num_inference_steps=args.num_inference_steps,
        ulysses_size=args.ulysses_size,
        t5_fsdp=args.t5_fsdp,
        t5_cpu=args.t5_cpu,
        dit_fsdp=args.dit_fsdp,
        convert_model_dtype=args.convert_model_dtype,
        use_base_model=args.use_base_model,
        use_int8=args.use_int8,
        verify_quant=args.verify_quant,
        vae_type=args.vae_type,
        lightvae_pruning_rate=args.lightvae_pruning_rate,
        use_async_vae=args.use_async_vae,
        async_vae_warmup_iters=args.async_vae_warmup_iters,
        compile_vae=args.compile_vae,
        fa_version=args.fa_version,
    )


def main():
    args = parse_args()
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    for import_root in (runtime_root,):
        root_str = str(import_root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
    patch_torch_dynamo_config_aliases()

    with open(args.actions_json, "r", encoding="utf-8") as file:
        action_spec = json.load(file)
    keyboard_condition_np = np.asarray(action_spec["keyboard_condition"], dtype=np.float32)
    mouse_condition_np = np.asarray(action_spec["mouse_condition"], dtype=np.float32)

    from PIL import Image

    from worldfoundry.base_models.diffusion_model.recipes.wan_configs.action_22 import MAX_AREA_CONFIGS, WAN_CONFIGS
    import pipeline.inference_pipeline as mg3_inference
    from utils.cam_utils import get_extrinsics
    from worldfoundry.core.utils.tensors.torch import set_seed_everywhere
    from utils.transform import get_video_transform
    from utils.utils import compute_all_poses_from_actions
    import utils.visualize as visualize_utils

    def custom_get_data(num_frames, height, width, pil_image, device=None, dtype=None):
        if keyboard_condition_np.shape[0] != num_frames or mouse_condition_np.shape[0] != num_frames:
            raise ValueError(
                f"Custom action sequence length mismatch: "
                f"expected {num_frames}, got keyboard={keyboard_condition_np.shape[0]}, mouse={mouse_condition_np.shape[0]}"
            )

        input_image = torch.from_numpy(np.array(pil_image)).unsqueeze(0)
        input_image = input_image.permute(0, 3, 1, 2)
        transform = get_video_transform(height, width, normalize_to_neg_one_to_one)
        input_image = transform(input_image)
        input_image = input_image.transpose(0, 1).unsqueeze(0)

        first_pose = np.zeros(5, dtype=np.float32)
        all_poses = compute_all_poses_from_actions(
            keyboard_condition_np,
            mouse_condition_np,
            first_pose=first_pose,
        )
        positions = all_poses[:, :3].tolist()
        rotations = np.concatenate(
            [
                np.zeros((all_poses.shape[0], 1), dtype=np.float32),
                all_poses[:, 3:5],
            ],
            axis=1,
        ).tolist()
        extrinsics_all = get_extrinsics(rotations, positions)

        keyboard_tensor = torch.tensor(keyboard_condition_np, dtype=dtype, device=device).unsqueeze(0)
        mouse_tensor = torch.tensor(mouse_condition_np, dtype=dtype, device=device).unsqueeze(0)
        return (
            input_image.to(device, dtype),
            extrinsics_all,
            keyboard_tensor,
            mouse_tensor,
        )

    def patched_process_video(
        input_video,
        output_video,
        config,
        mouse_icon_path,
        mouse_scale=1.0,
        mouse_rotation=0,
        default_frame_res=(704, 1280),
    ):
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.visualize_ops:
            return visualize_utils.process_video(
                input_video,
                output_video,
                config,
                mouse_icon_path,
                mouse_scale=mouse_scale,
                mouse_rotation=mouse_rotation,
                default_frame_res=default_frame_res,
            )
        imageio.mimsave(str(output_path), input_video, fps=args.fps)

    mg3_inference.get_data = custom_get_data
    mg3_inference.process_video = patched_process_video

    runtime_args = build_runtime_args(args)
    cfg = WAN_CONFIGS["matrix_game3"]
    if runtime_args.sample_shift is None:
        runtime_args.sample_shift = cfg.sample_shift
    if runtime_args.sample_guide_scale is None:
        runtime_args.sample_guide_scale = cfg.sample_guide_scale

    rank, world_size, local_rank = _prepare_distributed(runtime_args, cfg)
    set_seed_everywhere(runtime_args.seed)
    if dist.is_initialized():
        seed = [runtime_args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed, src=0)
        runtime_args.seed = seed[0]

    if rank == 0:
        logging.info("Generation job args: %s", runtime_args)
        logging.info("Generation model config: %s", cfg)

    pipeline = mg3_inference.MatrixGame3Pipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=runtime_args.t5_fsdp,
        dit_fsdp=runtime_args.dit_fsdp,
        use_sp=(runtime_args.ulysses_size > 1),
        t5_cpu=runtime_args.t5_cpu,
        convert_model_dtype=runtime_args.convert_model_dtype,
        args=runtime_args,
        fa_version=runtime_args.fa_version,
        use_base_model=runtime_args.use_base_model,
    )

    pil_image = Image.open(args.image_path).convert("RGB")
    pipeline.generate(
        args.prompt,
        pil_image,
        max_area=MAX_AREA_CONFIGS[args.size],
        shift=runtime_args.sample_shift,
        num_inference_steps=runtime_args.num_inference_steps,
        guide_scale=runtime_args.sample_guide_scale,
        seed=runtime_args.seed,
        use_base_model=runtime_args.use_base_model,
        args=runtime_args,
    )


if __name__ == "__main__":
    main()
