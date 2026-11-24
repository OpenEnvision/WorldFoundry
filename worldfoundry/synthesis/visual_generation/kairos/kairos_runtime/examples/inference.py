import argparse
import os
import time

import mmengine
import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.dist import get_dist_info, init_dist
from PIL import Image

import worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.apis.kairos_embodied_api  # noqa: F401 - registers KairosEmbodiedAPI
from worldfoundry.core.io.paths import package_data_path
from worldfoundry.synthesis.visual_generation.kairos.config import load_kairos_config
from worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.apis.builder import build_model_pipeline
from worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.modules.utils import (
    parallel_state,
    save_image,
    save_video,
)
from worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.modules.utils.prompt_rewriter import (
    PromptRewriter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="TRAIN_MODEL_LOOP")
    parser.add_argument("--input_file", default="", help="input_file")
    parser.add_argument(
        "--config_file",
        default=str(package_data_path("models", "runtime", "configs", "kairos", "kairos_4b_config_DMD.py")),
        help="path to config file",
    )

    args = parser.parse_args()

    return args


def print_gpu_memory(device_id=0):
    """
    Print the PyTorch memory usage of the GPU that belongs to this process.

    The GPU is chosen from ``LOCAL_RANK`` or ``RANK`` (falling back to GPU 0); ``device_id`` is ignored.
    """

    def get_current_rank():
        # LOCAL_RANK covers single-node multi-GPU runs; RANK covers multi-node runs.
        rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
        if rank is None:
            print("Warning: RANK/LOCAL_RANK is not set, using GPU 0")
            return 0
        try:
            return int(rank)
        except ValueError:
            print(f"Warning: rank '{rank}' from the environment is not an integer, using GPU 0")
            return 0

    device_id = get_current_rank()

    if device_id >= torch.cuda.device_count():
        print(
            f"Warning: GPU ID {device_id} for this rank exceeds the available GPUs ({torch.cuda.device_count()}), using GPU 0"
        )
        device_id = 0
    torch.cuda.set_device(device_id)

    used = torch.cuda.memory_allocated(device_id) / 1024**3  # GiB held by tensors
    cached = torch.cuda.memory_reserved(device_id) / 1024**3  # GiB reserved by the caching allocator
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3

    print(f"GPU {device_id} memory:")
    print(f"  Allocated: {used:.2f} GB")
    print(f"  Reserved: {cached:.2f} GB")
    print(f"  Total: {total:.2f} GB")
    print(f"  Allocated ratio: {used / total * 100:.2f}%")


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    init_dist(launcher="pytorch")
    rank, world_size = get_dist_info()

    print(f"RANK: {rank} || WORLD_SIZE: {world_size}")

    args = parse_args()

    cfg_path = args.config_file

    if cfg_path == "":
        ValueError("config path is empty")
        exit()

    input_file = args.input_file
    input_args_d = mmengine.load(input_file)

    output_dir = input_args_d["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/output.mp4"

    cfg = Config(load_kairos_config(cfg_path))

    use_prompt_rewriter = input_args_d["use_prompt_rewriter"]
    if rank == 0 and use_prompt_rewriter:
        prompt_rewriter_path = cfg.prompt_rewriter_path
        prompt_rewriter = PromptRewriter(prompt_rewriter_path)

    # ***********************************************************
    # Initialize parallel state
    # ---- defaults: force no parallel unless multi-GPU + dist initialized ----
    use_cfg_parallel = cfg.pipeline.get("use_cfg_parallel")
    parallel_state.reset_cfg()
    is_multi_gpu_dist = (world_size > 1) and dist.is_initialized()

    # Enforce: no parallelism in non-multi-GPU environments
    if not is_multi_gpu_dist:
        use_cfg_parallel = False  # treat cfg-parallel as OFF
        if rank == 0:
            print("[init] dist=OFF or WORLD_SIZE=1 -> force parallel OFF ", flush=True)
    else:
        parallel_state.set_vae_group(dist.group.WORLD)

        if use_cfg_parallel:
            cfg_size = 2  # fixed 2-way CFG (pos/neg)
            assert world_size in (4, 8), (
                f"CFG-parallel only supports WORLD_SIZE in {{4, 8}}, got {world_size}. "
                f"Please re-launch with 4 or 8 GPUs, or disable CFG-parallel."
            )
            tp_size = world_size // cfg_size  # -> 2 or 4
            assert tp_size in (2, 4), (
                f"CFG-parallel derived tp_size={tp_size} is not supported. Supported tp_size: {{2, 4}}"
            )
        else:
            cfg_size = 1
            tp_size = world_size

        # ---- TP group ----
        if tp_size == world_size:
            tp_group = dist.group.WORLD
            tp_rank = rank
            tp_gid = 0
        else:
            tp_group, tp_rank, tp_gid = parallel_state.init_tp_groups(tp_size=tp_size)

        parallel_state.set_tp_group(tp_group)

        # ---- CFG group (only when cfg-parallel enabled) ----
        if use_cfg_parallel:
            # build all cfg groups in same order on all ranks
            cfg_groups = []
            for tp_r in range(tp_size):
                ranks_g = [tp_r + i * tp_size for i in range(cfg_size)]
                cfg_groups.append(dist.new_group(ranks=ranks_g))

            cfg_group = cfg_groups[tp_rank]
            cfg_rank = rank // tp_size
            parallel_state.set_cfg_group(cfg_group, cfg_rank=cfg_rank, cfg_size=cfg_size)
            print(
                f"[init] rank={rank} tp_rank={tp_rank} cfg_rank={cfg_rank} "
                f"tp_group={dist.get_world_size(tp_group)} "
                f"cfg_group={dist.get_world_size(cfg_group)}",
                flush=True,
            )
        else:
            if rank == 0:
                print(f"[init] cfg_parallel=OFF tp_size={tp_size}", flush=True)

    print("build pipeline ...")
    pipeline = build_model_pipeline(cfg.pipeline)
    pipeline.eval()
    print("build pipeline done")

    print("start infer ...")
    raw_prompt = input_args_d.get("prompt", "")
    if raw_prompt.strip() != "":
        if use_prompt_rewriter:
            rewritten_prompt = prompt_rewriter.rewrite_prompt(
                raw_prompt, image_path=input_args_d.get("input_image", "")
            )
        else:
            rewritten_prompt = raw_prompt

        input_args_d["raw_prompt"] = raw_prompt
        input_args_d["prompt"] = rewritten_prompt

        print(f"rewritten prompt from [{raw_prompt}] to [{rewritten_prompt}]")

    input_args_d.pop("output_dir")
    input_args_d.pop("use_prompt_rewriter")

    # open image
    if "input_image" in input_args_d and isinstance(input_args_d["input_image"], str):
        if not input_args_d["input_image"]:
            input_args_d.pop("input_image")
        else:
            raw_imgage = input_args_d["input_image"]
            image = Image.open(input_args_d["input_image"])
            input_args_d["input_image"] = [image]

    input_args_d.pop("raw_prompt", "")

    # add prompt prefix in ti2v mode
    if raw_prompt.strip() != "" and "input_image" in input_args_d:
        prompt_prefix = "high-quality video, realistic motion, single continuous shot, no jump cuts, smooth motion. "
        input_args_d["prompt"] = prompt_prefix + input_args_d["prompt"]

    warmup = bool(input_args_d.pop("warmup", False))
    if warmup:
        print("=====================warmup====================")
        pipeline(**input_args_d)

    print_gpu_memory()

    print("=====================infer====================")

    start_time = time.perf_counter()
    video = pipeline(**input_args_d)

    elapsed = time.perf_counter() - start_time

    print(f"infer time: {elapsed:.4f}s")
    print("infer done")

    # save video
    if rank == 0:
        save_fps = 16
        if len(video) == 1:
            save_path = save_path.replace(".mp4", ".jpg")
            save_image(video[0], save_path)
        else:
            save_video(video, save_path, fps=save_fps, quality=5)

    if dist.is_initialized():
        dist.destroy_process_group()
