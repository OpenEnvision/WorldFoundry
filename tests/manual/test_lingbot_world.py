import os
import torch
import torch.distributed as dist
from PIL import Image
from diffusers.utils import export_to_video
from worldfoundry.pipelines.lingbot_world.pipeline_lingbot_world import LingBotPipeline
from worldfoundry.core.distributed.sequence_ops import init_distributed_group


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


image_path = os.getenv(
    "WORLDFOUNDRY_LINGBOT_IMAGE",
    "./worldfoundry/data/test_cases/test_image_case1/ref_image.png",
)
pretrained_model_path = os.getenv(
    "WORLDFOUNDRY_LINGBOT_CKPT_DIR",
    "robbyant/lingbot-world-base-cam",
)
input_image = Image.open(image_path).convert("RGB")
prompt = os.getenv(
    "WORLDFOUNDRY_LINGBOT_PROMPT",
    "A charming medieval village with cobblestone streets, thatched-roof houses.",
)
local_rank = int(os.getenv("LOCAL_RANK", 0))
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)

if world_size > 1 and not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://")
    ulysses_size = world_size
    if ulysses_size > 1:
        init_distributed_group()
else:
    ulysses_size = 1


offload_model = _env_bool("WORLDFOUNDRY_LINGBOT_OFFLOAD_MODEL", world_size <= 1)
pipeline = LingBotPipeline.from_pretrained(
    model_path=pretrained_model_path,
    mode="i2v-A14B",
    device=f"cuda:{local_rank}",
    rank=rank,
    t5_fsdp=(world_size > 1),
    dit_fsdp=(world_size > 1),
    ulysses_size=ulysses_size,
    t5_cpu=_env_bool("WORLDFOUNDRY_LINGBOT_T5_CPU", False),
    offload_model=offload_model,
    runtime_variant=os.getenv("WORLDFOUNDRY_LINGBOT_RUNTIME_VARIANT"),
    fast_model_path=os.getenv("WORLDFOUNDRY_LINGBOT_FAST_CKPT_DIR"),
)

action_commands = [
    item.strip()
    for item in os.getenv("WORLDFOUNDRY_LINGBOT_ACTIONS", "backward,camera_l").split(",")
    if item.strip()
]
action_path = os.getenv("WORLDFOUNDRY_LINGBOT_ACTION_PATH")
action_string = os.getenv("WORLDFOUNDRY_LINGBOT_ACTION_STRING")
sampling_steps = os.getenv("WORLDFOUNDRY_LINGBOT_SAMPLING_STEPS")
pipeline_kwargs = {}
if sampling_steps is not None:
    pipeline_kwargs["sampling_steps"] = int(sampling_steps)

output_video = pipeline(
    images=input_image,
    num_frames=int(os.getenv("WORLDFOUNDRY_LINGBOT_NUM_FRAMES", "161")),
    prompt=prompt,
    interactions=None if action_path else action_commands,
    action_path=action_path,
    allow_act2cam=_env_bool("WORLDFOUNDRY_LINGBOT_ALLOW_ACT2CAM", False),
    action_string=action_string,
    max_area=int(os.getenv("WORLDFOUNDRY_LINGBOT_MAX_AREA", str(480 * 832))),
    seed=int(os.getenv("WORLDFOUNDRY_LINGBOT_SEED", "42")),
    **pipeline_kwargs,
)

if rank == 0 and output_video is not None:
    output_path = os.getenv("WORLDFOUNDRY_LINGBOT_OUTPUT", "lingbot_command_demo.mp4")
    export_to_video(output_video, output_path, fps=16)
    print(f"Done! Video saved to {output_path}.")

if dist.is_initialized():
    dist.destroy_process_group()
