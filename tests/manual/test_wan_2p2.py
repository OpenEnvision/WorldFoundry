import os
from pathlib import Path
from PIL import Image

from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline
from worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.utils.utils import save_video
from worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.configs import WAN_CONFIGS


model_path: str = os.environ.get("WAN22_MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B")
mode = os.environ.get("WAN22_MODE", "ti2v-5B")
device = int(os.environ.get("WAN22_DEVICE", "0"))
rank = int(os.environ.get("WAN22_RANK", "0"))


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

pipeline = Wan2p2Pipeline.from_pretrained(
    model_path=model_path,
    mode=mode,
    device=device,
    rank=rank,
    t5_cpu=_env_bool("WAN22_T5_CPU", True),
    convert_model_dtype=_env_bool("WAN22_CONVERT_MODEL_DTYPE", True),
    t5_fsdp=_env_bool("WAN22_T5_FSDP", False),
    dit_fsdp=_env_bool("WAN22_DIT_FSDP", False),
    ulysses_size=int(os.environ.get("WAN22_ULYSSES_SIZE", "1")),
)

image_path = os.environ.get("WAN22_IMAGE_PATH")
images = Image.open(image_path).convert("RGB") if image_path else None

output_video = pipeline(
    prompt=os.environ.get(
        "WAN22_PROMPT",
        (
            "Summer beach vacation style, a white cat wearing sunglasses "
            "sits on a surfboard..."
        ),
    ),
    images=images,
    size=os.environ.get("WAN22_SIZE", "1280*704"),
    frame_num=int(os.environ.get("WAN22_FRAME_NUM", "121")),
    sample_steps=int(os.environ.get("WAN22_SAMPLE_STEPS", "50")),
    base_seed=int(os.environ["WAN22_SEED"]) if os.environ.get("WAN22_SEED") else None,
    offload_model=_env_bool("WAN22_OFFLOAD_MODEL", True),
)

save_file_path = os.environ.get("WAN22_OUTPUT_PATH", "./wan_app_demo_output.mp4")
Path(save_file_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
save_video(
    tensor=output_video[None],
    save_file=save_file_path,
    fps=WAN_CONFIGS[pipeline.mode].sample_fps,
    nrow=1,
    normalize=True,
    value_range=(-1, 1),
)
