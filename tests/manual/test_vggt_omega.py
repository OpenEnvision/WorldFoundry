import os
import sys
from pathlib import Path

sys.path.append("..")

from worldfoundry.pipelines.vggt_omega.pipeline_vggt_omega import VGGTOmegaPipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = os.environ.get("WORLDFOUNDRY_CKPT_DIR", str(REPO_ROOT / "cache" / "checkpoints"))
DATA_PATH = os.environ.get(
    "VGGT_OMEGA_DATA_PATH",
    "./worldfoundry/data/test_cases/test_image_seq_case1",
)
MODEL_PATH = os.environ.get("VGGT_OMEGA_MODEL_PATH", str(Path(CKPT_ROOT) / "VGGT-Omega"))
OUTPUT_DIR = os.environ.get("VGGT_OMEGA_OUTPUT_DIR", "./vggt_omega_output")
TASK_TYPE = os.environ.get("VGGT_OMEGA_TASK_TYPE", "vggt_omega_official_scene_export")

IMAGE_RESOLUTION = int(os.environ.get("VGGT_OMEGA_IMAGE_RESOLUTION", "512"))
PATCH_SIZE = int(os.environ.get("VGGT_OMEGA_PATCH_SIZE", "16"))
PREPROCESS_MODE = os.environ.get("VGGT_OMEGA_PREPROCESS_MODE", "balanced")
CONF_THRES = float(os.environ.get("VGGT_OMEGA_CONF_THRES", "20.0"))
SHOW_CAM = os.environ.get("VGGT_OMEGA_SHOW_CAM", "1") != "0"
MAX_POINTS_K = int(os.environ.get("VGGT_OMEGA_MAX_POINTS_K", "1000"))
OUTPUT_NAME = os.environ.get("VGGT_OMEGA_OUTPUT_NAME", "vggt_omega_scene.glb")


pipeline = VGGTOmegaPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

output = pipeline(
    image_path=DATA_PATH,
    task_type=TASK_TYPE,
    output_dir=OUTPUT_DIR,
    image_resolution=IMAGE_RESOLUTION,
    patch_size=PATCH_SIZE,
    preprocess_mode=PREPROCESS_MODE,
    conf_thres=CONF_THRES,
    show_cam=SHOW_CAM,
    max_points_k=MAX_POINTS_K,
    output_name=OUTPUT_NAME,
)

if isinstance(output, dict):
    print(f"VGGT-Omega official scene exported to: {output['glb_path']}")
    print(f"VGGT-Omega official predictions saved to: {output['prediction_path']}")
else:
    print(f"VGGT-Omega output saved to: {output}")
