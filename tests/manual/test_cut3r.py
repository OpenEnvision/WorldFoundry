import sys
import os
from pathlib import Path

sys.path.append("..")

from worldfoundry.pipelines.cut3r.pipeline_cut3r import CUT3RPipeline


DATA_PATH = os.environ.get("CUT3R_DATA_PATH", "./worldfoundry/data/test_cases/test_image_case1/ref_image.png")
MODEL_NAME = os.environ.get("CUT3R_MODEL_PATH", "liguang0115/cut3r")

SIZE = int(os.environ.get("CUT3R_SIZE", "512"))
VIS_THRESHOLD = float(os.environ.get("CUT3R_VIS_THRESHOLD", "1.5"))
OUTPUT_DIR = os.environ.get("CUT3R_OUTPUT_DIR", "./cut3r_output")
TASK_TYPE = os.environ.get("CUT3R_TASK_TYPE", "cut3r_official_export")

# Interaction sequence for camera control in the second stage.
# Keep None to use a default orbit; or set to a list like:
# ["forward", "camera_l"].
INTERACTIONS = ["forward", "camera_l"]

# Two-stage camera config for 3DGS rendering.
CAMERA_RADIUS = float(os.environ.get("CUT3R_CAMERA_RADIUS", "4.0"))
CAMERA_YAW = float(os.environ.get("CUT3R_CAMERA_YAW", "0.0"))
CAMERA_PITCH = float(os.environ.get("CUT3R_CAMERA_PITCH", "0.0"))

IMAGE_WIDTH = int(os.environ.get("CUT3R_IMAGE_WIDTH", "512"))
IMAGE_HEIGHT = int(os.environ.get("CUT3R_IMAGE_HEIGHT", "288"))
USE_POSE = os.environ.get("CUT3R_USE_POSE", "1") != "0"


pipeline = CUT3RPipeline.from_pretrained(
    representation_path=MODEL_NAME,
    size=SIZE,
)

output = pipeline(
    image_path=DATA_PATH,
    interactions=INTERACTIONS,
    task_type=TASK_TYPE,
    size=SIZE,
    vis_threshold=VIS_THRESHOLD,
    output_dir=OUTPUT_DIR,
    camera_radius=CAMERA_RADIUS,
    camera_yaw=CAMERA_YAW,
    camera_pitch=CAMERA_PITCH,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    use_pose=USE_POSE,
)

if isinstance(output, dict):
    print(f"CUT3R official outputs saved to: {output['output_dir']}")
    print(f"CUT3R official summary saved to: {output['summary_path']}")
else:
    print(f"Rendered CUT3R output saved to: {output}")
