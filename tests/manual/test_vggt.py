import sys
import os
sys.path.append("..")

from worldfoundry.pipelines.vggt.pipeline_vggt import VGGTPipeline


DATA_PATH = os.environ.get("VGGT_DATA_PATH", "./worldfoundry/data/test_cases/test_image_case1/ref_image.png")
MODEL_PATH = os.environ.get("VGGT_MODEL_PATH", "facebook/VGGT-1B")
OUTPUT_DIR = os.environ.get("VGGT_OUTPUT_DIR", "./vggt_output")
TASK_TYPE = os.environ.get("VGGT_TASK_TYPE", "vggt_official_scene_export")

# Interactions follow unified 3D schema, e.g.:
# ["forward", "left", "camera_zoom_in"]
INTERACTIONS = ["camera_zoom_in","left"]

# camera_view: [dx, dy, dz, theta_x, theta_z]
CAMERA_VIEW = None

POINT_CONF_THRESHOLD = float(os.environ.get("VGGT_POINT_CONF_THRESHOLD", "0.2"))
RESOLUTION = int(os.environ.get("VGGT_RESOLUTION", "518"))
PREPROCESS_MODE = os.environ.get("VGGT_PREPROCESS_MODE", "crop")
IMAGE_WIDTH = int(os.environ.get("VGGT_IMAGE_WIDTH", "704"))
IMAGE_HEIGHT = int(os.environ.get("VGGT_IMAGE_HEIGHT", "480"))
FPS = int(os.environ.get("VGGT_FPS", "12"))
OFFICIAL_CONF_THRES = float(os.environ.get("VGGT_CONF_THRES", "3.0"))
OFFICIAL_FRAME_FILTER = os.environ.get("VGGT_FRAME_FILTER", "All")
OFFICIAL_SHOW_CAM = os.environ.get("VGGT_SHOW_CAM", "1") != "0"
OFFICIAL_PREDICTION_MODE = os.environ.get("VGGT_PREDICTION_MODE", "Pointmap Regression")
DEFAULT_OUTPUT_NAME = "vggt_scene.glb" if TASK_TYPE in {
    "vggt_official_scene_export",
    "vggt_official_glb",
    "official",
} else "vggt_3dgs_demo.mp4"


pipeline = VGGTPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

output = pipeline(
    image_path=DATA_PATH,
    interactions=INTERACTIONS,
    task_type=TASK_TYPE,
    output_dir=OUTPUT_DIR,
    point_conf_threshold=POINT_CONF_THRESHOLD,
    resolution=RESOLUTION,
    preprocess_mode=PREPROCESS_MODE,
    camera_view=CAMERA_VIEW,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    output_name=os.environ.get("VGGT_OUTPUT_NAME", DEFAULT_OUTPUT_NAME),
    fps=FPS,
    conf_thres=OFFICIAL_CONF_THRES,
    frame_filter=OFFICIAL_FRAME_FILTER,
    show_cam=OFFICIAL_SHOW_CAM,
    prediction_mode=OFFICIAL_PREDICTION_MODE,
)

if isinstance(output, dict):
    print(f"VGGT official scene exported to: {output['glb_path']}")
    print(f"VGGT official predictions saved to: {output['prediction_path']}")
else:
    print(f"Rendered VGGT output saved to: {output}")
