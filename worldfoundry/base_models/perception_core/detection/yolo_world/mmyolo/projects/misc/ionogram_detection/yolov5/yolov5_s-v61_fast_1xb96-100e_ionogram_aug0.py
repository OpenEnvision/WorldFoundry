_base_ = "./yolov5_s-v61_fast_1xb96-100e_ionogram.py"

# ======================= Modified parameters =====================
# -----train val related-----
train_pipeline = [
    {"type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
    {"type": "YOLOv5KeepRatioResize", "scale": (640, 640)},
    {"type": "LetterResize", "scale": (640, 640), "allow_scale_up": False, "pad_val": {"img": 114}},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor", "pad_param"),
    },
]

# ===================== Unmodified in most cases ==================
train_dataloader = {"dataset": {"dataset": {"pipeline": train_pipeline}}}
