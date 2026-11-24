_base_ = "./yolov5_s-v61_fast_1xb96-100e_ionogram.py"

# ======================= Modified parameters =====================
# -----train val related-----
train_pipeline = [
    {"type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": True},
    {
        "type": "Mosaic",
        "img_scale": (640, 640),
        "pad_val": 114.0,
        "pre_transform": [{"type": "LoadImageFromFile"}, {"type": "LoadAnnotations", "with_bbox": True}],
    },
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (0.5, 1.5),
        "border": (-320, -320),
        "border_val": (114, 114, 114),
    },
    {"type": "mmdet.PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape")},
]

# ===================== Unmodified in most cases ==================
train_dataloader = {"dataset": {"dataset": {"pipeline": train_pipeline}}}
