_base_ = "./yolov8_m_syncbn_fast_8xb16-500e_coco.py"

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15

# =======================Unmodified in most cases==================
pre_transform = _base_.pre_transform
mosaic_affine_transform = _base_.mosaic_affine_transform
last_transform = _base_.last_transform

model = {
    "backbone": {
        "last_stage_out_channels": last_stage_out_channels,
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "in_channels": [256, 512, last_stage_out_channels],
        "out_channels": [256, 512, last_stage_out_channels],
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor, "in_channels": [256, 512, last_stage_out_channels]}},
}

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    {"type": "YOLOv5MixUp", "prob": mixup_prob, "pre_transform": [*pre_transform, *mosaic_affine_transform]},
    *last_transform,
]

train_dataloader = {"dataset": {"pipeline": train_pipeline}}
