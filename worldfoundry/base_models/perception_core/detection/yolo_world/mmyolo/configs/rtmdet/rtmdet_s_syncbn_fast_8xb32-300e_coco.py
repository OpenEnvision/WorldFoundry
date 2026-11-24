_base_ = "./rtmdet_l_syncbn_fast_8xb32-300e_coco.py"
checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth"

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.5
img_scale = _base_.img_scale

# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixup
mixup_max_cached_images = 20

# =======================Unmodified in most cases==================
model = {
    "backbone": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        # Since the checkpoint includes CUDA:0 data,
        # it must be forced to set map_location.
        # Once checkpoint is fixed, it can be removed.
        "init_cfg": {"type": "Pretrained", "prefix": "backbone.", "checkpoint": checkpoint, "map_location": "cpu"},
    },
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}

train_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "LoadAnnotations", "with_bbox": True},
    {
        "type": "Mosaic",
        "img_scale": img_scale,
        "use_cached": True,
        "max_cached_images": mosaic_max_cached_images,
        "pad_val": 114.0,
    },
    {
        "type": "mmdet.RandomResize",
        # img_scale is (width, height)
        "scale": (img_scale[0] * 2, img_scale[1] * 2),
        "ratio_range": random_resize_ratio_range,  # note
        "resize_type": "mmdet.Resize",
        "keep_ratio": True,
    },
    {"type": "mmdet.RandomCrop", "crop_size": img_scale},
    {"type": "mmdet.YOLOXHSVRandomAug"},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {"type": "mmdet.Pad", "size": img_scale, "pad_val": {"img": (114, 114, 114)}},
    {"type": "YOLOv5MixUp", "use_cached": True, "max_cached_images": mixup_max_cached_images},
    {"type": "mmdet.PackDetInputs"},
]

train_pipeline_stage2 = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "LoadAnnotations", "with_bbox": True},
    {
        "type": "mmdet.RandomResize",
        "scale": img_scale,
        "ratio_range": random_resize_ratio_range,  # note
        "resize_type": "mmdet.Resize",
        "keep_ratio": True,
    },
    {"type": "mmdet.RandomCrop", "crop_size": img_scale},
    {"type": "mmdet.YOLOXHSVRandomAug"},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {"type": "mmdet.Pad", "size": img_scale, "pad_val": {"img": (114, 114, 114)}},
    {"type": "mmdet.PackDetInputs"},
]

train_dataloader = {"dataset": {"pipeline": train_pipeline}}

custom_hooks = [
    {
        "type": "EMAHook",
        "ema_type": "ExpMomentumEMA",
        "momentum": 0.0002,
        "update_buffers": True,
        "strict_load": False,
        "priority": 49,
    },
    {
        "type": "mmdet.PipelineSwitchHook",
        "switch_epoch": _base_.max_epochs - _base_.num_epochs_stage2,
        "switch_pipeline": train_pipeline_stage2,
    },
]
