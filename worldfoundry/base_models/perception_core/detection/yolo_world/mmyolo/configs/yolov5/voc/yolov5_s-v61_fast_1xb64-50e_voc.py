_base_ = "../yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py"

# dataset settings
data_root = "data/VOCdevkit/"
dataset_type = "YOLOv5VOCDataset"

# parameters that often need to be modified
num_classes = 20
img_scale = (512, 512)  # width, height
max_epochs = 50
train_batch_size_per_gpu = 64
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

lr_factor = 0.15135
affine_scale = 0.75544

# only on Val
batch_shapes_cfg = {"img_size": img_scale[0]}

anchors = [[(26, 44), (67, 57), (61, 130)], [(121, 118), (120, 239), (206, 182)], [(376, 161), (234, 324), (428, 322)]]
num_det_layers = 3

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth"

tta_img_scales = [img_scale, (416, 416), (640, 640)]

# Hyperparameter reference from:
# https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.VOC.yaml
model = {
    "bbox_head": {
        "head_module": {"num_classes": num_classes},
        "prior_generator": {"base_sizes": anchors},
        "loss_cls": {"loss_weight": 0.21638 * (num_classes / 80 * 3 / num_det_layers), "class_weight": 0.5},
        "loss_bbox": {"loss_weight": 0.02 * (3 / num_det_layers)},
        "loss_obj": {
            "loss_weight": 0.51728 * ((img_scale[0] / 640) ** 2 * 3 / num_det_layers),
            "class_weight": 0.67198,
        },
        # Different from COCO
        "prior_match_thr": 3.3744,
    },
    "test_cfg": {"nms": {"iou_threshold": 0.6}},
}

albu_train_transforms = _base_.albu_train_transforms
pre_transform = _base_.pre_transform

with_mosiac_pipeline = [
    {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_translate_ratio": 0.04591,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        "border": (-img_scale[0] // 2, -img_scale[1] // 2),
        "border_val": (114, 114, 114),
    },
    {
        "type": "YOLOv5MixUp",
        "prob": 0.04266,
        "pre_transform": [
            *pre_transform,
            {"type": "Mosaic", "img_scale": img_scale, "pad_val": 114.0, "pre_transform": pre_transform},
            {
                "type": "YOLOv5RandomAffine",
                "max_rotate_degree": 0.0,
                "max_translate_ratio": 0.04591,
                "max_shear_degree": 0.0,
                "scaling_ratio_range": (1 - affine_scale, 1 + affine_scale),
                # img_scale is (width, height)
                "border": (-img_scale[0] // 2, -img_scale[1] // 2),
                "border_val": (114, 114, 114),
            },
        ],
    },
]

without_mosaic_pipeline = [
    {
        "type": "YOLOv5RandomAffine",
        "max_rotate_degree": 0.0,
        "max_translate_ratio": 0.04591,
        "max_shear_degree": 0.0,
        "scaling_ratio_range": (1 - affine_scale, 1 + affine_scale),
        "border": (0, 0),
        "border_val": (114, 114, 114),
    },
    {"type": "LetterResize", "scale": img_scale, "allow_scale_up": True, "pad_val": {"img": 114}},
]

# Because the border parameter is inconsistent when
# using mosaic or not, `RandomChoice` is used here.
randchoice_mosaic_pipeline = {
    "type": "RandomChoice",
    "transforms": [with_mosiac_pipeline, without_mosaic_pipeline],
    "prob": [0.85834, 0.14166],
}

train_pipeline = [
    *pre_transform,
    randchoice_mosaic_pipeline,
    {
        "type": "mmdet.Albu",
        "transforms": albu_train_transforms,
        "bbox_params": {
            "type": "BboxParams",
            "format": "pascal_voc",
            "label_fields": ["gt_bboxes_labels", "gt_ignore_flags"],
        },
        "keymap": {"img": "image", "gt_bboxes": "bboxes"},
    },
    {"type": "YOLOv5HSVRandomAug", "hue_delta": 0.01041, "saturation_delta": 0.54703, "value_delta": 0.27739},
    {"type": "mmdet.RandomFlip", "prob": 0.5},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    },
]

train_dataloader = {
    "_delete_": True,
    "batch_size": train_batch_size_per_gpu,
    "num_workers": train_num_workers,
    "persistent_workers": persistent_workers,
    "pin_memory": True,
    "sampler": {"type": "DefaultSampler", "shuffle": True},
    "dataset": {
        "type": "ConcatDataset",
        "datasets": [
            {
                "type": dataset_type,
                "data_root": data_root,
                "ann_file": "VOC2007/ImageSets/Main/trainval.txt",
                "data_prefix": {"sub_data_root": "VOC2007/"},
                "filter_cfg": {"filter_empty_gt": False, "min_size": 32},
                "pipeline": train_pipeline,
            },
            {
                "type": dataset_type,
                "data_root": data_root,
                "ann_file": "VOC2012/ImageSets/Main/trainval.txt",
                "data_prefix": {"sub_data_root": "VOC2012/"},
                "filter_cfg": {"filter_empty_gt": False, "min_size": 32},
                "pipeline": train_pipeline,
            },
        ],
        # Use ignore_keys to avoid judging metainfo is
        # not equal in `ConcatDataset`.
        "ignore_keys": "dataset_type",
    },
    "collate_fn": {"type": "yolov5_collate"},
}

test_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {"type": "YOLOv5KeepRatioResize", "scale": img_scale},
    {"type": "LetterResize", "scale": img_scale, "allow_scale_up": False, "pad_val": {"img": 114}},
    {"type": "LoadAnnotations", "with_bbox": True, "_scope_": "mmdet"},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor", "pad_param"),
    },
]

val_dataloader = {
    "batch_size": val_batch_size_per_gpu,
    "num_workers": val_num_workers,
    "persistent_workers": persistent_workers,
    "pin_memory": True,
    "drop_last": False,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        "data_root": data_root,
        "ann_file": "VOC2007/ImageSets/Main/test.txt",
        "data_prefix": {"sub_data_root": "VOC2007/"},
        "test_mode": True,
        "pipeline": test_pipeline,
        "batch_shapes_cfg": batch_shapes_cfg,
    },
}

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = {
    "optimizer": {
        "lr": 0.00334,
        "momentum": 0.74832,
        "weight_decay": 0.00025,
        "batch_size_per_gpu": train_batch_size_per_gpu,
    }
}

default_hooks = {
    "param_scheduler": {
        "lr_factor": lr_factor,
        "max_epochs": max_epochs,
        "warmup_epochs": 3.3835,
        "warmup_momentum": 0.59462,
        "warmup_bias_lr": 0.18657,
    }
}

custom_hooks = [
    {
        "type": "EMAHook",
        "ema_type": "ExpMomentumEMA",
        "momentum": 0.0001,
        "update_buffers": True,
        # To load COCO pretrained model, need to set `strict_load=False`
        "strict_load": False,
        "priority": 49,
    }
]

# TODO: Support using coco metric in voc dataset
val_evaluator = {"_delete_": True, "type": "mmdet.VOCMetric", "metric": "mAP", "eval_mode": "area"}

test_evaluator = val_evaluator

train_cfg = {"max_epochs": max_epochs}

# Config for Test Time Augmentation. (TTA)
_multiscale_resize_transforms = [
    {
        "type": "Compose",
        "transforms": [
            {"type": "YOLOv5KeepRatioResize", "scale": s},
            {"type": "LetterResize", "scale": s, "allow_scale_up": False, "pad_val": {"img": 114}},
        ],
    }
    for s in tta_img_scales
]

tta_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {
        "type": "TestTimeAug",
        "transforms": [
            _multiscale_resize_transforms,
            [{"type": "mmdet.RandomFlip", "prob": 1.0}, {"type": "mmdet.RandomFlip", "prob": 0.0}],
            [{"type": "mmdet.LoadAnnotations", "with_bbox": True}],
            [
                {
                    "type": "mmdet.PackDetInputs",
                    "meta_keys": (
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "pad_param",
                        "flip",
                        "flip_direction",
                    ),
                }
            ],
        ],
    },
]
