_base_ = "../../../../../../detection/yolo_world/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py"
custom_imports = {"imports": ["yolo_world"], "allow_failed_imports": False}

# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = (
    "pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth"
)
persistent_workers = False
text_model_name = "openai/clip-vit-base-patch32/"
img_scale = (800, 800)

# model settings
model = {
    "type": "YOLOWorldImageDetector",
    "mm_neck": True,
    "num_train_classes": num_training_classes,
    "num_test_classes": num_classes,
    "vision_model": text_model_name,
    "prompt_dim": text_channels,
    "data_preprocessor": {"type": "YOLOv5DetDataPreprocessor"},
    "backbone": {
        "_delete_": True,
        "type": "MultiModalYOLOBackbone",
        "image_model": {{_base_.model.backbone}},
        "frozen_stages": 4,
        "text_model": {
            "type": "HuggingCLIPLanguageBackbone",
            "model_name": text_model_name,
            "frozen_modules": ["all"],
        },
    },
    "neck": {
        "type": "YOLOWorldPAFPN",
        "freeze_all": True,
        "guide_channels": text_channels,
        "embed_channels": neck_embed_channels,
        "num_heads": neck_num_heads,
        "block_cfg": {"type": "MaxSigmoidCSPLayerWithTwoConv"},
    },
    "bbox_head": {
        "type": "YOLOWorldHead",
        "head_module": {
            "type": "YOLOWorldHeadModule",
            "freeze_all": True,
            "use_bn_head": True,
            "embed_dims": text_channels,
            "num_classes": num_training_classes,
        },
    },
    "train_cfg": {"assigner": {"num_classes": num_training_classes}},
}

# dataset settings
coco_train_dataset = {
    "type": "YOLOv5CocoDataset",
    "data_root": "data/coco",
    "ann_file": "annotations/instances_train2017.json",
    "data_prefix": {"img": "train2017/"},
    "filter_cfg": {"filter_empty_gt": False, "min_size": 32},
    "pipeline": _base_.train_pipeline,
}

train_dataloader = {
    "persistent_workers": persistent_workers,
    "batch_size": train_batch_size_per_gpu,
    "collate_fn": {"type": "yolow_collate"},
    "dataset": coco_train_dataset,
}
test_pipeline = [
    *_base_.test_pipeline[:-1],
    {"type": "LoadText"},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": (
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
            "texts",
        ),
    },
]
coco_val_dataset = {
    "_delete_": True,
    "type": "MultiModalDataset",
    "dataset": {
        "type": "YOLOv5CocoDataset",
        "data_root": "data/coco",
        "ann_file": "annotations/instances_val2017.json",
        "data_prefix": {"img": "val2017/"},
        "filter_cfg": {"filter_empty_gt": False, "min_size": 32},
    },
    "class_text_path": "data/texts/coco_class_texts.json",
    "pipeline": test_pipeline,
}
val_dataloader = {"dataset": coco_val_dataset}
test_dataloader = val_dataloader
# training settings
default_hooks = {
    "param_scheduler": {"scheduler_type": "linear", "lr_factor": 0.01, "max_epochs": max_epochs},
    "checkpoint": {"max_keep_ckpts": -1, "save_best": None, "interval": save_epoch_intervals},
}
custom_hooks = [
    {
        "type": "EMAHook",
        "ema_type": "ExpMomentumEMA",
        "momentum": 0.0001,
        "update_buffers": True,
        "strict_load": False,
        "priority": 49,
    },
    {
        "type": "mmdet.PipelineSwitchHook",
        "switch_epoch": max_epochs - close_mosaic_epochs,
        "switch_pipeline": _base_.train_pipeline_stage2,
    },
]
train_cfg = {
    "max_epochs": max_epochs,
    "val_interval": 5,
    "dynamic_intervals": [((max_epochs - close_mosaic_epochs), _base_.val_interval_stage2)],
}

optim_wrapper = {
    "optimizer": {
        "_delete_": True,
        "type": "AdamW",
        "lr": base_lr,
        "weight_decay": weight_decay,
        "batch_size_per_gpu": train_batch_size_per_gpu,
    },
    "paramwise_cfg": {
        "custom_keys": {
            "backbone.text_model": {"lr_mult": 0.01},
            "logit_scale": {"weight_decay": 0.0},
            "embeddings": {"weight_decay": 0.0},
        }
    },
    "constructor": "YOLOWv5OptimizerConstructor",
}

# evaluation settings
val_evaluator = {
    "_delete_": True,
    "type": "mmdet.CocoMetric",
    "proposal_nums": (100, 1, 10),
    "ann_file": "data/coco/annotations/instances_val2017.json",
    "metric": "bbox",
}
