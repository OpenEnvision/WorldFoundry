_base_ = [
    "mmcls::_base_/datasets/imagenet_bs256_rsb_a12.py",
    "mmcls::_base_/schedules/imagenet_bs2048_rsb.py",
    "mmcls::_base_/default_runtime.py",
]

custom_imports = {"imports": ["mmdet.models", "mmyolo.models"], "allow_failed_imports": False}

model = {
    "type": "ImageClassifier",
    "backbone": {
        "type": "mmyolo.CSPNeXt",
        "arch": "P5",
        "out_indices": (4,),
        "expand_ratio": 0.5,
        "deepen_factor": 0.33,
        "widen_factor": 0.5,
        "channel_attention": True,
        "norm_cfg": {"type": "BN"},
        "act_cfg": {"type": "mmyolo.SiLU"},
    },
    "neck": {"type": "GlobalAveragePooling"},
    "head": {
        "type": "LinearClsHead",
        "num_classes": 1000,
        "in_channels": 512,
        "loss": {"type": "LabelSmoothLoss", "label_smooth_val": 0.1, "mode": "original", "loss_weight": 1.0},
        "topk": (1, 5),
    },
    "train_cfg": {
        "augments": [
            {"type": "Mixup", "alpha": 0.2, "num_classes": 1000},
            {"type": "CutMix", "alpha": 1.0, "num_classes": 1000},
        ]
    },
}

# dataset settings
train_dataloader = {"sampler": {"type": "RepeatAugSampler", "shuffle": True}}

# schedule settings
optim_wrapper = {
    "optimizer": {"weight_decay": 0.01},
    "paramwise_cfg": {"bias_decay_mult": 0.0, "norm_decay_mult": 0.0},
}

param_scheduler = [
    # warm up learning rate scheduler
    {
        "type": "LinearLR",
        "start_factor": 0.0001,
        "by_epoch": True,
        "begin": 0,
        "end": 5,
        # update by iter
        "convert_to_iter_based": True,
    },
    # main learning rate scheduler
    {"type": "CosineAnnealingLR", "T_max": 595, "eta_min": 1.0e-6, "by_epoch": True, "begin": 5, "end": 600},
]

train_cfg = {"by_epoch": True, "max_epochs": 600}
