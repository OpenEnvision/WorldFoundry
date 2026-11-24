_base_ = "../rtmdet_s_syncbn_fast_8xb32-300e_coco.py"

teacher_ckpt = "https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth"

norm_cfg = {"type": "BN", "affine": False, "track_running_stats": False}

model = {
    "_delete_": True,
    "_scope_": "mmrazor",
    "type": "FpnTeacherDistill",
    "architecture": {"cfg_path": "mmyolo::rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py"},
    "teacher": {"cfg_path": "mmyolo::rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco.py"},
    "teacher_ckpt": teacher_ckpt,
    "distiller": {
        "type": "ConfigurableDistiller",
        # `recorders` are used to record various intermediate results during
        # the model forward.
        "student_recorders": {
            "fpn0": {"type": "ModuleOutputs", "source": "neck.out_layers.0.conv"},
            "fpn1": {"type": "ModuleOutputs", "source": "neck.out_layers.1.conv"},
            "fpn2": {"type": "ModuleOutputs", "source": "neck.out_layers.2.conv"},
        },
        "teacher_recorders": {
            "fpn0": {"type": "ModuleOutputs", "source": "neck.out_layers.0.conv"},
            "fpn1": {"type": "ModuleOutputs", "source": "neck.out_layers.1.conv"},
            "fpn2": {"type": "ModuleOutputs", "source": "neck.out_layers.2.conv"},
        },
        # `connectors` are adaptive layers which usually map teacher's and
        # students features to the same dimension.
        "connectors": {
            "fpn0_s": {
                "type": "ConvModuleConnector",
                "in_channel": 128,
                "out_channel": 192,
                "bias": False,
                "norm_cfg": norm_cfg,
                "act_cfg": None,
            },
            "fpn0_t": {"type": "NormConnector", "in_channels": 192, "norm_cfg": norm_cfg},
            "fpn1_s": {
                "type": "ConvModuleConnector",
                "in_channel": 128,
                "out_channel": 192,
                "bias": False,
                "norm_cfg": norm_cfg,
                "act_cfg": None,
            },
            "fpn1_t": {"type": "NormConnector", "in_channels": 192, "norm_cfg": norm_cfg},
            "fpn2_s": {
                "type": "ConvModuleConnector",
                "in_channel": 128,
                "out_channel": 192,
                "bias": False,
                "norm_cfg": norm_cfg,
                "act_cfg": None,
            },
            "fpn2_t": {"type": "NormConnector", "in_channels": 192, "norm_cfg": norm_cfg},
        },
        "distill_losses": {
            "loss_fpn0": {"type": "ChannelWiseDivergence", "loss_weight": 1},
            "loss_fpn1": {"type": "ChannelWiseDivergence", "loss_weight": 1},
            "loss_fpn2": {"type": "ChannelWiseDivergence", "loss_weight": 1},
        },
        # `loss_forward_mappings` are mappings between distill loss forward
        # arguments and records.
        "loss_forward_mappings": {
            "loss_fpn0": {
                "preds_S": {"from_student": True, "recorder": "fpn0", "connector": "fpn0_s"},
                "preds_T": {"from_student": False, "recorder": "fpn0", "connector": "fpn0_t"},
            },
            "loss_fpn1": {
                "preds_S": {"from_student": True, "recorder": "fpn1", "connector": "fpn1_s"},
                "preds_T": {"from_student": False, "recorder": "fpn1", "connector": "fpn1_t"},
            },
            "loss_fpn2": {
                "preds_S": {"from_student": True, "recorder": "fpn2", "connector": "fpn2_s"},
                "preds_T": {"from_student": False, "recorder": "fpn2", "connector": "fpn2_t"},
            },
        },
    },
}

find_unused_parameters = True

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
        "switch_pipeline": _base_.train_pipeline_stage2,
    },
    # stop distillation after the 280th epoch
    {"type": "mmrazor.StopDistillHook", "stop_epoch": 280},
]
