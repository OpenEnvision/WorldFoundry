# This file includes code originally from the Metric3D repository:
# https://github.com/YvanYin/Metric3D
# Licensed under the BSD-2 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> three_dimensions -> depth -> metric3d -> model -> configs -> convlarge.0.3_150.py functionality."""

from omegaconf import DictConfig

config = DictConfig(
    {
        "model": {
            "backbone": {
                "type": "convnext_large",
                "pretrained": False,
                "in_22k": True,
                "out_indices": [0, 1, 2, 3],
                "drop_path_rate": 0.4,
                "layer_scale_init_value": 1.0,
                "checkpoint": "data/pretrained_weight_repo/convnext/convnext_large_22k_1k_384.pth",
                "prefix": "backbones.",
                "out_channels": [192, 384, 768, 1536],
            },
            "type": "DensePredModel",
            "decode_head": {
                "type": "HourglassDecoder",
                "in_channels": [192, 384, 768, 1536],
                "decoder_channel": [128, 128, 256, 512],
                "prefix": "decode_heads.",
            },
        },
        "data_basic": {
            "canonical_space": {"img_size": (512, 960), "focal_length": 1000.0},
            "depth_range": (0, 1),
            "depth_normalize": (0.3, 150),
            "crop_size": (544, 1216),
            "clip_depth_range": (0.9, 150),
        },
        "load_from": None,
        "cudnn_benchmark": True,
        "test_metrics": [
            "abs_rel",
            "rmse",
            "silog",
            "delta1",
            "delta2",
            "delta3",
            "rmse_log",
            "log10",
            "sq_rel",
        ],
        "batchsize_per_gpu": 2,
        "thread_per_gpu": 4,
    }
)
