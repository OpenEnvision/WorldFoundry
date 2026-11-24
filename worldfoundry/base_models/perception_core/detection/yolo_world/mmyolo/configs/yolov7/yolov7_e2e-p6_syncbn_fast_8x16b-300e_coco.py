_base_ = "./yolov7_w-p6_syncbn_fast_8x16b-300e_coco.py"

model = {
    "backbone": {"arch": "E2E"},
    "neck": {
        "use_maxpool_in_downsample": True,
        "use_in_channels_in_downsample": True,
        "block_cfg": {
            "type": "EELANBlock",
            "num_elan_block": 2,
            "middle_ratio": 0.4,
            "block_ratio": 0.2,
            "num_blocks": 6,
            "num_convs_in_block": 1,
        },
        "in_channels": [320, 640, 960, 1280],
        "out_channels": [160, 320, 480, 640],
    },
    "bbox_head": {"head_module": {"in_channels": [160, 320, 480, 640], "main_out_channels": [320, 640, 960, 1280]}},
}
