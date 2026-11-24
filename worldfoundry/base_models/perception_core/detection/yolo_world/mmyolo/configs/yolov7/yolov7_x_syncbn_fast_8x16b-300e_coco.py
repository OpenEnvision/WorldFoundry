_base_ = "./yolov7_l_syncbn_fast_8x16b-300e_coco.py"

model = {
    "backbone": {"arch": "X"},
    "neck": {
        "in_channels": [640, 1280, 1280],
        "out_channels": [160, 320, 640],
        "block_cfg": {
            "type": "ELANBlock",
            "middle_ratio": 0.4,
            "block_ratio": 0.4,
            "num_blocks": 3,
            "num_convs_in_block": 2,
        },
        "use_repconv_outs": False,
    },
    "bbox_head": {"head_module": {"in_channels": [320, 640, 1280]}},
}
