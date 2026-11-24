_base_ = "./yolov7_w-p6_syncbn_fast_8x16b-300e_coco.py"

model = {
    "backbone": {"arch": "D"},
    "neck": {
        "use_maxpool_in_downsample": True,
        "use_in_channels_in_downsample": True,
        "block_cfg": {
            "type": "ELANBlock",
            "middle_ratio": 0.4,
            "block_ratio": 0.2,
            "num_blocks": 6,
            "num_convs_in_block": 1,
        },
        "in_channels": [384, 768, 1152, 1536],
        "out_channels": [192, 384, 576, 768],
    },
    "bbox_head": {
        "head_module": {
            "in_channels": [192, 384, 576, 768],
            "main_out_channels": [384, 768, 1152, 1536],
            "aux_out_channels": [384, 768, 1152, 1536],
        }
    },
}
