# TODO: Need to solve the problem of multiple backend_args parameters
# _backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

_backend_args = None

tta_model = {
    "type": "mmdet.DetTTAModel",
    "tta_cfg": {"nms": {"type": "nms", "iou_threshold": 0.65}, "max_per_img": 300},
}

img_scales = [(640, 640), (320, 320), (960, 960)]

#                                LoadImageFromFile
#              /                        |                          \
#          Resize                     Resize                       Resize
#        /      \                    /      \                    /        \
#  RandomFlip RandomFlip      RandomFlip RandomFlip        RandomFlip RandomFlip
#      |          |                |         |                  |         |
#  LoadAnn    LoadAnn           LoadAnn    LoadAnn           LoadAnn    LoadAnn
#      |          |                |         |                  |         |
#  PackDetIn  PackDetIn         PackDetIn  PackDetIn        PackDetIn  PackDetIn

tta_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _backend_args},
    {
        "type": "TestTimeAug",
        "transforms": [
            [{"type": "mmdet.Resize", "scale": s, "keep_ratio": True} for s in img_scales],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                {"type": "mmdet.RandomFlip", "prob": 1.0},
                {"type": "mmdet.RandomFlip", "prob": 0.0},
            ],
            [
                {"type": "mmdet.Pad", "pad_to_square": True, "pad_val": {"img": (114.0, 114.0, 114.0)}},
            ],
            [
                {
                    "type": "mmdet.PackDetInputs",
                    "meta_keys": (
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                }
            ],
        ],
    },
]
