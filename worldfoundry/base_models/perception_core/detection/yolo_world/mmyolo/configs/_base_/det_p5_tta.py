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
#                     /                 |                     \
# (RatioResize,LetterResize) (RatioResize,LetterResize) (RatioResize,LetterResize)
#        /      \                    /      \                    /        \
#  RandomFlip RandomFlip      RandomFlip RandomFlip        RandomFlip RandomFlip
#      |          |                |         |                  |         |
#  LoadAnn    LoadAnn           LoadAnn    LoadAnn           LoadAnn    LoadAnn
#      |          |                |         |                  |         |
#  PackDetIn  PackDetIn         PackDetIn  PackDetIn        PackDetIn  PackDetIn

_multiscale_resize_transforms = [
    {
        "type": "Compose",
        "transforms": [
            {"type": "YOLOv5KeepRatioResize", "scale": s},
            {"type": "LetterResize", "scale": s, "allow_scale_up": False, "pad_val": {"img": 114}},
        ],
    }
    for s in img_scales
]

tta_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _backend_args},
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
