_base_ = "../../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py"

test_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": _base_.backend_args},
    {
        "type": "LetterResize",
        "scale": _base_.img_scale,
        "allow_scale_up": False,
        "use_mini_pad": False,
    },
    {"type": "LoadAnnotations", "with_bbox": True, "_scope_": "mmdet"},
    {
        "type": "mmdet.PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor", "pad_param"),
    },
]

test_dataloader = {"dataset": {"pipeline": test_pipeline, "batch_shapes_cfg": None}}
