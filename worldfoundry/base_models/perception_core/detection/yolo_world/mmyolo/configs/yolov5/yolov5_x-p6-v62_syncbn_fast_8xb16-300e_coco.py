_base_ = "./yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco.py"
deepen_factor = 1.33
widen_factor = 1.25

model = {
    "backbone": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
