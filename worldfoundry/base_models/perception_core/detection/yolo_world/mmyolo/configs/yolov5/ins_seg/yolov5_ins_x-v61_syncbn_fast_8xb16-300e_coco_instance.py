_base_ = "./yolov5_ins_l-v61_syncbn_fast_8xb16-300e_coco_instance.py"

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
