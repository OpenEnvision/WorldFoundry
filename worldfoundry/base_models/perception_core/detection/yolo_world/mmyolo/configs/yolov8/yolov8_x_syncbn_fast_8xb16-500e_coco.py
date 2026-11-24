_base_ = "./yolov8_l_syncbn_fast_8xb16-500e_coco.py"

deepen_factor = 1.00
widen_factor = 1.25

model = {
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "neck": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
