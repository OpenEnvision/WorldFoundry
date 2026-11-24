_base_ = "./yolov8_s_syncbn_fast_8xb16-500e_coco.py"

deepen_factor = 0.33
widen_factor = 0.25

model = {
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "neck": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
