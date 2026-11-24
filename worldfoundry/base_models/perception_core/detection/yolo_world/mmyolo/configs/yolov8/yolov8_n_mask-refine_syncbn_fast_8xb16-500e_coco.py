_base_ = "./yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py"

# This config will refine bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

deepen_factor = 0.33
widen_factor = 0.25

model = {
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "neck": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
