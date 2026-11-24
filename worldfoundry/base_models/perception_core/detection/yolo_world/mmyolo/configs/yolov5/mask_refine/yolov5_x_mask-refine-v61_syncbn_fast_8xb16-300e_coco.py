_base_ = "./yolov5_l_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py"

# This config use refining bbox and `YOLOv5CopyPaste`.
# Refining bbox means refining bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

# ========================modified parameters======================
deepen_factor = 1.33
widen_factor = 1.25

# ===============================Unmodified in most cases====================
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
