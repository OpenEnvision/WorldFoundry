_base_ = "./yolov5u_s_syncbn_fast_8xb16-300e_coco.py"

# ========================modified parameters======================
# TODO: Update the training hyperparameters
deepen_factor = 0.67
widen_factor = 0.75

# =======================Unmodified in most cases==================
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
