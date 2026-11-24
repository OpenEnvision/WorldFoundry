_base_ = ["./yolox-pose_m_8xb32-300e-rtmdet-hyp_coco.py"]

load_from = "https://download.openmmlab.com/mmyolo/v0/yolox/yolox_l_fast_8xb8-300e_coco/yolox_l_fast_8xb8-300e_coco_20230213_160715-c731eb1c.pth"

# ========================modified parameters======================
deepen_factor = 1.0
widen_factor = 1.0

# =======================Unmodified in most cases==================
# model settings
model = {
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "neck": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
