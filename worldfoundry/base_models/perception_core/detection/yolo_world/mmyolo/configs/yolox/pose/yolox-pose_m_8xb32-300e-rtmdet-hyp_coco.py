_base_ = ["./yolox-pose_s_8xb32-300e-rtmdet-hyp_coco.py"]

load_from = "https://download.openmmlab.com/mmyolo/v0/yolox/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco_20230210_144328-e657e182.pth"

# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75

# =======================Unmodified in most cases==================
# model settings
model = {
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "neck": {"deepen_factor": deepen_factor, "widen_factor": widen_factor},
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
