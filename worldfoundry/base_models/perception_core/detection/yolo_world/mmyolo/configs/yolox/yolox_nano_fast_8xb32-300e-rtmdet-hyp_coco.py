_base_ = "./yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco.py"

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.25
use_depthwise = True

# =======================Unmodified in most cases==================
# model settings
model = {
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor, "use_depthwise": use_depthwise},
    "neck": {"deepen_factor": deepen_factor, "widen_factor": widen_factor, "use_depthwise": use_depthwise},
    "bbox_head": {"head_module": {"widen_factor": widen_factor, "use_depthwise": use_depthwise}},
}
