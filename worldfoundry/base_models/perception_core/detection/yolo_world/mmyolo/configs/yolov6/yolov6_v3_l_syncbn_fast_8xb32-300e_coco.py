_base_ = "./yolov6_v3_m_syncbn_fast_8xb32-300e_coco.py"

# ======================= Possible modified parameters =======================
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1
# The scaling factor that controls the width of the network structure
widen_factor = 1

# ============================== Unmodified in most cases ===================
model = {
    "backbone": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "hidden_ratio": 1.0 / 2,
        "block_cfg": {"type": "ConvWrapper", "norm_cfg": {"type": "BN", "momentum": 0.03, "eps": 0.001}},
        "act_cfg": {"type": "SiLU", "inplace": True},
    },
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
        "hidden_ratio": 1.0 / 2,
        "block_cfg": {"type": "ConvWrapper", "norm_cfg": {"type": "BN", "momentum": 0.03, "eps": 0.001}},
        "block_act_cfg": {"type": "SiLU", "inplace": True},
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
