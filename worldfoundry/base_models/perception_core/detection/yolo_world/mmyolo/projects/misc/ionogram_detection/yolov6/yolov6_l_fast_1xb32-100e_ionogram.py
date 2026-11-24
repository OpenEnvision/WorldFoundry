_base_ = "./yolov6_m_fast_1xb32-100e_ionogram.py"

# ======================= Modified parameters =======================
# -----model related-----
deepen_factor = 1
widen_factor = 1

# -----train val related-----
load_from = "https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco/yolov6_l_syncbn_fast_8xb32-300e_coco_20221109_183156-91e3c447.pth"

# ====================== Unmodified in most cases ===================
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
