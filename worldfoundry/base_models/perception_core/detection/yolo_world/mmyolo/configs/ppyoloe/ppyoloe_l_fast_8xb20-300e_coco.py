_base_ = "./ppyoloe_s_fast_8xb32-300e_coco.py"

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
checkpoint = (
    "https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/cspresnet_l_imagenet1k_pretrained-c0010e6c.pth"
)

deepen_factor = 1.0
widen_factor = 1.0

train_batch_size_per_gpu = 20

model = {
    "backbone": {"deepen_factor": deepen_factor, "widen_factor": widen_factor, "init_cfg": {"checkpoint": checkpoint}},
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}

train_dataloader = {"batch_size": train_batch_size_per_gpu}
