_base_ = "./ppyoloe_s_fast_8xb32-300e_coco.py"

max_epochs = 400

model = {"train_cfg": {"initial_epoch": 133}}

default_hooks = {"param_scheduler": {"total_epochs": int(max_epochs * 1.2)}}

train_cfg = {"max_epochs": max_epochs}
