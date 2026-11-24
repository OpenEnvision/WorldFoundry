_base_ = "./yolov5_s-v61_fast_1xb96-100e_ionogram.py"

# ======================= Modified parameters =====================
# -----train val related-----
base_lr = _base_.base_lr * 4
max_epochs = 200
load_from = None
logger_interval = 50

train_cfg = {
    "max_epochs": max_epochs,
}

# ===================== Unmodified in most cases ==================
optim_wrapper = {"optimizer": {"lr": base_lr}}

default_hooks = {
    "param_scheduler": {"max_epochs": max_epochs},
    "logger": {"type": "LoggerHook", "interval": logger_interval},
}
