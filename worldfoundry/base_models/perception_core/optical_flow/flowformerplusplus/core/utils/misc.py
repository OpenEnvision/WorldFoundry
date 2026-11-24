import os
import shutil
import time


def process_transformer_cfg(cfg):
    log_dir = ""
    if "critical_params" in cfg:
        critical_params = [cfg[key] for key in cfg.critical_params]
        for name, param in zip(cfg["critical_params"], critical_params, strict=False):
            log_dir += f"{name:s}[{param!s:s}]"

    return log_dir


def process_cfg(cfg):
    log_dir = "logs/" + cfg.name + "/" + cfg.transformer + "/"
    critical_params = [cfg.trainer[key] for key in cfg.critical_params]
    for name, param in zip(cfg["critical_params"], critical_params, strict=False):
        log_dir += f"{name:s}[{param!s:s}]"

    log_dir += process_transformer_cfg(cfg[cfg.transformer])

    now = time.localtime()
    now_time = f"{now.tm_mon:02d}_{now.tm_mday:02d}_{now.tm_hour:02d}_{now.tm_min:02d}"
    log_dir += cfg.suffix + "(" + now_time + ")"
    cfg.log_dir = log_dir
    os.makedirs(log_dir)

    shutil.copytree("configs", f"{log_dir}/configs")
    shutil.copytree("core/FlowFormer", f"{log_dir}/FlowFormer")
