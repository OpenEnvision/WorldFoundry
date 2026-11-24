import os
from copy import deepcopy

import numpy as np
import torch
from rich import print

from worldfoundry.core.configuration.factory import (
    get_obj_from_str as get_obj_from_str,
)
from worldfoundry.core.configuration.factory import (
    instantiate_from_config,
)


def print0(*args, **kwargs):
    print(*args, **kwargs)  # python -m rich.color


def tensor_to_uint8(tensor):
    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    tensor = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return tensor


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    # Released configs use upstream module names; resolve them only at the load site.
    targets = {
        "lvm.LlamaLVM": "worldfoundry.synthesis.visual_generation.mineworld.lvm.LlamaLVM",
        "lvm.LlamaForCausalLM": "worldfoundry.synthesis.visual_generation.mineworld.lvm.LlamaForCausalLM",
        "vae.VAE": "worldfoundry.synthesis.visual_generation.mineworld.vae.VAE",
    }
    config = deepcopy(config)
    params = config.get("params", {})
    for node, key in (
        (config, "target"),
        (params, "model_class"),
        (params.get("tokenizer_config") or {}, "target"),
    ):
        target = node.get(key)
        if target in targets:
            node[key] = targets[target]
    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) != 0:
            raise ValueError(f"Missing keys: {missing}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_valid_dirs(dir1: str, dir2: None | str = None, dir3: None | str = None) -> None | str:
    if (dir1 is not None) and os.path.isdir(dir1):
        return dir1
    elif (dir2 is not None) and os.path.isdir(dir2):
        return dir2
    elif (dir3 is not None) and os.path.isdir(dir3):
        return dir3
    else:
        return None


def get_valid_paths(path1: str, path2: None | str = None, path3: None | str = None) -> None | str:
    if (path1 is not None) and os.path.isfile(path1):
        return path1
    elif (path2 is not None) and os.path.isfile(path2):
        return path2
    elif (path3 is not None) and os.path.isfile(path3):
        return path3
    else:
        return None


def load_model(config, ckpt, gpu, eval_mode):
    if str(ckpt).endswith(".bin"):
        weight = torch.load(ckpt)
    elif ckpt:
        weight = torch.load(ckpt, map_location="cpu")["state_dict"]
    model = load_model_from_config(config.model, weight, gpu=gpu, eval_mode=eval_mode)["model"]
    model.load_state_dict(weight, strict=False)
    model.to(torch.float16)

    return model
