# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

import math


def adjust_learning_rate(
    optimizer,
    base_lr,
    p,
    itr,
    max_itr,
    restart=1,
    warm_up_steps=1000,
    is_cosine_decay=False,
    min_lr=1e-5,
    encoder_lr_ratio=1.0,
    freeze_params=[],  # noqa: B006 - Keeps the upstream default object and signature.
):
    if restart > 1:
        each_max_itr = int(math.ceil(float(max_itr) / restart))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        itr = itr % each_max_itr
        warm_up_steps /= restart
        max_itr = each_max_itr

    if itr < warm_up_steps:
        now_lr = min_lr + (base_lr - min_lr) * itr / warm_up_steps
    else:
        itr = itr - warm_up_steps
        max_itr = max_itr - warm_up_steps
        if is_cosine_decay:
            now_lr = min_lr + (base_lr - min_lr) * (math.cos(math.pi * itr / (max_itr + 1)) + 1.0) * 0.5
        else:
            now_lr = min_lr + (base_lr - min_lr) * (1 - itr / (max_itr + 1)) ** p

    for param_group in optimizer.param_groups:
        if encoder_lr_ratio != 1.0 and "encoder." in param_group["name"]:
            param_group["lr"] = (now_lr - min_lr) * encoder_lr_ratio + min_lr
        else:
            param_group["lr"] = now_lr

        for freeze_param in freeze_params:
            if freeze_param in param_group["name"]:
                param_group["lr"] = 0
                param_group["weight_decay"] = 0
                break

    return now_lr


def get_trainable_params(
    model,
    base_lr,
    weight_decay,
    use_frozen_bn=False,
    exclusive_wd_dict={},  # noqa: B006 - Keeps the upstream default object and signature.
    no_wd_keys=[],  # noqa: B006 - Keeps the upstream default object and signature.
):
    params = []
    memo = set()
    total_param = 0
    for key, value in model.named_parameters():
        if value in memo:
            continue
        total_param += value.numel()
        if not value.requires_grad:
            continue
        memo.add(value)
        wd = weight_decay
        for exclusive_key in exclusive_wd_dict.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
            if exclusive_key in key:
                wd = exclusive_wd_dict[exclusive_key]
                break
        if len(value.shape) == 1:  # normalization layers
            # bias requires no weight decay
            # if not use frozen BN, apply zero weight decay
            # if use frozen BN, apply weight decay to all frozen BNs in the encoder
            if "bias" in key or not use_frozen_bn or "encoder." not in key:
                wd = 0.0
        else:
            for no_wd_key in no_wd_keys:
                if no_wd_key in key:
                    wd = 0.0
                    break
        params += [{"params": [value], "lr": base_lr, "weight_decay": wd, "name": key}]

    print(f"Total Param: {total_param / 1e6:.2f}M")
    return params


def freeze_params(module):
    for p in module.parameters():
        p.requires_grad = False


def calculate_params(state_dict):
    memo = set()
    total_param = 0
    for _key, value in state_dict.items():
        if value in memo:
            continue
        memo.add(value)
        total_param += value.numel()
    print(f"Total Param: {total_param / 1e6:.2f}M")
