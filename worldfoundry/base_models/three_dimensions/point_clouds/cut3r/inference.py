"""CUT3R batch and recurrent inference, without loss or optimizer dependencies."""

import torch

from .utils.device import to_cpu

_METADATA = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}


def _to_device(view, device):
    return {key: value if key in _METADATA else _move(value, device) for key, value in view.items()}


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, (list, tuple)):
        return type(value)(_move(x, device) for x in value)
    return value


@torch.no_grad()
def inference(groups, model, device, verbose=True):
    if not groups:
        raise ValueError("CUT3R requires at least one view")
    if len({tuple(view["img"].shape) for view in groups}) > 1:
        return inference_recurrent(groups, model, device, verbose=verbose)
    groups = [_to_device(view, device) for view in groups]
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        output, state_args = model(groups, ret_state=True)
    return to_cpu(dict(views=output.views, pred=output.ress)), state_args


@torch.no_grad()
def inference_recurrent(groups, model, device, verbose=True):
    if not groups:
        raise ValueError("CUT3R requires at least one view")
    groups = [_to_device(view, device) for view in groups]
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        preds, views, state_args = model.forward_recurrent(groups, device, ret_state=True)
    return to_cpu(dict(views=views, pred=preds)), state_args


@torch.no_grad()
def inference_step(view, state_args, model, device, verbose=True):
    view = _to_device(view, device)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        pred, _ = model.inference_step(view, *state_args)
    return to_cpu(dict(pred=pred))
