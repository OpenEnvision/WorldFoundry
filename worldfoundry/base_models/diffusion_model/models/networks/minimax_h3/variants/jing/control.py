# Adapted from XGEN-JING; attribution and license: root THIRD-PARTY-NOTICES.
"""Slice-wise camera controls and checkpoint-compatible FiLM layers."""

import torch
from torch import nn
from torch.nn import functional as F


def control_vector(keys):
    """Map simultaneous WASD/IJKL keys to [tx, ty, tz, pitch, yaw, roll]."""
    if not isinstance(keys, (str, list, tuple, set)):
        raise TypeError("control must be a WASD/IJKL string or a list of keys")
    if isinstance(keys, str):
        keys = ([key.strip().lower() for key in keys.split(",")]
                if "," in keys else list(keys.strip().lower()))
    if any(
        not isinstance(key, str)
        or key.lower() not in {"w", "a", "s", "d", "i", "j", "k", "l"}
        for key in keys
    ):
        raise ValueError(
            "control only accepts individual w/a/s/d/i/j/k/l keys; repetition expressions are not supported"
        )
    keys = {key.lower() for key in keys}
    return [
        float(("d" in keys) - ("a" in keys)),
        0.0,
        float(("w" in keys) - ("s" in keys)),
        float(("i" in keys) - ("k" in keys)),
        float(("l" in keys) - ("j" in keys)),
        0.0,
    ]


class ControlEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        t, h, w = self.patch_size
        self.input_proj = nn.Linear(dim_in * t, dim_out)
        # Retained because control state dictionaries contain this projection.
        self.dense_input_proj = nn.Linear(dim_in * t * h * w, dim_out)
        self.mlp1 = nn.Linear(dim_out, dim_out)
        self.mlp2 = nn.Linear(dim_out, dim_out)

    def forward(self, control, height, width):
        t, h, w = self.patch_size
        value = self.input_proj(
            control.reshape(-1, t * control.shape[-1]).to(self.input_proj.weight.dtype)
        )
        value = value.repeat_interleave((height // h) * (width // w), dim=0)
        return value + self.mlp2(F.silu(self.mlp1(value)))


class FiLM(nn.Module):
    def __init__(self, control_dim, hidden_dim):
        super().__init__()
        self.mlp1 = nn.Linear(control_dim, control_dim)
        self.mlp2 = nn.Linear(control_dim, control_dim)
        self.scale = nn.Linear(control_dim, hidden_dim)
        self.shift = nn.Linear(control_dim, hidden_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, hidden, encoded, indices):
        if indices.numel() == 0:
            return hidden
        encoded = encoded.to(self.mlp1.weight.dtype)
        value = encoded + self.mlp2(F.silu(self.mlp1(encoded)))
        selected = hidden.index_select(1, indices)
        value = selected * (1 + self.scale(value).to(hidden.dtype)) + self.shift(
            value
        ).to(hidden.dtype)
        return hidden.index_copy(1, indices, value)


class Control(nn.Module):
    def __init__(self, config, model_config):
        super().__init__()
        if config["dim_in"] != 6:
            raise ValueError("This demo supports six-dimensional camera control")
        self.encoder = ControlEncoder(
            config["dim_in"], config["dim_out"], model_config["patch_size"]
        )
        self.block_injectors = nn.ModuleList(
            [
                FiLM(config["dim_out"], model_config["hidden_size"])
                for _ in range(model_config["num_layers"])
            ]
        )

    def forward(self, segments):
        values, indices = [], []
        for raw, height, width, joint_indices in segments:
            values.append(self.encoder(raw, height, width))
            indices.append(joint_indices)
        if not values:
            return None, None
        return torch.cat(values), torch.cat(indices)
