"""Exponential moving-average module shared by model implementations.

Shadow weights for eval without a second optimizer. Updates stay on
the same device as the source; FSDP2 uses
``DTensorFastEmaModelUpdater`` in ``device_mesh_collectives`` instead.
"""

import torch
from torch import nn

from worldfoundry.core.distributed.collectives.device_mesh import get_local_tensor_if_dtensor


# ──────────────────────────────────────────────────────────────────────────
# Shadow buffers — dots stripped from names; decay stays FP32 (BF16 would freeze)
# ──────────────────────────────────────────────────────────────────────────


class LitEma(nn.Module):
    """Lit ema implementation."""

    def __init__(self, model, decay=0.9999, use_num_upates=True):
        """Init.

        Args:
            model: The model.
            decay: The decay.
            use_num_upates: The use num upates.
        """
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        # Keep the coefficient in FP32, matching the released LVDM EMA. In
        # BF16, 0.9999 rounds to 1.0 and the long-run average stops updating.
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self,model):
        """Forward.

        Args:
            model: The model.
        """
        updates_enabled = self.num_updates.ge(0)
        self.num_updates.add_(updates_enabled.to(dtype=self.num_updates.dtype))
        scheduled_decay = (1 + self.num_updates) / (10 + self.num_updates)
        decay = torch.where(updates_enabled, torch.minimum(self.decay, scheduled_decay), self.decay)
        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())
            grouped: dict[
                tuple[torch.device, torch.dtype],
                tuple[list[torch.Tensor], list[torch.Tensor]],
            ] = {}
            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow = get_local_tensor_if_dtensor(shadow_params[sname])
                    parameter = get_local_tensor_if_dtensor(m_param[key]).detach()
                    if parameter.device != shadow.device or parameter.dtype != shadow.dtype:
                        parameter = parameter.to(device=shadow.device, dtype=shadow.dtype)
                    shadows, parameters = grouped.setdefault((shadow.device, shadow.dtype), ([], []))
                    shadows.append(shadow)
                    parameters.append(parameter)
                else:
                    assert key not in self.m_name2s_name
            # Numerical-behavior note: foreach_lerp evaluates the same EMA
            # equation in fused device kernels, but the operation ordering can
            # differ by a final rounding bit from the former per-parameter
            # ``sub_(shadow - parameter)`` expression (CC-37).
            for (device, dtype), (shadows, parameters) in grouped.items():
                del dtype
                weight = one_minus_decay.to(device=device)
                torch._foreach_lerp_(shadows, parameters, weight)

    # ──────────────────────────────────────────────────────────────────────
    # Eval copy / store-restore — swap shadows in without a second optimizer
    # ──────────────────────────────────────────────────────────────────────

    def copy_to(self, model):
        """Copy to.

        Args:
            model: The model.
        """
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                # Buffer names dropped '.' so the map is required to copy back.
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert key not in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        parameters = list(parameters)
        if len(parameters) != len(self.collected_params):
            raise ValueError(
                "restore parameter count does not match the preceding store: "
                f"{len(parameters)} != {len(self.collected_params)}"
            )
        for c_param, param in zip(self.collected_params, parameters, strict=True):
            param.data.copy_(c_param.data)
