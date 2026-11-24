import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: torch.FloatTensor,
        max_period: float = 10000.0,
    ) -> torch.FloatTensor:
        half_dim = self.dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int, time_cond: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width)
        if time_cond:
            self.linear_2 = nn.Linear(2 * width, width)
        else:
            self.linear_2 = nn.Linear(width, width)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width)
        self.time_cond = time_cond

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        if self.time_cond:
            # repeat time embedding for seq_len
            # [Batch_Size, Seq_Len, Width]
            time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb


class ActionDecoder(nn.Module):
    def __init__(self, action_hidden_size: int, action_dim: int, num_layers: int = 2):
        super().__init__()

        proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(action_hidden_size, action_hidden_size), nn.SiLU()) for _ in range(num_layers - 1)]
        )
        proj.append(nn.Linear(action_hidden_size, action_dim))
        self.proj = nn.Sequential(*proj)

    def forward(
        self,
        action_embed: torch.FloatTensor,
    ) -> torch.FloatTensor:
        action = self.proj(action_embed)
        return action
