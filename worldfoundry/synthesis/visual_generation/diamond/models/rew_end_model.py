from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .blocks import Conv3x3, Downsample, ResBlocks
from utils import init_lstm


@dataclass
class RewEndModelConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[int]
    num_actions: Optional[int] = None


class RewEndModel(nn.Module):
    def __init__(self, cfg: RewEndModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RewEndEncoder(2 * cfg.img_channels, cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)
        self.act_emb = nn.Embedding(cfg.num_actions, cfg.cond_channels)
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (len(cfg.depths) - 1)) ** 2
        self.lstm = nn.LSTM(input_dim_lstm, cfg.lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, 3 + 2, bias=False),
        )
        init_lstm(self.lstm)

    def predict_rew_end(
        self,
        obs: Tensor,
        act: Tensor,
        next_obs: Tensor,
        hx_cx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        b, t, c, h, w = obs.shape
        obs, act, next_obs = obs.reshape(b * t, c, h, w), act.reshape(b * t), next_obs.reshape(b * t, c, h, w)
        x = self.encoder(torch.cat((obs, next_obs), dim=1), self.act_emb(act))
        x = x.reshape(b, t, -1)  # (b t) e h w -> b t (e h w)
        x, hx_cx = self.lstm(x, hx_cx)
        logits = self.head(x)
        return logits[:, :, :-2], logits[:, :, -2:], hx_cx



class RewEndEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.conv_in = Conv3x3(in_channels, channels[0])
        blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        blocks.append(
            ResBlocks(
                list_in_channels=[channels[-1]] * 2,
                list_out_channels=[channels[-1]] * 2,
                cond_channels=cond_channels,
                attn=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]] + [nn.Identity()])

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block, down in zip(self.blocks, self.downsamples):
            x = down(x)
            x, _ = block(x, cond)
        return x
