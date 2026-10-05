from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .blocks import Conv3x3, SmallResBlock
from utils import init_lstm


ActorCriticOutput = namedtuple("ActorCriticOutput", "logits_act val hx_cx")




@dataclass
class ActorCriticConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    channels: List[int]
    down: List[int]
    num_actions: Optional[int] = None


class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.encoder = ActorCriticEncoder(cfg)
        self.lstm_dim = cfg.lstm_dim
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (sum(cfg.down))) ** 2
        self.lstm = nn.LSTMCell(input_dim_lstm, cfg.lstm_dim)
        self.critic_linear = nn.Linear(cfg.lstm_dim, 1)
        self.actor_linear = nn.Linear(cfg.lstm_dim, cfg.num_actions)

        self.actor_linear.weight.data.fill_(0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        init_lstm(self.lstm)

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device


    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> ActorCriticOutput:
        assert obs.ndim == 4
        x = self.encoder(obs)
        x = x.flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        return ActorCriticOutput(self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx))



class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        encoder_layers = [Conv3x3(cfg.img_channels, cfg.channels[0])]
        for i in range(len(cfg.channels)):
            encoder_layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
