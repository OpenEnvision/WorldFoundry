
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


ATARI_100K_GAMES = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MsPacman",
    "Pong",
    "PrivateEye",
    "Qbert",
    "RoadRunner",
    "Seaquest",
    "UpNDown",
]


Logs = list[dict[str, float]]
LossAndLogs = Tuple[Tensor, Dict[str, Any]]


class StateDictMixin:
    def _init_fields(self) -> None:
        def has_sd(x: str) -> bool:
            return callable(getattr(x, "state_dict", None)) and callable(getattr(x, "load_state_dict", None))

        self._all_fields = {k for k in vars(self) if not k.startswith("_")}
        self._fields_sd = {k for k in self._all_fields if has_sd(getattr(self, k))}

    def _get_field(self, k: str) -> Any:
        return getattr(self, k).state_dict() if k in self._fields_sd else getattr(self, k)

    def _set_field(self, k: str, v: Any) -> None:
        getattr(self, k).load_state_dict(v) if k in self._fields_sd else setattr(self, k, v)

    def state_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        return {k: self._get_field(k) for k in self._all_fields}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        assert set(list(state_dict.keys())) == self._all_fields
        for k, v in state_dict.items():
            self._set_field(k, v)


@dataclass
class CommonTools(StateDictMixin):
    denoiser: Any
    rew_end_model: Any
    actor_critic: Any

    def get(self, name: str) -> Any:
        return getattr(self, name)

    def set(self, name: str, value: Any):
        return setattr(self, name, value)




def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def extract_state_dict(state_dict: OrderedDict, module_name: str) -> OrderedDict:
    return OrderedDict({k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})



def init_lstm(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias_ih" in name:
            p.data.fill_(0)
            # Set forget-gate bias to 1
            n = p.size(0)
            p.data[(n // 4) : (n // 2)].fill_(1)
        elif "bias_hh" in name:
            p.data.fill_(0)


def get_path_agent_ckpt(path_ckpt_dir: Union[str, Path], epoch: int, num_zeros: int = 5) -> Path:
    d = Path(path_ckpt_dir) / "agent_versions"
    if epoch >= 0:
        return d / f"agent_epoch_{epoch:0{num_zeros}d}.pt"
    else:
        all_ = sorted(list(d.iterdir()))
        assert len(all_) >= -epoch
        return all_[epoch]




def prompt_atari_game():
    for i, game in enumerate(ATARI_100K_GAMES):
        print(f"{i:2d}: {game}")
    while True:
        x = input("\nEnter a number: ")
        if not x.isdigit():
            print("Invalid.")
            continue
        x = int(x)
        if x < 0 or x > 25:
            print("Invalid.")
            continue
        break
    game = ATARI_100K_GAMES[x]
    return game





def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
