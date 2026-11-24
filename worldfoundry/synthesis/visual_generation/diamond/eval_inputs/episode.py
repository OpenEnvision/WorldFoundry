from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from worldfoundry.core.utils.tensors.torch import merge_info as merge_info


@dataclass
class Episode:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    info: dict[str, Any]

    def __len__(self) -> int:
        return self.obs.size(0)

    def __add__(self, other: Episode) -> Episode:
        assert self.dead.sum() == 0
        d = {k: torch.cat((v, other.__dict__[k]), dim=0) for k, v in self.__dict__.items() if k != "info"}
        return Episode(**d, info=merge_info(self.info, other.info))

    def to(self, device) -> Episode:
        return Episode(**{k: v.to(device) if k != "info" else v for k, v in self.__dict__.items()})

    @property
    def dead(self) -> torch.ByteTensor:
        return (self.end + self.trunc).clip(max=1)

    def compute_metrics(self) -> dict[str, Any]:
        return {"length": len(self), "return": self.rew.sum().item()}

    @classmethod
    def load(cls, path: Path, map_location: torch.device | None = None) -> Episode:
        return cls(
            **{
                k: v.div(255).mul(2).sub(1) if k == "obs" else v
                for k, v in torch.load(
                    Path(path),
                    map_location=map_location,
                    weights_only=True,
                ).items()
            }
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v.add(1).div(2).mul(255).byte() if k == "obs" else v for k, v in self.__dict__.items()}
        torch.save(d, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
