# Adapted from https://github.com/eloialonso/diamond/tree/csgo
from __future__ import annotations

import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SegmentId:
    episode_id: Union[int, str]
    start: int
    stop: int


@dataclass
class Segment:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    mask_padding: torch.BoolTensor
    info: Dict[str, Any]
    id: SegmentId

    @property
    def effective_size(self):
        return self.mask_padding.sum().item()

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
class Episode:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    info: Dict[str, Any]

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

    def compute_metrics(self) -> Dict[str, Any]:
        return {"length": len(self), "return": self.rew.sum().item()}

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        return cls(
            **{
                k: v.div(255).mul(2).sub(1) if k == "obs" else v
                for k, v in torch.load(Path(path), map_location=map_location).items()
            }
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v.add(1).div(2).mul(255).byte() if k == "obs" else v for k, v in self.__dict__.items()}
        torch.save(d, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)


def merge_info(info_a, info_b):
    keys_a = set(info_a)
    keys_b = set(info_b)
    intersection = keys_a & keys_b
    info = {
        **{k: info_a[k] for k in keys_a if k not in intersection},
        **{k: info_b[k] for k in keys_b if k not in intersection},
        **{k: torch.cat((info_a[k], info_b[k]), dim=0) for k in intersection},
    }
    return info

class CSGOHdf5Dataset(StateDictMixin, Dataset):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        filenames = sorted(Path(directory).rglob("*.hdf5"), key=lambda x: int(x.stem.split("_")[-1]))
        self._filenames = {f"{x.parent.name}/{x.name}": x for x in filenames}
        self._length_one_episode = 1000
        self.num_episodes = len(self._filenames)
        self.num_steps = self._length_one_episode * self.num_episodes
        self.lengths = np.array([self._length_one_episode] * self.num_episodes, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        assert segment_id.start < self._length_one_episode and segment_id.stop > 0 and segment_id.start < segment_id.stop
        pad_len_right = max(0, segment_id.stop - self._length_one_episode)
        pad_len_left = max(0, -segment_id.start)

        start = max(0, segment_id.start)
        stop = min(self._length_one_episode, segment_id.stop)
        mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

        with h5py.File(self._filenames[segment_id.episode_id], "r") as f:
            obs = torch.stack([torch.tensor(f[f"frame_{i}_x"][:]).flip(2).permute(2, 0, 1) for i in range(start, stop)])
            act = torch.tensor(np.array([f[f"frame_{i}_y"][:] for i in range(start, stop)]))

        def pad(x):
            right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
            return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

        obs = pad(obs)
        act = pad(act)
        rew = torch.zeros(obs.size(0))
        end = torch.zeros(obs.size(0), dtype=torch.uint8)
        trunc = torch.zeros(obs.size(0), dtype=torch.uint8)
        return Segment(obs, act, rew, end, trunc, mask_padding, info={}, id=SegmentId(segment_id.episode_id, start, stop))

    def load_episode(self, episode_id: int) -> Episode:  # used by DatasetTraverser
        s = self[SegmentId(episode_id, 0, self._length_one_episode)]
        return Episode(s.obs, s.act, s.rew, s.end, s.trunc, s.info)
