# ruff: noqa: F722 - jaxtyping dimension strings in annotations.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

try:
    from dacite import Config, from_dict
except ImportError:
    from ...dacite_compat import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ..types import Stage
from .three_view_hack import add_third_context_index
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular)

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config) for k, v in json.load(f).items()
            }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),  # noqa: B008 - Default evaluated once at definition, as upstream.
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)

        # Handle 2-view index for 3 views.
        v = self.cfg.num_context_views
        if v > len(context_indices) and v == 3:
            context_indices = add_third_context_index(context_indices)

        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
