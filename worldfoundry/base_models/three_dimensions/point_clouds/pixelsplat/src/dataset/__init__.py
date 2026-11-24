from torch.utils.data import Dataset

from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
}


DatasetCfg = DatasetRE10kCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
