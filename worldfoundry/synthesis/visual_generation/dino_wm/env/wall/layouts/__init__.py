from enum import Enum, auto

from .configs import ConfigBase as ConfigBase
from .single import DotDataset as DotDataset
from .single import DotDatasetConfig as DotDatasetConfig
from .single import Sample as Sample
from .wall import WallDataset as WallDataset
from .wall import WallDatasetConfig as WallDatasetConfig


class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    Wall = auto()
    WallExpert = auto()
    WallEigenfunc = auto()
