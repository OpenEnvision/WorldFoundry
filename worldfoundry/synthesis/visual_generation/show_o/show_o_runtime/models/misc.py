from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from collections.abc import Sized as Sized
from typing import (
    Any,
)
from typing import (  # noqa: UP035 - Re-exported to modules that star-import this namespace.
    Dict as Dict,
)
from typing import (  # noqa: UP035 - Re-exported to modules that star-import this namespace.
    List as List,
)
from typing import (
    Literal as Literal,
)
from typing import (
    NamedTuple as NamedTuple,
)
from typing import (
    NewType as NewType,
)
from typing import (
    Optional as Optional,
)
from typing import (  # noqa: UP035 - Re-exported to modules that star-import this namespace.
    Tuple as Tuple,
)
from typing import (  # noqa: UP035 - Re-exported to modules that star-import this namespace.
    Type as Type,
)
from typing import (
    TypeVar as TypeVar,
)
from typing import (
    Union as Union,
)

import torch

# Config type
from omegaconf import DictConfig, OmegaConf

# PyTorch Tensor type
from torch import Tensor as Tensor


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def parse_structured(fields: Any, cfg: dict | DictConfig | None = None) -> Any:
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    if "--local-rank" in cfg:
        del cfg["--local-rank"]
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
