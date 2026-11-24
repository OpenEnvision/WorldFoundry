import importlib
from typing import *  # noqa: F403 - Upstream star import kept as written.

if TYPE_CHECKING:  # noqa: F405 - Name comes from the upstream star import.
    from .v1 import MoGeModel as MoGeModelV1
    from .v2 import MoGeModel as MoGeModelV2


def import_model_class_by_version(version: str) -> Type[Union["MoGeModelV1", "MoGeModelV2"]]:  # noqa: F405 - Name comes from the upstream star import.
    assert version in ["v1", "v2"], f"Unsupported model version: {version}"

    try:
        module = importlib.import_module(f".{version}", __package__)
    except ModuleNotFoundError:
        raise ValueError(f'Model version "{version}" not found.')  # noqa: B904 - Keeps the implicit exception context.

    cls = module.MoGeModel
    return cls
