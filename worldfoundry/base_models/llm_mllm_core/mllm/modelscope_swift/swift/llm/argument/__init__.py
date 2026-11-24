# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .base_args import BaseArguments as BaseArguments
    from .deploy_args import DeployArguments as DeployArguments
    from .infer_args import InferArguments as InferArguments
else:
    _import_structure = {
        "base_args": ["BaseArguments"],
        "deploy_args": ["DeployArguments"],
        "infer_args": ["InferArguments"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
