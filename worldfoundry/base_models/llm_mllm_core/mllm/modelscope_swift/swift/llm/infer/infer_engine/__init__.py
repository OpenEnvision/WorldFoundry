# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .vllm_engine import VllmEngine as VllmEngine

    # isort: split
    # isort: split
    from .grpo_vllm_engine import GRPOVllmEngine as GRPOVllmEngine

    # isort: split
    # isort: split
    from .lmdeploy_engine import LmdeployEngine as LmdeployEngine

    # isort: split
    # isort: split
    from .pt_engine import PtEngine as PtEngine

    # isort: split
    # isort: split
    from .infer_client import InferClient as InferClient

    # isort: split
    # isort: split
    from .infer_engine import InferEngine as InferEngine

    # isort: split
    # isort: split
    from .base import BaseInferEngine as BaseInferEngine

    # isort: split
    # isort: split
    from .utils import AdapterRequest as AdapterRequest
    from .utils import patch_vllm_memory_leak as patch_vllm_memory_leak
    from .utils import prepare_generation_config as prepare_generation_config
else:
    _import_structure = {
        "vllm_engine": ["VllmEngine"],
        "grpo_vllm_engine": ["GRPOVllmEngine"],
        "lmdeploy_engine": ["LmdeployEngine"],
        "pt_engine": ["PtEngine"],
        "infer_client": ["InferClient"],
        "infer_engine": ["InferEngine"],
        "base": ["BaseInferEngine"],
        "utils": ["prepare_generation_config", "AdapterRequest", "patch_vllm_memory_leak"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
