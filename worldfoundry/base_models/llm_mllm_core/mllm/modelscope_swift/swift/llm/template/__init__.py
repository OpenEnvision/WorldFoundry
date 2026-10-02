# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .base import MaxLengthError, Template
    from .constant import TemplateType
    from .grounding import draw_bbox
    from .register import TEMPLATE_MAPPING, get_template, get_template_meta, register_template
    from .template_inputs import InferRequest, TemplateInputs
    from .template_meta import TemplateMeta
    from .utils import Prompt, Word, split_str_parts_by
    from .vision_utils import load_file, load_image
else:
    _import_structure = {
        "base": ["MaxLengthError", "Template"],
        "constant": ["TemplateType"],
        "grounding": ["draw_bbox"],
        "register": ["TEMPLATE_MAPPING", "get_template", "get_template_meta", "register_template"],
        "template_inputs": ["InferRequest", "TemplateInputs"],
        "template_meta": ["TemplateMeta"],
        "utils": ["Prompt", "Word", "split_str_parts_by"],
        "vision_utils": ["load_file", "load_image"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
