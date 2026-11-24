# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .base import MaxLengthError as MaxLengthError
    from .base import Template as Template
    from .constant import TemplateType as TemplateType
    from .grounding import draw_bbox as draw_bbox
    from .register import TEMPLATE_MAPPING as TEMPLATE_MAPPING
    from .register import get_template as get_template
    from .register import get_template_meta as get_template_meta
    from .register import register_template as register_template
    from .template_inputs import InferRequest as InferRequest
    from .template_inputs import TemplateInputs as TemplateInputs
    from .template_meta import TemplateMeta as TemplateMeta
    from .utils import Prompt as Prompt
    from .utils import Word as Word
    from .utils import split_str_parts_by as split_str_parts_by
    from .vision_utils import load_file as load_file
    from .vision_utils import load_image as load_image
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
