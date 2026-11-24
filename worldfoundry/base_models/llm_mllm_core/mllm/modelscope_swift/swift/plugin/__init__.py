# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .agent_template import agent_templates as agent_templates
    from .callback import extra_callbacks as extra_callbacks
    from .loss import LOSS_MAPPING as LOSS_MAPPING
    from .loss import get_loss_func as get_loss_func
    from .loss_scale import loss_scale_map as loss_scale_map
    from .metric import InferStats as InferStats
    from .metric import MeanMetric as MeanMetric
    from .metric import Metric as Metric
    from .metric import compute_acc as compute_acc
    from .metric import compute_rouge_bleu as compute_rouge_bleu
    from .metric import get_metric as get_metric
    from .multi_turn import multi_turns as multi_turns
    from .orm import ORM as ORM
    from .orm import orms as orms
    from .prm import PRM as PRM
    from .prm import prms as prms
    from .tuner import PeftTuner as PeftTuner
    from .tuner import Tuner as Tuner
    from .tuner import extra_tuners as extra_tuners

    # isort: split
    from .rm_plugin import rm_plugins as rm_plugins

else:
    _import_structure = {
        "callback": ["extra_callbacks"],
        "loss": ["LOSS_MAPPING", "get_loss_func"],
        "loss_scale": ["loss_scale_map"],
        "metric": ["InferStats", "MeanMetric", "Metric", "compute_acc", "get_metric", "compute_rouge_bleu"],
        "agent_template": ["agent_templates"],
        "tuner": ["Tuner", "extra_tuners", "PeftTuner"],
        "prm": ["prms", "PRM"],
        "orm": ["orms", "ORM"],
        "multi_turn": ["multi_turns"],
        "rm_plugin": ["rm_plugins"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
