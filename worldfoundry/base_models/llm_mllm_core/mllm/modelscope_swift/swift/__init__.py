# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .tuners import (
        SWIFT_MAPPING as SWIFT_MAPPING,
    )
    from .tuners import (
        AdaLoraConfig as AdaLoraConfig,
    )
    from .tuners import (
        Adapter as Adapter,
    )
    from .tuners import (
        AdapterConfig as AdapterConfig,
    )
    from .tuners import (
        AdapterModule as AdapterModule,
    )
    from .tuners import (
        LoftQConfig as LoftQConfig,
    )
    from .tuners import (
        LoHaConfig as LoHaConfig,
    )
    from .tuners import (
        LoKrConfig as LoKrConfig,
    )
    from .tuners import (
        LongLoRA as LongLoRA,
    )
    from .tuners import (
        LongLoRAConfig as LongLoRAConfig,
    )
    from .tuners import (
        LongLoRAModelType as LongLoRAModelType,
    )
    from .tuners import (
        LoRA as LoRA,
    )
    from .tuners import (
        LoRAConfig as LoRAConfig,
    )
    from .tuners import (
        LoraConfig as LoraConfig,
    )
    from .tuners import (
        OFTConfig as OFTConfig,
    )
    from .tuners import (
        PeftConfig as PeftConfig,
    )
    from .tuners import (
        PeftModel as PeftModel,
    )
    from .tuners import (
        PeftModelForCausalLM as PeftModelForCausalLM,
    )
    from .tuners import (
        PeftModelForSeq2SeqLM as PeftModelForSeq2SeqLM,
    )
    from .tuners import (
        PeftModelForSequenceClassification as PeftModelForSequenceClassification,
    )
    from .tuners import (
        PeftModelForTokenClassification as PeftModelForTokenClassification,
    )
    from .tuners import (
        PrefixTuningConfig as PrefixTuningConfig,
    )
    from .tuners import (
        Prompt as Prompt,
    )
    from .tuners import (
        PromptConfig as PromptConfig,
    )
    from .tuners import (
        PromptEncoderConfig as PromptEncoderConfig,
    )
    from .tuners import (
        PromptLearningConfig as PromptLearningConfig,
    )
    from .tuners import (
        PromptModule as PromptModule,
    )
    from .tuners import (
        PromptTuningConfig as PromptTuningConfig,
    )
    from .tuners import (
        ResTuningConfig as ResTuningConfig,
    )
    from .tuners import (
        SCETuning as SCETuning,
    )
    from .tuners import (
        SCETuningConfig as SCETuningConfig,
    )
    from .tuners import (
        SideConfig as SideConfig,
    )
    from .tuners import (
        Swift as Swift,
    )
    from .tuners import (
        SwiftConfig as SwiftConfig,
    )
    from .tuners import (
        SwiftModel as SwiftModel,
    )
    from .tuners import (
        SwiftOutput as SwiftOutput,
    )
    from .tuners import (
        SwiftTuners as SwiftTuners,
    )
    from .tuners import (
        get_peft_config as get_peft_config,
    )
    from .tuners import (
        get_peft_model as get_peft_model,
    )
    from .tuners import (
        get_peft_model_state_dict as get_peft_model_state_dict,
    )
    from .utils import get_logger as get_logger
    from .version import __release_datetime__ as __release_datetime__
    from .version import __version__ as __version__
else:
    _import_structure = {
        "version": ["__release_datetime__", "__version__"],
        "tuners": [
            "Adapter",
            "AdapterConfig",
            "AdapterModule",
            "SwiftModel",
            "LoRA",
            "LoRAConfig",
            "SWIFT_MAPPING",
            "LoraConfig",
            "AdaLoraConfig",
            "LoftQConfig",
            "LoHaConfig",
            "LoKrConfig",
            "OFTConfig",
            "PeftConfig",
            "ResTuningConfig",
            "SideConfig",
            "PeftModel",
            "PeftModelForCausalLM",
            "PeftModelForSeq2SeqLM",
            "PeftModelForSequenceClassification",
            "PeftModelForTokenClassification",
            "PrefixTuningConfig",
            "PromptEncoderConfig",
            "PromptLearningConfig",
            "PromptTuningConfig",
            "get_peft_config",
            "get_peft_model",
            "get_peft_model_state_dict",
            "Prompt",
            "PromptConfig",
            "PromptModule",
            "SwiftConfig",
            "SwiftOutput",
            "Swift",
            "SwiftTuners",
            "LongLoRAConfig",
            "LongLoRA",
            "LongLoRAModelType",
            "SCETuning",
            "SCETuningConfig",
        ],
        "utils": ["get_logger"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
