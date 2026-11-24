# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .adapter import Adapter as Adapter
    from .adapter import AdapterConfig as AdapterConfig
    from .adapter import AdapterModule as AdapterModule
    from .base import Swift as Swift
    from .base import SwiftModel as SwiftModel

    # isort: split
    from .lora import LoRA as LoRA
    from .lora import LoRAConfig as LoRAConfig

    # isort: split
    # isort: split
    from .mapping import SWIFT_MAPPING as SWIFT_MAPPING
    from .mapping import SwiftTuners as SwiftTuners

    # isort: split
    from .neftune import NEFTune as NEFTune
    from .neftune import NEFTuneConfig as NEFTuneConfig
    from .side import Side as Side
    from .side import SideConfig as SideConfig
    from .side import SideModule as SideModule

    # isort: split
    from .longlora.longlora import LongLoRA as LongLoRA
    from .longlora.longlora import LongLoRAConfig as LongLoRAConfig
    from .longlora.longlora import LongLoRAModelType as LongLoRAModelType

    # isort: split
    from .llamapro import LLaMAPro as LLaMAPro
    from .llamapro import LLaMAProConfig as LLaMAProConfig
    from .reft import Reft as Reft
    from .reft import ReftConfig as ReftConfig
    from .restuning import ResTuning as ResTuning
    from .restuning import ResTuningBypassModule as ResTuningBypassModule
    from .restuning import ResTuningConfig as ResTuningConfig

    # isort: split
    from .peft import (
        AdaLoraConfig as AdaLoraConfig,
    )
    from .peft import (
        BOFTConfig as BOFTConfig,
    )
    from .peft import (
        LoftQConfig as LoftQConfig,
    )
    from .peft import (
        LoHaConfig as LoHaConfig,
    )
    from .peft import (
        LoKrConfig as LoKrConfig,
    )
    from .peft import (
        LoraConfig as LoraConfig,
    )
    from .peft import (
        OFTConfig as OFTConfig,
    )
    from .peft import (
        PeftConfig as PeftConfig,
    )
    from .peft import (
        PeftModel as PeftModel,
    )
    from .peft import (
        PeftModelForCausalLM as PeftModelForCausalLM,
    )
    from .peft import (
        PeftModelForSeq2SeqLM as PeftModelForSeq2SeqLM,
    )
    from .peft import (
        PeftModelForSequenceClassification as PeftModelForSequenceClassification,
    )
    from .peft import (
        PeftModelForTokenClassification as PeftModelForTokenClassification,
    )
    from .peft import (
        PrefixTuningConfig as PrefixTuningConfig,
    )
    from .peft import (
        PromptEncoderConfig as PromptEncoderConfig,
    )
    from .peft import (
        PromptLearningConfig as PromptLearningConfig,
    )
    from .peft import (
        PromptTuningConfig as PromptTuningConfig,
    )
    from .peft import (
        VeraConfig as VeraConfig,
    )
    from .peft import (
        get_peft_config as get_peft_config,
    )
    from .peft import (
        get_peft_model as get_peft_model,
    )
    from .peft import (
        get_peft_model_state_dict as get_peft_model_state_dict,
    )

    # isort: split
    from .prompt import Prompt as Prompt
    from .prompt import PromptConfig as PromptConfig
    from .prompt import PromptModule as PromptModule
    from .scetuning.scetuning import SCETuning as SCETuning
    from .scetuning.scetuning import SCETuningConfig as SCETuningConfig
    from .utils import SwiftConfig as SwiftConfig
    from .utils import SwiftOutput as SwiftOutput
    from .utils import swift_to_peft_format as swift_to_peft_format
else:
    _import_structure = {
        "adapter": ["Adapter", "AdapterConfig", "AdapterModule"],
        "base": ["SwiftModel", "Swift"],
        "lora": ["LoRA", "LoRAConfig"],
        "longlora.longlora": ["LongLoRAModelType", "LongLoRAConfig", "LongLoRA"],
        "mapping": ["SWIFT_MAPPING", "SwiftTuners"],
        "side": ["Side", "SideConfig", "SideModule"],
        "reft": ["Reft", "ReftConfig"],
        "llamapro": ["LLaMAPro", "LLaMAProConfig"],
        "neftune": ["NEFTune", "NEFTuneConfig"],
        "restuning": ["ResTuning", "ResTuningConfig", "ResTuningBypassModule"],
        "peft": [
            "AdaLoraConfig",
            "LoftQConfig",
            "LoHaConfig",
            "LoKrConfig",
            "LoraConfig",
            "VeraConfig",
            "BOFTConfig",
            "OFTConfig",
            "PeftConfig",
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
        ],
        "prompt": ["Prompt", "PromptConfig", "PromptModule"],
        "scetuning": ["SCETuning", "SCETuningConfig"],
        "utils": ["SwiftConfig", "SwiftOutput", "swift_to_peft_format"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
