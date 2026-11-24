from typing import Any

import attrs

from worldfoundry.core.configuration import Config as InferenceConfig
from worldfoundry.core.configuration.hydra import (
    import_all_modules_from_package,
)
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.configs.defaults.common.checkpoint import (
    register_checkpoint,
)
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.configs.defaults.common.tokenizer import (
    register_tokenizer,
)
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.configs.defaults.conditioner import (
    lyra_register_conditioner,
)
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.configs.defaults.model import lyra_register_model
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.configs.defaults.net import lyra_register_net


@attrs.define(slots=False)
class Config(InferenceConfig):
    defaults: list[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "fsdp_wan2pt1_lyra2_spatial"},
            {"net": "wan2pt1_14B_i2v_lyra2"},
            {"conditioner": "lyra2_conditioner"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "local"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(model=None)

    c.job.project = "lyra_2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Register common defaults
    register_tokenizer()
    register_checkpoint()

    # Register lyra_2-specific configs
    lyra_register_model()
    lyra_register_net()
    lyra_register_conditioner()

    # Register experiment configs
    import_all_modules_from_package("worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.configs", reload=True)

    return c
