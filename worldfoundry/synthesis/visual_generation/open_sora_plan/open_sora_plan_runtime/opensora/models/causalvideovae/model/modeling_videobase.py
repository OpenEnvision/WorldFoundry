import glob
import os

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import ConfigMixin  # noqa: F811 - Later import provides the binding, as upstream.
from diffusers.models.modeling_utils import ModelMixin  # noqa: F811 - Later import provides the binding, as upstream.


class VideoBaseAE(ModelMixin, ConfigMixin):
    config_name = "config.json"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def encode(self, x: torch.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: torch.Tensor, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, **kwargs):
        ckpt_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*.ckpt"))
        if ckpt_files:
            # Adapt to checkpoint
            last_ckpt_file = ckpt_files[-1]
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            model = cls.from_config(config_file)
            model.init_from_ckpt(last_ckpt_file)
            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
