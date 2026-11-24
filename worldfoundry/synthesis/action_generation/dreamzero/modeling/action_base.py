# Inference-only DreamZero architecture.
from torch import nn


class ActionHead(nn.Module):
    def __init__(self):
        super().__init__()

    def set_override_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            setattr(self, key, value)
