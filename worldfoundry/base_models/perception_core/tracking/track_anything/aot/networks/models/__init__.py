# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

from .aot import AOT
from .deaot import DeAOT


def build_vos_model(name, cfg, **kwargs):
    if name == "aot":
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == "deaot":
        return DeAOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
