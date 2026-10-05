# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> aot -> networks -> decoders -> __init__.py functionality."""

from .fpn import FPNSegmentationHead


def build_decoder(name, **kwargs):
    """Build decoder.

    Args:
        name: The name.
    """
    if name == "fpn":
        return FPNSegmentationHead(**kwargs)
    else:
        raise NotImplementedError
