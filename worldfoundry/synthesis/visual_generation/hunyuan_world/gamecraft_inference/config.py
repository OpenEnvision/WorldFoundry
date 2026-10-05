# Adapted from Tencent Hunyuan-GameCraft-1.0, under the Tencent Hunyuan Community License.
"""Inference defaults are maintained under worldfoundry/data."""

from argparse import Namespace

from ..runtime_config import load_hunyuan_world_runtime_defaults


def parse_args(args=None):
    if args:
        raise ValueError("Pass inference options through HunyuanGameCraftPipeline.")
    return Namespace(**load_hunyuan_world_runtime_defaults("game_craft"))
