# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Facebook, Inc. and its affiliates.
"""Detectron2-lineage lazy object graphs for inference configs.

Responsibility: expose :class:`LazyCall`, :class:`LazyConfig`,
:class:`LazyDict`, and :func:`instantiate` as the default way to write
deferred constructor trees (``_target_`` + kwargs).

Not this package: Hydra compose of attrs :class:`Config` lives in
:mod:`worldfoundry.core.configuration.hydra`. DiT architecture
dataclasses live in :mod:`worldfoundry.core.configuration.model_config`.
Legacy ``cls``/``class`` instantiate lives in
:mod:`worldfoundry.core.configuration.hydra_utils`.

Trust boundary: ``.py`` configs loaded here are trusted executable code
(``exec``). YAML configs are data (``yaml.safe_load``).

Public surface: :class:`LazyCall`, :class:`LazyConfig`, :class:`LazyDict`,
:func:`instantiate`, :data:`PLACEHOLDER`.
"""

from omegaconf import DictConfig

from worldfoundry.core.configuration.lazy_config.config import LazyConfig
from worldfoundry.core.configuration.lazy_config.instantiate import instantiate
from worldfoundry.core.configuration.lazy_config.lazy_call import LazyCall

# ──────────────────────────────────────────────────────────────────────────
# Sentinels and marker types — PLACEHOLDER is an unset LazyCall field
# ──────────────────────────────────────────────────────────────────────────

PLACEHOLDER = None


class LazyDict(DictConfig):  # NOTE: to differentiate between LazyDict & DictConfig
    """Marker subclass for editable, lazily instantiated object graphs.

    It behaves like OmegaConf ``DictConfig`` but lets WorldFoundry distinguish
    a deferred constructor tree from an ordinary runtime mapping. Construct it
    through ``LazyCall`` in normal code.
    """

    def __init__(self, *args, **kwargs):
        """Forward content and OmegaConf flags to ``DictConfig``."""
        super().__init__(*args, **kwargs)


__all__ = ["instantiate", "LazyCall", "LazyConfig", "PLACEHOLDER", "LazyDict"]
