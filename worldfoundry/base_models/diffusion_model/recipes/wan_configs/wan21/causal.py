# SPDX-License-Identifier: Apache-2.0
"""FSDP shard predicate and DiT config for causal Wan 2.1 graphs.

:func:`is_causal_block` matches ``blocks.<index>…`` module names so
FSDP2 shards each transformer block.  :class:`CausalWanArchConfig`
installs that predicate; :class:`CausalWanConfig` is the
:class:`~worldfoundry.core.configuration.model_config.DiTConfig`
wrapper used by causal / streaming Wan 2.1 recipes.

This is wiring only; the causal attention mask lives on the network,
not in these dataclasses.
"""

from dataclasses import dataclass, field

from worldfoundry.core.configuration.model_config import DiTArchConfig, DiTConfig


def is_causal_block(n: str, m) -> bool:
    """Return True when ``n`` is a ``blocks.<index>…`` FSDP shard root.

    ``m`` is unused; the signature matches the generic shard predicate.
    """
    parts = n.split(".")
    return len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit()


@dataclass
class CausalWanArchConfig(DiTArchConfig):
    """Causal wan arch config implementation."""
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_causal_block]
    )


@dataclass
class CausalWanConfig(DiTConfig):
    """Causal wan config implementation."""
    arch_config: DiTArchConfig = field(default_factory=CausalWanArchConfig)

    prefix: str = "CausalWan"
