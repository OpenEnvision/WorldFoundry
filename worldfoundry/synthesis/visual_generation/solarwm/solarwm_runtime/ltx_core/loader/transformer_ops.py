# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SolarWM execution policy for the canonical LTX graphs."""

from worldfoundry.core.nn import PytorchAdaZeroFunction, TransformerAttentionOps, TransformerOpsConfig

from .attention_policy import automatic_attention, automatic_masked_attention


def build_attention_ops() -> TransformerAttentionOps:
    """Keep SolarWM's independently selected masked and unmasked backends."""
    return TransformerAttentionOps(
        attention_function=automatic_attention(),
        masked_attention_function=automatic_masked_attention(),
    )


DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig(
    attention_ops=build_attention_ops(),
    ada_zero_function=PytorchAdaZeroFunction(backend="torch"),
)
