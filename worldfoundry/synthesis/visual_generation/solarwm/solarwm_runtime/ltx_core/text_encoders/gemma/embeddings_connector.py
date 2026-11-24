# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SolarWM connector construction with the family's execution policy."""

from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_connector import (
    Embeddings1DConnector,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_connector import (
    Embeddings1DConnectorConfigurator as CanonicalConnectorConfigurator,
)

from ...loader.transformer_ops import build_attention_ops


class Embeddings1DConnectorConfigurator(CanonicalConnectorConfigurator):
    @classmethod
    def from_metadata(cls, metadata: dict) -> Embeddings1DConnector:
        """Read the canonical connector topology and retain SolarWM backend selection."""
        return cls.from_config(metadata.get("config", {}), attention_ops=build_attention_ops())


class AudioEmbeddings1DConnectorConfigurator(Embeddings1DConnectorConfigurator):
    _DIMENSION_PREFIX = "audio_connector_"
