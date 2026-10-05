from __future__ import annotations

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.adapter import (
    WanAutoencoderAdapterMixin,
    WanAutoencoderConfig,
)


class _Adapter(WanAutoencoderAdapterMixin):
    config = WanAutoencoderConfig(
        latent_channels=48,
        temporal_compression_ratio=4,
        spatial_compression_ratio=8,
    )


def test_adapter_exposes_legacy_wan_geometry_attributes() -> None:
    adapter = _Adapter()

    assert adapter.latent_channels == 48
    assert adapter.temporal_compression_ratio == 4
    assert adapter.spatial_compression_ratio == 8
