"""First-frame image generation."""

from wrbench.firstframe.generate import (
    DashScopeT2IProvider,
    FirstFrameManifest,
    MockT2IProvider,
    generate_first_frame,
    generate_first_frames_from_families,
    get_t2i_provider,
    write_manifest,
)

__all__ = [
    "DashScopeT2IProvider",
    "FirstFrameManifest",
    "MockT2IProvider",
    "generate_first_frame",
    "generate_first_frames_from_families",
    "get_t2i_provider",
    "write_manifest",
]
