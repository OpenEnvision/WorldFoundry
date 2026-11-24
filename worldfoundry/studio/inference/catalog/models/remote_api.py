"""Curated overrides for hosted (remote API) generation models."""

from __future__ import annotations

from typing import Any

OVERRIDES: dict[str, dict[str, Any]] = {
    "worldlabs-marble-1.1": {
        "display_name": "World Labs Marble 1.1",
        "category": "Remote API",
        "default_backend": "api_init",
        "supports_api_init": True,
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_endpoint": "https://api.worldlabs.ai",
        "default_task_type": "3dgs-world-generation",
        "tags": ("api", "3dgs", "worldlabs", "marble"),
        "env_hints": ("WORLDLABS_API_KEY",),
        "notes": "3D Gaussian splat world generation via the World Labs Marble API.",
    },
    "wan-2p5": {
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://dashscope.aliyuncs.com/api/v1",
        "tags": ("api", "text-to-video", "image-to-video"),
    },
    "wan-2p6": {
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://dashscope.aliyuncs.com/api/v1",
        "tags": ("api", "image-to-video"),
    },
    "wan-2p7": {
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://dashscope.aliyuncs.com/api/v1",
        "tags": ("api", "image-to-video"),
    },
    "worldlabs": {
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://api.worldlabs.ai",
        "tags": ("api", "world-json", "assets"),
        "notes": "Can return structured world JSON plus downloadable assets.",
    },
    "runway-gen4p5": {
        "display_name": "Runway Gen-4.5",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://api.dev.runwayml.com/v1",
        "tags": ("api", "hosted-video"),
    },
    "luma-ray2": {
        "display_name": "Luma Ray2",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "hosted-video"),
    },
    "kling-api": {
        "display_name": "Kling API",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "hosted-video"),
    },
    "veo3": {
        "display_name": "Veo3",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://generativelanguage.googleapis.com",
        "tags": ("api", "t2v", "i2v"),
    },
    "sora2": {
        "display_name": "Sora2",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "t2v", "i2v"),
    },
    "hailuo-2p3": {
        "display_name": "Hailuo 2.3",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "audio-video"),
    },
}
