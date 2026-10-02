from __future__ import annotations

"""Point-cloud generation representation modules."""

from importlib import import_module

_EXPORTS = {
    "CUT3RRepresentation": ".cut3r.cut3r_representation",
    "FlashWorldRepresentation": ".flash_world.flash_world_representation",
    "HunyuanWorldMirrorRepresentation": ".hunyuan_world.hunyuan_world_mirror_representation",
    "HunyuanWorldVoyagerRepresentation": ".hunyuan_world.hunyuan_world_voyager_representation",
    "InfiniteVGGTRepresentation": ".vggt",
    "LingBotMapRepresentation": ".lingbot_map",
    "LoGeRRepresentation": ".pi3",
    "Lyra1Representation": ".lyra",
    "Lyra2Representation": ".lyra",
    "Pi3Representation": ".pi3",
    "Pi3XRepresentation": ".pi3",
    "VGGTOmegaRepresentation": ".vggt",
    "VGGTRepresentation": ".vggt",
    "WorldFMRepresentation": ".worldfm",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
