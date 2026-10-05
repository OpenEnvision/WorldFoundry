"""Visual generation synthesis modules."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AnimateDiffSynthesis": ".animatediff",
    "CameraCtrlSynthesis": ".cameractrl",
    "CogVideoX2bT2VSynthesis": ".cogvideox",
    "CogVideoX5bI2VSynthesis": ".cogvideox",
    "CogVideoX5bT2VSynthesis": ".cogvideox",
    "FantasyWorldWan21Synthesis": ".fantasy_world",
    "FantasyWorldWan22Synthesis": ".fantasy_world",
    "EchoInfinitySynthesis": ".echo_infinity",
    "Gen3CSynthesis": ".gen3c",
    "HYWorld2PanoSynthesis": ".hunyuan_world",
    "HYWorld2Synthesis": ".hunyuan_world",
    "HunyuanGameCraftSynthesis": ".hunyuan_world",
    "HunyuanWorld1Synthesis": ".hunyuan_world",
    "HunyuanWorldMirrorSynthesis": ".hunyuan_world",
    "HunyuanWorldPlaySynthesis": ".hunyuan_world",
    "HunyuanWorldVoyagerSynthesis": ".hunyuan_world",
    "IRASimSynthesis": ".irasim",
    "InfiniteWorldSynthesis": ".infinite_world",
    "InspatioWorldSynthesis": ".inspatio_world",
    "Lyra1Synthesis": ".lyra_1",
    "Lyra2Synthesis": ".lyra_2",
    "MotionCtrlSynthesis": ".motionctrl",
    "MoVerseSynthesis": ".moverse",
    "NeoVerseSynthesis": ".neoverse",
    "OpenMAGVIT2Synthesis": ".open_magvit2",
    "PandoraSynthesis": ".pandora",
    "PixelSplatSynthesis": ".pixelsplat",
    "RollingForcingSynthesis": ".rolling_forcing",
    "SanaSynthesis": ".sana",
    "SCOPESynthesis": ".scope",
    "ShowOSynthesis": ".show_o",
    "StableVideoInfinitySynthesis": ".stable_video_infinity",
    "Uni3CSynthesis": ".uni3c",
    "VMemSynthesis": ".vmem",
    "VideoCrafter1I2VSynthesis": ".videocrafter",
    "VideoCrafter1T2VSynthesis": ".videocrafter",
    "VideoCrafter2T2VSynthesis": ".videocrafter",
    "Wan2p5Synthesis": ".wan",
    "Wan2p6Synthesis": ".wan",
    "Wan2p7Synthesis": ".wan",
    "WorldFMSynthesis": ".worldfm",
    "ZeroScopeSynthesis": ".zeroscope",
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
