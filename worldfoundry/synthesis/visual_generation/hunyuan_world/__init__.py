"""Lazy exports for Hunyuan World synthesis modules."""

from importlib import import_module

__all__ = [
    "hunyuan_worldplay",
    "HunyuanWorldPlaySynthesis",
    "_HunyuanWorldPlayInternalPipeline",
    "HunyuanWorldVoyagerSynthesis",
    "HunyuanGameCraftSynthesis",
    "HunyuanWorldIntegrationError",
    "HunyuanWorldIntegrationStatus",
    "HunyuanWorldPlanSynthesis",
    "HunyuanWorld1Synthesis",
    "HunyuanWorldMirrorSynthesis",
    "HYWorld2Synthesis",
    "HYWorld2PanoSynthesis",
    "apply_hunyuan_world_argparse_defaults",
    "hunyuan_world_runtime_config_path",
    "load_hunyuan_world_runtime_defaults",
    "load_hunyuan_world_state_dict",
    "generate_crop_size_list",
    "get_closest_ratio",
    "resize_and_center_crop",
]


def __getattr__(name):
    if name == "hunyuan_worldplay":
        module = import_module(
            "worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay"
        )
        globals()[name] = module
        return module
    if name in {"HunyuanWorldPlaySynthesis", "_HunyuanWorldPlayInternalPipeline"}:
        module = import_module(".hunyuan_worldplay_synthesis", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "HunyuanWorldVoyagerSynthesis":
        module = import_module(".hunyuan_world_voyager_synthesis", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "HunyuanGameCraftSynthesis":
        module = import_module(".hunyuan_game_craft_synthesis", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {
        "HunyuanWorldIntegrationError",
        "HunyuanWorldIntegrationStatus",
        "HunyuanWorldPlanSynthesis",
        "HunyuanWorld1Synthesis",
        "HunyuanWorldMirrorSynthesis",
        "HYWorld2Synthesis",
        "HYWorld2PanoSynthesis",
    }:
        module = import_module(".hunyuan_world_family_synthesis", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {
        "apply_hunyuan_world_argparse_defaults",
        "hunyuan_world_runtime_config_path",
        "load_hunyuan_world_runtime_defaults",
    }:
        module = import_module(".runtime_config", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "load_hunyuan_world_state_dict":
        module = import_module(".checkpoint", __name__)
        value = module.load_hunyuan_world_state_dict
        globals()[name] = value
        return value
    if name in {"generate_crop_size_list", "get_closest_ratio"}:
        module = import_module(".data_utils", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "resize_and_center_crop":
        from worldfoundry.core.media.processing.image_utils import resize_and_center_crop

        globals()[name] = resize_and_center_crop
        return resize_and_center_crop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
