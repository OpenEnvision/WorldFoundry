import importlib

_CANONICAL_PREFIX = "worldfoundry.base_models.perception_core.frame_interpolation.amt."


def base_build_fn(module, cls, params):
    return getattr(importlib.import_module(module, package=None), cls)(**params)


def build_from_cfg(config):
    module, cls = config["name"].rsplit(".", 1)
    if module.startswith("networks."):
        module = _CANONICAL_PREFIX + module
    params = config.get("params", {})
    return base_build_fn(module, cls, params)
