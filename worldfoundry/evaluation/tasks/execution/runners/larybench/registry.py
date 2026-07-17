import json
import os
from importlib import import_module
from pathlib import Path


def _config_root() -> Path:
    configured = os.environ.get("LARY_CONFIG_ROOT")
    if configured:
        return Path(configured)
    from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

    return bundled_benchmark_asset("larybench", "configs")


class Registry:
    def __init__(self):
        self._modules = {}

    def register_module(self):
        def register(module):
            self._modules[module.__name__] = module
            return module

        return register

    def build(self, config_name):
        config_path = _config_root() / "models" / f"{config_name}.json"
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        model_type = config.pop("type")
        if model_type not in self._modules:
            import_module(f"{__package__}.models")
        try:
            model_class = self._modules[model_type]
        except KeyError as error:
            raise KeyError(f"Unknown LARYBench model type: {model_type}") from error
        return model_class(**config)


MODEL = Registry()
