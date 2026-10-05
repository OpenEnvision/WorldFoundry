from .base import WorldModelRunner
from .veo import VeoRunner
from .local_script import LocalScriptRunner
from .sora import SoraRunner
from .lingbot import LingBotRunner

# Maps API provider name -> adapter class.
# Add new API-based models here.
_API_PROVIDERS = {
    "google_veo": VeoRunner,
    "openai_sora": SoraRunner,
}

# Maps runner_class name -> adapter class for local models.
# Add new local model runners here.
_LOCAL_RUNNERS = {
    "lingbot": LingBotRunner,
}


def build_runner(name: str, config: dict) -> WorldModelRunner:
    """Instantiate the correct adapter from a model config dict.

    Args:
        name:   Model name (used for logging and output folder naming).
        config: A single model entry from configs/models.yaml.

    Returns:
        A ready-to-use WorldModelRunner instance.
    """
    model_type = config.get("type", "")

    if model_type == "api":
        provider = config.get("provider", "")
        cls = _API_PROVIDERS.get(provider)
        if cls is None:
            raise ValueError(
                f"Unknown API provider '{provider}' for model '{name}'. "
                f"Available: {list(_API_PROVIDERS)}"
            )
        return cls(name, config)

    if model_type == "local":
        runner_class = config.get("runner_class")
        if runner_class:
            cls = _LOCAL_RUNNERS.get(runner_class)
            if cls is None:
                raise ValueError(
                    f"Unknown runner_class '{runner_class}' for model '{name}'. "
                    f"Available: {list(_LOCAL_RUNNERS)}"
                )
            return cls(name, config)
        return LocalScriptRunner(name, config)

    raise ValueError(
        f"Unknown model type '{model_type}' for model '{name}'. "
        f"Expected 'api' or 'local'."
    )


__all__ = ["WorldModelRunner", "VeoRunner", "LocalScriptRunner", "SoraRunner", "LingBotRunner", "build_runner"]
