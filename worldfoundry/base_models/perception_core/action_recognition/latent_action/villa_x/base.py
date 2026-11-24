"""Minimal config and checkpoint loading for the Villa-X encoder."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from torch import nn


class PretrainedConfig(BaseModel):
    _config_path: str | None = None

    def resolve_path(self) -> None:
        for key, value in self.model_dump().items():
            if isinstance(value, str) and "$CONFIG_PATH" in value:
                setattr(self, key, value.replace("$CONFIG_PATH", self._config_path))


class ModelConfig(BaseModel):
    architecture: str
    config: dict

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        return cls.model_validate_json(Path(path).read_text())


class PretrainedModel(nn.Module):
    config_class: type[PretrainedConfig]

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        pass

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_igor_initialized", False):
            return
        self._init_weights(module)
        module._igor_initialized = True

    def post_init(self) -> None:
        self.apply(self._initialize_weights)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, strict: bool = False, **kwargs) -> PretrainedModel:
        from safetensors.torch import load_model

        model_dir = Path(pretrained_model_path)
        config_path = model_dir / "model_config.json"
        weights_path = model_dir / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Expected model_config.json and model.safetensors in {model_dir}")

        model_config = ModelConfig.load(config_path)
        if model_config.architecture != cls.__name__:
            raise ValueError(f"Expected architecture {cls.__name__}, got {model_config.architecture}")

        config = cls.config_class.model_validate(model_config.config)
        config._config_path = model_dir.absolute().as_posix()
        config.resolve_path()

        model = cls(config, **kwargs)
        model.eval()
        load_model(model=model, filename=weights_path, strict=strict)
        return model
