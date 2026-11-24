"""Inference-only OpenPI configs used by the WorldFoundry runtime."""

from __future__ import annotations

import abc
import dataclasses
import difflib
import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

from worldfoundry.core.io.paths import resolve_data_path

from . import normalize as _normalize
from . import transforms as _transforms
from .checkpoints import resolve_local_path
from .modeling import model as _model
from .modeling import pi0_config, pi0_fast
from .modeling import tokenizer as _tokenizer
from .policies import aloha_policy, droid_policy, libero_policy

ModelType: TypeAlias = _model.ModelType


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group: ...


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory:
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = model_config.fast_model_tokenizer or _tokenizer.FASTTokenizer
                tokenizer_kwargs = model_config.fast_model_tokenizer_kwargs or {}
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
        raise ValueError(f"Unsupported OpenPI model type: {model_config.model_type}")


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str | None = None
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: DataConfig | None = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create an inference data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        asset_id = self.assets.asset_id or self.repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=self.repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(pathlib.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: pathlib.Path, asset_id: str | None
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = resolve_local_path(
                assets_dir / asset_id,
                label="OpenPI normalization statistics",
                require_directory=True,
            )
            norm_stats = _normalize.load(data_assets_dir)
            logging.info("Loaded norm stats from %s", data_assets_dir)
            return norm_stats
        except FileNotFoundError:
            logging.info("Norm stats not found in %s, skipping.", assets_dir / asset_id)
        return None


@dataclasses.dataclass(frozen=True)
class SyntheticDataConfig(DataConfigFactory):
    repo_id: str | None = "synthetic"

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    data_transforms: GroupFactory | None = None
    model_transforms: GroupFactory = dataclasses.field(default_factory=ModelTransformFactory)

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = (
            self.data_transforms(model_config) if self.data_transforms is not None else _transforms.Group()
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    repack_transforms: _transforms.Group = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """OpenPI runtime config shape retained for upstream policy_config."""

    name: str
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    pytorch_weight_path: str | None = None
    data: DataConfigFactory = dataclasses.field(default_factory=SyntheticDataConfig)
    data_family: str = "synthetic"
    assets_base_dir: str = "./assets"
    seed: int = 42
    policy_metadata: dict[str, Any] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()


def _droid_transforms(model_config: _model.BaseModelConfig) -> _transforms.Group:
    return _transforms.Group(
        inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
        outputs=[droid_policy.DroidOutputs()],
    )


def _model_from_spec(spec: dict[str, Any]) -> _model.BaseModelConfig:
    values = dict(spec)
    model_type = str(values.pop("type"))
    if model_type == "pi0":
        return pi0_config.Pi0Config(**values)
    if model_type == "pi0_fast":
        return pi0_fast.Pi0FASTConfig(**values)
    raise ValueError(f"Unsupported OpenPI model config type: {model_type!r}")


def _data_from_spec(spec: dict[str, Any]) -> DataConfigFactory:
    values = dict(spec)
    kind = str(values.pop("kind"))
    asset_id = values.pop("asset_id", None)
    assets = AssetsConfig(asset_id=str(asset_id) if asset_id is not None else None)
    prompt_from_task = bool(values.pop("prompt_from_task", False))
    if kind == "aloha":
        return LeRobotAlohaDataConfig(
            assets=assets,
            default_prompt=values.pop("default_prompt", None),
            **values,
        )
    if kind == "droid":
        return SimpleDataConfig(
            assets=assets,
            data_transforms=_droid_transforms,
            base_config=DataConfig(prompt_from_task=prompt_from_task),
            **values,
        )
    if kind == "libero":
        return LeRobotLiberoDataConfig(
            assets=assets,
            repo_id=values.pop("repo_id", None),
            base_config=DataConfig(prompt_from_task=prompt_from_task),
            **values,
        )
    raise ValueError(f"Unsupported OpenPI data config kind: {kind!r}")


def _load_configs() -> list[RuntimeConfig]:
    import yaml

    path = resolve_data_path("models", "runtime", "configs", "openpi", "policy_configs.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    records = payload.get("configs") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        raise TypeError(f"OpenPI policy config must contain a configs mapping: {path}")
    configs: list[RuntimeConfig] = []
    for name, record in records.items():
        if not isinstance(record, dict):
            raise TypeError(f"OpenPI config {name!r} must be a mapping")
        values = dict(record)
        model_spec = values.pop("model")
        data_spec = values.pop("data")
        data_family = str(data_spec.get("kind") or "").strip().lower()
        if data_family not in {"aloha", "droid", "libero"}:
            raise ValueError(f"OpenPI config {name!r} has unsupported data family {data_family!r}")
        configs.append(
            RuntimeConfig(
                name=str(name),
                model=_model_from_spec(dict(model_spec)),
                data=_data_from_spec(dict(data_spec)),
                data_family=data_family,
                **values,
            )
        )
    return configs


_CONFIGS = _load_configs()

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("OpenPI config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_config(config_name: str) -> RuntimeConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]


def configure_local_tokenizers(*, paligemma: str, fast: str) -> None:
    """Configure local tokenizer assets before materializing policy transforms."""

    _tokenizer.configure_local_tokenizer_assets(paligemma=paligemma, fast=fast)
