"""Inference-only configuration for the in-tree MME-VLA runtime."""

from __future__ import annotations

import dataclasses
import difflib
import logging

import numpy as np

from ..openpi import config as _openpi_config
from ..openpi import transforms
from ..openpi.modeling import model as _model
from ..openpi.modeling import tokenizer as _tokenizer
from ..runtime_config import load_vla_va_wam_runtime_config
from .adapters import RoboMMEInputs, RoboMMEOutputs
from .history_config import get_history_config
from .modeling.model import HistoryPi0Config

AssetsConfig = _openpi_config.AssetsConfig
DataConfig = _openpi_config.DataConfig
RuntimeConfig = _openpi_config.RuntimeConfig


class MMEPaligemmaTokenizer(_tokenizer.PaligemmaTokenizer):
    """PaliGemma prompt format used by released MME-VLA checkpoints."""

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        subgoal: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if subgoal is not None:
            subgoal = str(subgoal).strip().replace("_", " ").replace("\n", " ")
            if state is None:
                text = f"Task: {cleaned};\nCurrent Subgoal: {subgoal};\nAction: "
            else:
                text = f"Task: {cleaned}\nCurrent Subgoal: {subgoal}.\nAction: "
            tokens = self._tokenizer.encode(text, add_bos=True)
        elif state is not None:
            state = np.clip(state, -1, 1)
            bins = np.linspace(-1, 1, 257)[:-1]
            state_text = " ".join(map(str, np.digitize(state, bins=bins) - 1))
            tokens = self._tokenizer.encode(
                f"Task: {cleaned}; State: {state_text};\nAction: ",
                add_bos=True,
            )
        else:
            tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")

        if len(tokens) > self._max_len:
            logging.warning(
                "MME-VLA prompt has %d tokens; truncating to %d",
                len(tokens),
                self._max_len,
            )
        tokens = tokens[: self._max_len]
        mask = [True] * len(tokens)
        padding = self._max_len - len(tokens)
        if padding:
            tokens.extend([0] * padding)
            mask.extend([False] * padding)
        return np.asarray(tokens), np.asarray(mask)


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithSymbolicMemory(transforms.DataTransformFn):
    tokenizer: MMEPaligemmaTokenizer
    discrete_state_input: bool = False
    symbolic_memory_type: str | None = None

    def __call__(self, data: dict) -> dict:
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("MME-VLA requires a prompt")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        state = data.get("state") if self.discrete_state_input else None
        if self.discrete_state_input and state is None:
            raise ValueError("MME-VLA discrete-state prompting requires state")

        tokens, token_mask = self.tokenizer.tokenize(prompt, state)
        simple_subgoal = data.pop("simple_subgoal", None)
        grounded_subgoal = data.pop("grounded_subgoal", None)
        result = {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
        }
        if self.symbolic_memory_type is None:
            return result

        subgoal = {
            "simple_subgoal": simple_subgoal,
            "grounded_subgoal": grounded_subgoal,
        }.get(self.symbolic_memory_type)
        if subgoal is None:
            raise ValueError(f"MME-VLA requires {self.symbolic_memory_type}")
        symbolic_tokens, symbolic_mask = self.tokenizer.tokenize(prompt, state, str(subgoal))
        result["symbolic_tokenized_prompt"] = symbolic_tokens
        result["symbolic_tokenized_prompt_mask"] = symbolic_mask
        return result


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory:
    default_prompt: str | None = None

    def __call__(self, model_config: HistoryPi0Config) -> transforms.Group:
        if model_config.model_type is not _model.ModelType.PI05:
            raise ValueError("MME-VLA supports PI0.5 checkpoints only")

        symbolic_type = None
        max_token_len = model_config.max_token_len
        if model_config.use_history and model_config.history_config is not None:
            history = get_history_config(model_config.history_config)
            if history.representation_type == "symbolic":
                symbolic_type = history.symbolic_memory.type
                max_token_len *= 2

        return transforms.Group(
            inputs=[
                transforms.InjectDefaultPrompt(self.default_prompt),
                transforms.ResizeImages(224, 224),
                TokenizePromptWithSymbolicMemory(
                    MMEPaligemmaTokenizer(max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                    symbolic_memory_type=symbolic_type,
                ),
                transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )


@dataclasses.dataclass(frozen=True)
class RoboMMEDataConfig(_openpi_config.DataConfigFactory):
    repo_id: str | None = "robomme"

    def create(self, assets_dirs, model_config: HistoryPi0Config) -> DataConfig:
        data_transforms = transforms.Group(
            inputs=[RoboMMEInputs(model_type=model_config.model_type)],
            outputs=[RoboMMEOutputs()],
        )
        delta_mask = transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[transforms.DeltaActions(delta_mask)],
            outputs=[transforms.AbsoluteActions(delta_mask)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )


def _load_configs() -> tuple[RuntimeConfig, ...]:
    payload = load_vla_va_wam_runtime_config("mme-vla")
    specs = payload.get("policy_configs")
    if not isinstance(specs, dict):
        raise TypeError("mme-vla data config requires a policy_configs mapping")
    configs: list[RuntimeConfig] = []
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise TypeError(f"MME-VLA policy config {name!r} must be a mapping")
        model = spec.get("model")
        data = spec.get("data", {})
        if not isinstance(model, dict) or not isinstance(data, dict):
            raise TypeError(f"MME-VLA policy config {name!r} requires model/data mappings")
        configs.append(
            RuntimeConfig(
                name=str(name),
                model=HistoryPi0Config(**model),
                data=RoboMMEDataConfig(**data),
            )
        )
    return tuple(configs)


_CONFIGS = _load_configs()
_CONFIGS_BY_NAME = {config.name: config for config in _CONFIGS}


def get_config(name: str) -> RuntimeConfig:
    try:
        return _CONFIGS_BY_NAME[name]
    except KeyError as error:
        closest = difflib.get_close_matches(name, _CONFIGS_BY_NAME, n=1)
        suggestion = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Unknown MME-VLA config '{name}'.{suggestion}") from error
