"""Utils visual generation pipeline module."""

from __future__ import annotations

import inspect
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Sequence


class PipelineABC:
    """Shared, non-strict base for WorldFoundry pipelines.

    The class intentionally avoids abstract methods because many existing
    pipelines predate this contract. Subclasses can override any method while
    still sharing a stable framework surface.
    """

    MODEL_ID: ClassVar[str | None] = None
    OPERATOR_CLS: ClassVar[type[Any] | None] = None
    MEMORY_CLS: ClassVar[type[Any] | None] = None
    SYNTHESIS_CLS: ClassVar[type[Any] | None] = None
    MODEL_PATH_OPTION: ClassVar[str] = "repo_root"
    MEMORY_RECORD_TYPE: ClassVar[str] = "runtime_result"
    FRAMEWORK_LOADING_OPTION_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "adapter",
            "adapter_target",
            "acquisition_root",
            "hf_models_root",
            "manifest_path",
            "model_adapter",
            "model_id",
            "pipeline_target",
            "profile_id",
            "profile_path",
        }
    )
    RESERVED_SYNTHESIS_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"prompt", "images", "video", "actions", "ref_image_path", "extra_inputs"}
    )

    def __init__(
        self,
        model_id: str | None = None,
        operators: Any = None,
        operator: Any = None,
        synthesis_model: Any = None,
        memory_module: Any = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.model_id = model_id or self.MODEL_ID
        self.synthesis_model = synthesis_model
        self.operator = operators if operators is not None else operator
        self.operators = self.operator
        self.memory_module = memory_module if memory_module is not None else self._create_memory_module_if_available()
        self.device = device
        self.options = dict(kwargs)

    @classmethod
    def _uses_component_contract(cls) -> bool:
        """Determine whether the pipeline class implements the component contract."""
        return all(
            getattr(cls, attribute_name, None) is not None
            for attribute_name in ("OPERATOR_CLS", "MEMORY_CLS", "SYNTHESIS_CLS")
        )

    @classmethod
    def _component(cls, attribute_name: str) -> type[Any]:
        """Retrieve a pipeline component class by attribute name."""
        component = getattr(cls, attribute_name, None)
        if component is None:
            raise RuntimeError(f"{cls.__name__} must define {attribute_name}.")
        return component

    @classmethod
    def _create_memory_module_if_available(cls) -> Any:
        """Instantiate and return the memory module if specified."""
        if getattr(cls, "MEMORY_CLS", None) is None:
            return None
        return cls._component("MEMORY_CLS")(model_id=cls.MODEL_ID)

    @classmethod
    def _runtime_options(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct the combined runtime options dictionary."""
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options[cls.MODEL_PATH_OPTION] = str(model_path)
        options.update(required_components or {})
        options.update(kwargs or {})
        return options

    @classmethod
    def _resolve_model_id(cls, options: dict[str, Any], model_id: str | None = None) -> str:
        """Resolve the target model or profile identifier."""
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID or "")
        if not resolved_model_id:
            raise ValueError(f"{cls.__name__}.from_pretrained requires model_id/profile_id.")
        return resolved_model_id

    @classmethod
    def _strip_framework_loading_options(cls, options: dict[str, Any]) -> dict[str, Any]:
        """Remove WorldFoundry loader metadata before calling model internals."""
        for key in cls.FRAMEWORK_LOADING_OPTION_KEYS:
            options.pop(key, None)
        return options

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "PipelineABC":
        """Create a pipeline with the unified loading signature.

        This default is a compatibility implementation for lightweight or test
        pipelines. Production pipelines are expected to override it when they
        need to load model components.
        """
        if cls._uses_component_contract():
            options = cls._runtime_options(model_path, required_components, kwargs)
            resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
            synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
                {**options, "model_id": resolved_model_id},
                device=device,
            )
            profile = getattr(synthesis_model, "profile", None)
            operator = cls._component("OPERATOR_CLS")(input_schema=dict(getattr(profile, "input_schema", {}) or {}))
            return cls(
                model_id=resolved_model_id,
                operators=operator,
                synthesis_model=synthesis_model,
                memory_module=cls._component("MEMORY_CLS")(model_id=resolved_model_id),
                device=device,
            )

        init_kwargs = dict(kwargs)
        if model_path is not None:
            init_kwargs["model_path"] = model_path
        if required_components is not None:
            init_kwargs["required_components"] = required_components
        if device is not None:
            init_kwargs["device"] = device
        if model_id is not None:
            init_kwargs["model_id"] = model_id

        signature = inspect.signature(cls)
        parameters = signature.parameters
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return cls(**init_kwargs)

        supported_kwargs = {
            key: value
            for key, value in init_kwargs.items()
            if key in parameters
            and parameters[key].kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return cls(**supported_kwargs)

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Normalize inputs before inference.

        Pipelines with operators should override this. The fallback preserves
        all caller data in a predictable shape for simple passthrough pipelines.
        """
        if self._uses_component_contract() and getattr(self, "operator", None) is not None:
            return self._process_component_inputs(*args, **kwargs)
        if args and kwargs:
            return {"args": args, **kwargs}
        if kwargs:
            return kwargs
        if len(args) == 1:
            return args[0]
        return args

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline by delegating to :meth:`process` by default."""
        if self._uses_component_contract() and getattr(self, "synthesis_model", None) is not None:
            return self._call_component_pipeline(*args, **kwargs)
        return self.process(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Yield pipeline outputs using the same call semantics as ``__call__``."""
        if self._uses_component_contract() and getattr(self, "synthesis_model", None) is not None:
            if kwargs.get("images") is None and kwargs.get("video") is None:
                previous = self.memory_module.select() if self.memory_module is not None else None
                if isinstance(previous, dict):
                    kwargs["images"] = previous.get("artifact_path")
            return self(*args, **kwargs)
        return self._stream_fallback(*args, **kwargs)

    def _stream_fallback(self, *args: Any, **kwargs: Any) -> Generator[Any, None, None]:
        """Stream fallback for PipelineABC."""
        result = self(*args, **kwargs)
        if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray, dict)):
            yield from result
        else:
            yield result

    def _process_component_inputs(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] | Any | None = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process component inputs for PipelineABC."""
        if self.operator is None:
            raise RuntimeError(f"{self.__class__.__name__} operator is not initialized.")
        self.operator.get_interaction(interactions)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return {
            **self.operator.process_prompt(prompt, **kwargs),
            **self.operator.process_perception(
                images=images,
                video=video,
                ref_image_path=ref_image_path,
                **kwargs,
            ),
            **interaction,
        }

    def _call_component_pipeline(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] | Any | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call component pipeline for PipelineABC."""
        if self.synthesis_model is None:
            raise RuntimeError(f"{self.__class__.__name__} synthesis_model is not initialized.")
        ref_image_path = kwargs.pop("ref_image_path", None)
        explicit_operator_kwargs = kwargs.pop("operator_kwargs", {})
        if explicit_operator_kwargs is None:
            explicit_operator_kwargs = {}
        if not isinstance(explicit_operator_kwargs, Mapping):
            raise TypeError("operator_kwargs must be a mapping")
        # Workspace and the CLI expose model controls as normal call kwargs.
        # Operators need those values too so state, camera keys, variants and
        # seeds are represented in the normalized observation. Explicit
        # operator kwargs retain precedence for advanced callers.
        operator_inputs = {**kwargs, **dict(explicit_operator_kwargs)}
        processed = self.process(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            ref_image_path=ref_image_path,
            **operator_inputs,
        )
        # The synthesis dispatch below reads these keys unconditionally; a
        # custom operator/process override that drops one used to surface as a
        # bare KeyError far from the offending component (PL-04).
        required_keys = ("prompt", "images", "video", "actions")
        if not isinstance(processed, Mapping):
            raise TypeError(
                f"{type(self).__name__}.process() must return a mapping with keys "
                f"{required_keys}, got {type(processed).__name__}; check the "
                f"operator ({type(self.operator).__name__ if self.operator is not None else None}) "
                "process_prompt/process_perception/process_interaction outputs"
            )
        missing = [key for key in required_keys if key not in processed]
        if missing:
            raise TypeError(
                f"{type(self).__name__}.process() output is missing required keys {missing}; "
                f"the component contract requires {required_keys}. Check the operator "
                f"({type(self.operator).__name__ if self.operator is not None else None}) "
                "process_prompt/process_perception/process_interaction outputs"
            )
        model_specific = {
            key: value for key, value in processed.items() if key not in self.RESERVED_SYNTHESIS_KEYS
        }
        synthesis_kwargs = {**processed.get("extra_inputs", {}), **model_specific, **kwargs}
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            images=processed["images"],
            video=processed["video"],
            interactions=processed["actions"],
            output_path=output_path,
            fps=fps,
            **synthesis_kwargs,
        )
        if self.memory_module is not None:
            self.memory_module.record(result, metadata={"type": self.MEMORY_RECORD_TYPE, "model_id": self.model_id})
        if return_dict:
            return result
        return result.get("artifact_path") or result

    def get_operator(self) -> Any:
        """Get operator for PipelineABC."""
        if getattr(self, "operator", None) is None:
            raise RuntimeError(f"{self.__class__.__name__} operator is not initialized.")
        return self.operator

    def get_synthesis_model(self) -> Any:
        """Get synthesis model for PipelineABC."""
        if getattr(self, "synthesis_model", None) is None:
            raise RuntimeError(f"{self.__class__.__name__} synthesis_model is not initialized.")
        return self.synthesis_model
