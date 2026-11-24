"""Forcing visual generation pipeline module."""

from __future__ import annotations

from typing import Any

from ...operators.forcing_operator import RollingForcingOperator
from ...synthesis.visual_generation.rolling_forcing import RollingForcingSynthesis
from ..pipeline_utils import PipelineABC


class _BaseForcingPipeline(PipelineABC):
    """Shared pipeline assembly for forcing-family official runners."""

    MODEL_ID = ""
    MODEL_PATH_OPTION = "checkpoint_path"
    OPERATOR_CLS = RollingForcingOperator
    SYNTHESIS_CLS = RollingForcingSynthesis
    MEMORY_TARGET = None

    @classmethod
    def _uses_component_contract(cls) -> bool:
        """Determine whether the pipeline class implements the component contract."""
        return True

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> _BaseForcingPipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        if resolved_model_id != cls.MODEL_ID:
            raise ValueError(f"{cls.__name__} cannot load {resolved_model_id!r}.")
        synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        return cls(
            model_id=resolved_model_id,
            operators=cls._component("OPERATOR_CLS")(),
            synthesis_model=synthesis_model,
            memory_module=None,
            device=device,
        )

    def __call__(
        self,
        *args: Any,
        num_frames: int | None = None,
        num_output_frames: int | None = None,
        seed: int | None = None,
        num_samples: int | None = None,
        use_ema: bool | None = None,
        save_with_index: bool | None = None,
        report_timing: bool | None = None,
        extended_prompt: str | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Map shared Workspace controls to the official forcing runners."""

        if num_frames is not None:
            kwargs["num_output_frames"] = int(num_frames)
        if num_output_frames is not None:
            kwargs["num_output_frames"] = int(num_output_frames)
        if seed is not None:
            kwargs["seed"] = int(seed)
        if num_samples is not None:
            kwargs["num_samples"] = int(num_samples)
        if use_ema is not None:
            kwargs["use_ema"] = bool(use_ema)
        if save_with_index is not None:
            kwargs["save_with_index"] = bool(save_with_index)
        if report_timing is not None:
            kwargs["report_timing"] = bool(report_timing)
        if extended_prompt is not None:
            kwargs["extended_prompt"] = str(extended_prompt)
        return super().__call__(*args, return_dict=return_dict, **kwargs)


class RollingForcingPipeline(_BaseForcingPipeline):
    """WorldFoundry pipeline for RollingForcing long-video generation."""

    MODEL_ID = "rolling-forcing"
    OPERATOR_CLS = RollingForcingOperator
    SYNTHESIS_CLS = RollingForcingSynthesis


__all__ = ["RollingForcingPipeline"]
