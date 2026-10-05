"""Module for defining the EchoInfinitySynthesis integration within the runtime video synthesis framework.

This module provides the `EchoInfinitySynthesis` class, which serves as a concrete
implementation of `RuntimeVideoSynthesis` specifically tailored for the
EchoInfinity text-to-video generation model. It handles the dynamic loading
of the EchoInfinity runtime and specifies its configuration parameters.
"""
from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class _EchoInfinityRuntime:
    """A proxy class for dynamically importing and instantiating the EchoInfinity runtime.

    This class intercepts the instantiation call to `_EchoInfinityRuntime` and
    instead imports the actual `EchoInfinity` class from `worldfoundry.synthesis`
    at runtime, then instantiates and returns it. This lazy loading prevents
    circular dependencies or unnecessary imports until the runtime is actually needed.
    """

    def __new__(cls, *args, **kwargs):
        """Dynamically imports and instantiates the EchoInfinity runtime.

        This method is called when an instance of `_EchoInfinityRuntime` is
        requested. It performs the actual import of the `EchoInfinity` class
        and then delegates the instantiation to it, returning an instance
        of the real `EchoInfinity` object.
        """
        # Dynamically import the actual EchoInfinity runtime class to avoid
        # direct import at module load time, enabling lazy loading.
        from worldfoundry.synthesis.visual_generation.echo_infinity.worldfoundry_runtime import EchoInfinity

        # Return an instance of the actual EchoInfinity runtime.
        signature = inspect.signature(EchoInfinity)
        parameters = signature.parameters
        if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return EchoInfinity(*args, **kwargs)


class EchoInfinitySynthesis(RuntimeVideoSynthesis):
    """Integrates the EchoInfinity text-to-video model into the runtime synthesis framework.

    This class extends `RuntimeVideoSynthesis` to provide specific configuration
    and runtime details for the EchoInfinity model. It defines metadata such as
    model name, generation type, and paths to configuration files.
    """

    MODEL_NAME = "echo-infinity"
    GENERATION_TYPE = "t2v"
    # Points to the proxy class that handles the dynamic loading of the actual EchoInfinity runtime.
    RUNTIME_CLS = _EchoInfinityRuntime
    PRIMARY_PATH_KEY = "generator_ckpt"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/echo_infinity/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME

    def _prediction_runtime_overrides(self, kwargs: Mapping[str, Any], *, fps: Optional[int]):
        overrides = super()._prediction_runtime_overrides(kwargs, fps=None)
        overrides.pop("fps", None)
        # The generic seed alias prefers ``base_seed`` for proxy runtimes with
        # **kwargs, but EchoInfinity only accepts ``seed``. Preserve the model's
        # own frame-count option as well; it is not a generic video alias.
        overrides.pop("base_seed", None)
        if kwargs.get("seed") is not None:
            overrides["seed"] = int(kwargs["seed"])
        if kwargs.get("num_output_frames") is not None:
            overrides["num_output_frames"] = int(kwargs["num_output_frames"])
        return overrides


__all__ = ["EchoInfinitySynthesis"]
