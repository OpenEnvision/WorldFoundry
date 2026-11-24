"""
Provides a thin facade over the VMem base runtime for synthesis tasks.

This module defines the `VMemSynthesis` class, which acts as a wrapper around
the `VMemRuntime` from `worldfoundry.synthesis.visual_generation.vmem`.
It simplifies the initialization and interaction with the VMem runtime,
allowing users to seamlessly integrate VMem's visual generation capabilities
into a broader synthesis framework. It also exposes default repository paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...base_synthesis import BaseSynthesis

if TYPE_CHECKING:
    from worldfoundry.synthesis.visual_generation.vmem.worldfoundry_runtime import VMemRuntime


def _runtime_module():
    """
    Lazily imports and returns the worldfoundry_runtime module.

    This helper function prevents circular imports and ensures the runtime
    module is loaded only when needed, supporting dynamic loading patterns.
    """
    from worldfoundry.synthesis.visual_generation.vmem import worldfoundry_runtime

    return worldfoundry_runtime


def _runtime_cls():
    """
    Lazily retrieves and returns the VMemRuntime class.

    This helper function avoids direct import of VMemRuntime at the module
    level, further preventing potential circular dependencies and allowing
    for dynamic loading.
    """
    return _runtime_module().VMemRuntime


# Default repository paths for VMem models, fetched from the runtime module.
DEFAULT_VMEM_REPO = _runtime_module().DEFAULT_VMEM_REPO
DEFAULT_VMEM_SURFEL_REPO = _runtime_module().DEFAULT_VMEM_SURFEL_REPO


class VMemSynthesis(BaseSynthesis):
    """
    A facade class for the VMem base runtime, simplifying its usage in synthesis pipelines.

    VMemSynthesis wraps an instance of `VMemRuntime`, delegating most
    method calls (like `predict`) to the underlying runtime object. It
    provides convenient initialization methods, including a factory method
    `from_pretrained`, to easily set up the VMem system. This design
    decouples the synthesis interface from the specific VMem implementation.
    """

    def __init__(
        self,
        runtime_pipeline: Any = None,
        config: Any = None,
        *,
        transform_img_and_K: Any = None,
        get_default_intrinsics: Any = None,
        device: str = "cuda",
        weight_dtype: Any = None,
        step_size: float = 0.1,
        num_interpolation_frames: int = 4,
        runtime: VMemRuntime | None = None,
    ):
        """
        Initializes the VMemSynthesis facade.

        This constructor can either take an existing `VMemRuntime` instance
        or create a new one using the provided configuration and pipeline.

        Args:
            runtime_pipeline: The pipeline object required by `VMemRuntime` if
                              `runtime` is not provided. This is typically an
                              instance configuring the VMem execution flow.
            config: The configuration object required by `VMemRuntime` if
                    `runtime` is not provided. This specifies model parameters
                    and other runtime settings.
            transform_img_and_K: Optional callable for image and intrinsic
                                 matrix transformation, passed to `VMemRuntime`.
            get_default_intrinsics: Optional callable to get default intrinsics,
                                    passed to `VMemRuntime`.
            device: The device to run the model on (e.g., "cuda" or "cpu").
                    Passed to `VMemRuntime`.
            weight_dtype: Data type for model weights (e.g., torch.float16).
                          Passed to `VMemRuntime`.
            step_size: Step size for the synthesis process. Passed to `VMemRuntime`.
            num_interpolation_frames: Number of frames for interpolation.
                                      Passed to `VMemRuntime`.
            runtime: An optional pre-initialized `VMemRuntime` instance. If provided,
                     `runtime_pipeline` and `config` are ignored, and this instance
                     is used directly.

        Raises:
            ValueError: If `runtime` is None but `runtime_pipeline` or `config`
                        are not provided, as they are necessary to create a new
                        `VMemRuntime` instance.
        """
        super().__init__()
        if runtime is None:
            # If no runtime instance is provided, create one using the given pipeline and config.
            if runtime_pipeline is None or config is None:
                raise ValueError("runtime_pipeline and config are required when runtime is not provided.")
            runtime = _runtime_cls()(
                runtime_pipeline,
                config,
                transform_img_and_K=transform_img_and_K,
                get_default_intrinsics=get_default_intrinsics,
                device=device,
                # Conditionally pass 'weight_dtype' as a keyword argument if it's not None.
                **({} if weight_dtype is None else {"weight_dtype": weight_dtype}),
                step_size=step_size,
                num_interpolation_frames=num_interpolation_frames,
            )
        self.runtime = runtime

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying `VMemRuntime` instance.

        This method allows `VMemSynthesis` to act as a proxy, passing calls
        to methods or accesses to attributes that are not explicitly defined
        on `VMemSynthesis` directly to `self.runtime`. This enables seamless
        interaction with the underlying VMem system's functionality.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The attribute or method from the `self.runtime` object.

        Raises:
            AttributeError: If `name` is 'runtime' (to prevent infinite recursion
                            when `getattr` is called on `self.runtime` itself)
                            or if the attribute does not exist on `self.runtime`.
        """
        if name == "runtime":
            # Prevent infinite recursion if 'runtime' attribute itself is accessed via __getattr__.
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | None = None,
        args: Any = None,
        device: str | None = None,
        weight_dtype: Any = None,
        config_path: str | None = None,
        surfel_model_path: str | None = None,
        step_size: float = 0.1,
        num_interpolation_frames: int = 4,
        runtime_root: str | None = None,
        visualization_dir: str | None = None,
        **kwargs: Any,
    ) -> VMemSynthesis:
        """
        Creates a VMemSynthesis instance by loading a pretrained VMemRuntime model.

        This class method simplifies the process of initializing `VMemSynthesis`
        by directly loading a pretrained model configuration. It forwards
        all relevant arguments to `VMemRuntime.from_pretrained` and then
        wraps the created runtime instance.

        Args:
            pretrained_model_path: Path to the pretrained VMem model.
                                   Passed to `VMemRuntime.from_pretrained`.
            args: Additional arguments passed to the underlying runtime's
                  `from_pretrained` method.
            device: The device to load the model on (e.g., "cuda", "cpu").
            weight_dtype: Data type for model weights (e.g., torch.float16).
            config_path: Path to the configuration file for the runtime.
            surfel_model_path: Path to the surfel model.
            step_size: Step size for the synthesis process.
            num_interpolation_frames: Number of frames for interpolation.
            runtime_root: Data root searched for configs/inference/inference.yaml; code stays in-tree.
            visualization_dir: Directory for saving visualizations.
            **kwargs: Arbitrary keyword arguments passed directly to
                      `VMemRuntime.from_pretrained`.

        Returns:
            An instance of `VMemSynthesis` with the loaded runtime.
        """
        # Collect all arguments destined for VMemRuntime.from_pretrained into a single dictionary.
        runtime_kwargs = {
            "pretrained_model_path": pretrained_model_path,
            "args": args,
            "device": device,
            "config_path": config_path,
            "surfel_model_path": surfel_model_path,
            "step_size": step_size,
            "num_interpolation_frames": num_interpolation_frames,
            "runtime_root": runtime_root,
            "visualization_dir": visualization_dir,
            **kwargs,
        }
        # Add weight_dtype to runtime_kwargs only if it's explicitly provided, as VMemRuntime
        # might have a default or handle its absence differently.
        if weight_dtype is not None:
            runtime_kwargs["weight_dtype"] = weight_dtype
        runtime = _runtime_cls().from_pretrained(**runtime_kwargs)
        return cls(runtime=runtime)

    def predict(self, *args: Any, **kwargs: Any):
        """
        Delegates the prediction call to the underlying `VMemRuntime` instance.

        This method acts as a pass-through for the primary prediction functionality
        of the VMem system, allowing `VMemSynthesis` to be used directly
        for inference.

        Args:
            *args: Positional arguments to pass to `self.runtime.predict`.
            **kwargs: Keyword arguments to pass to `self.runtime.predict`.

        Returns:
            The result of `self.runtime.predict(*args, **kwargs)`.
        """
        return self.runtime.predict(*args, **kwargs)


__all__ = ["DEFAULT_VMEM_REPO", "DEFAULT_VMEM_SURFEL_REPO", "VMemSynthesis"]
