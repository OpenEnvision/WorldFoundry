"""
This module provides the InspatioWorldSynthesis class, which acts as a thin adapter over the
InspatioWorldRuntime for easy integration into evaluation frameworks. It abstracts the underlying
runtime's initialization and core functionalities like `plan` and `predict`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from worldfoundry.synthesis.visual_generation.inspatio_world.worldfoundry_runtime import (
    DEFAULT_CHECKPOINT_REPO,
    DEFAULT_DA3_MODEL_REPO,
    DEFAULT_FLORENCE_MODEL_REPO,
    DEFAULT_WAN_MODEL_REPO,
    InspatioWorldRuntime,
)
from worldfoundry.synthesis.base_synthesis import BaseSynthesis


class InspatioWorldSynthesis(BaseSynthesis):
    """Thin synthesis adapter over the in-tree InSpatio-World runtime."""

    def __init__(
        self,
        repo_root: Optional[str] = None,
        checkpoint_source: str = DEFAULT_CHECKPOINT_REPO,
        wan_model_source: str = DEFAULT_WAN_MODEL_REPO,
        da3_model_source: str = DEFAULT_DA3_MODEL_REPO,
        florence_model_source: str = DEFAULT_FLORENCE_MODEL_REPO,
        device: str = "cuda",
        defaults: Optional[Dict[str, Any]] = None,
        runtime: Optional[InspatioWorldRuntime] = None,
    ) -> None:
        """
        Initializes the InspatioWorldSynthesis adapter.

        It creates or uses an existing `InspatioWorldRuntime` instance to manage the underlying
        InSpatio-World environment and models.

        Args:
            repo_root (Optional[str]): Path to the root of the InSpatio-World repository. If None, uses the bundled repo.
            checkpoint_source (str): Source for the main InSpatio-World model checkpoint.
                                     Defaults to `DEFAULT_CHECKPOINT_REPO`.
            wan_model_source (str): Source for the WAN model checkpoint. Defaults to `DEFAULT_WAN_MODEL_REPO`.
            da3_model_source (str): Source for the DA3 model checkpoint. Defaults to `DEFAULT_DA3_MODEL_REPO`.
            florence_model_source (str): Source for the Florence model checkpoint.
                                         Defaults to `DEFAULT_FLORENCE_MODEL_REPO`.
            device (str): The device to load models onto (e.g., "cuda", "cpu"). Defaults to "cuda".
            defaults (Optional[Dict[str, Any]]): A dictionary of default settings to pass to the runtime.
            runtime (Optional[InspatioWorldRuntime]): An optional pre-initialized `InspatioWorldRuntime` instance.
                                                     If provided, other initialization arguments (repo_root,
                                                     checkpoint_source, etc.) are ignored.
        """
        super().__init__()
        # Initialize the underlying InspatioWorldRuntime. If a runtime is provided, use it;
        # otherwise, create a new one with the given configuration.
        self.runtime = runtime or InspatioWorldRuntime(
            repo_root=repo_root,
            checkpoint_source=checkpoint_source,
            wan_model_source=wan_model_source,
            da3_model_source=da3_model_source,
            florence_model_source=florence_model_source,
            device=device,
            defaults=defaults,
        )
        self._sync_public_attrs()

    def _sync_public_attrs(self) -> None:
        """
        Synchronizes public attributes of this synthesis adapter with the attributes of the underlying runtime.

        This ensures that properties like `repo_root` or `device` are accessible directly from the synthesis instance.
        """
        self.repo_root = self.runtime.repo_root
        self.checkpoint_source = self.runtime.checkpoint_source
        self.wan_model_source = self.runtime.wan_model_source
        self.da3_model_source = self.runtime.da3_model_source
        self.florence_model_source = self.runtime.florence_model_source
        self.device = self.runtime.device
        self.defaults = self.runtime.defaults

    @classmethod
    def bundled_repo_root(cls) -> str:
        """
        Returns the path to the bundled InSpatio-World repository root.

        Returns:
            str: The absolute path to the bundled repository root.
        """
        return InspatioWorldRuntime.bundled_repo_root()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = DEFAULT_CHECKPOINT_REPO,
        args=None,
        device: Optional[str] = None,
        wan_model_path: str = DEFAULT_WAN_MODEL_REPO,
        da3_model_path: str = DEFAULT_DA3_MODEL_REPO,
        florence_model_path: str = DEFAULT_FLORENCE_MODEL_REPO,
        config_path: Optional[str] = None,
        tae_checkpoint_path: Optional[str] = None,
        default_traj_txt_path: Optional[str] = None,
        **kwargs,
    ) -> "InspatioWorldSynthesis":
        """
        Initializes the `InspatioWorldSynthesis` adapter by loading a pretrained InSpatio-World model.

        This is a convenience constructor that wraps the `InspatioWorldRuntime.from_pretrained` method,
        allowing for easy model loading.

        Args:
            pretrained_model_path (str): Path or identifier for the primary pretrained InSpatio-World model.
                                         Defaults to `DEFAULT_CHECKPOINT_REPO`.
            args (Any): Additional arguments to pass to the underlying runtime's `from_pretrained` method.
            device (Optional[str]): The device to load models onto (e.g., "cuda", "cpu").
                                    If None, runtime default is used.
            wan_model_path (str): Path or identifier for the pretrained WAN model. Defaults to `DEFAULT_WAN_MODEL_REPO`.
            da3_model_path (str): Path or identifier for the pretrained DA3 model. Defaults to `DEFAULT_DA3_MODEL_REPO`.
            florence_model_path (str): Path or identifier for the pretrained Florence model.
                                       Defaults to `DEFAULT_FLORENCE_MODEL_REPO`.
            config_path (Optional[str]): Path to a configuration file.
            tae_checkpoint_path (Optional[str]): Path to a TAE (Trajectory-Aware Embedding) model checkpoint.
            default_traj_txt_path (Optional[str]): Path to a default trajectory text file.
            **kwargs: Arbitrary keyword arguments passed directly to the underlying
                      `InspatioWorldRuntime.from_pretrained`.

        Returns:
            InspatioWorldSynthesis: An initialized instance of `InspatioWorldSynthesis`.
        """
        # Delegate the actual model loading to the InspatioWorldRuntime's from_pretrained method,
        # then wrap the resulting runtime in a new InspatioWorldSynthesis instance.
        runtime = InspatioWorldRuntime.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            args=args,
            device=device,
            wan_model_path=wan_model_path,
            da3_model_path=da3_model_path,
            florence_model_path=florence_model_path,
            config_path=config_path,
            tae_checkpoint_path=tae_checkpoint_path,
            default_traj_txt_path=default_traj_txt_path,
            **kwargs,
        )
        return cls(runtime=runtime)

    def plan(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes the planning phase using the underlying InSpatio-World runtime.

        All arguments are passed directly to the runtime's `plan` method.

        Args:
            *args: Positional arguments to pass to the runtime's plan method.
            **kwargs: Keyword arguments to pass to the runtime's plan method.

        Returns:
            Dict[str, Any]: The result of the planning operation, typically a dictionary containing plan details.
        """
        return self.runtime.plan(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Executes the prediction phase using the underlying InSpatio-World runtime.

        All arguments are passed directly to the runtime's `predict` method.

        Args:
            *args: Positional arguments to pass to the runtime's predict method.
            **kwargs: Keyword arguments to pass to the runtime's predict method.

        Returns:
            Any: The result of the prediction operation. The exact type depends on the runtime's implementation.
        """
        return self.runtime.predict(*args, **kwargs)