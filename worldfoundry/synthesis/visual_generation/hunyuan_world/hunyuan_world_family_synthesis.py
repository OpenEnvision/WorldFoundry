"""This module defines synthesis classes for various models within the Hunyuan World family.

It manages their integration status, planning, and execution capabilities
within an in-tree backend. It includes classes for tracking integration status,
handling errors, and providing plan-only or fully integrated synthesis surfaces
depending on the model's in-tree readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...base_synthesis import BaseSynthesis


@dataclass(frozen=True)
class HunyuanWorldIntegrationStatus:
    """Represents the integration status of a Hunyuan World family model.

    This dataclass provides a structured way to store metadata about a model's
    readiness for in-tree use, including its ID, display name, current status,
    source repository, evidence supporting its status, and recommendations for
    further integration.
    """

    model_id: str
    display_name: str
    status: str
    source_repo: str
    evidence: tuple[str, ...]
    recommendation: str


class HunyuanWorldIntegrationError(RuntimeError):
    """Raised when a Hunyuan World family model is not a runnable in-tree backend."""


class HunyuanWorldPlanSynthesis(BaseSynthesis):
    """Provides a plan-only synthesis surface for Hunyuan World models.

    This class serves as a base for Hunyuan World models that are not yet fully
    runnable as in-tree backends. It primarily exposes the integration status
    and planning information, rejecting actual inference or API calls.
    """

    STATUS = HunyuanWorldIntegrationStatus(
        model_id="hunyuanworld-family",
        display_name="Hunyuan World Family",
        status="blocked",
        source_repo="",
        evidence=(),
        recommendation="Do not register as runnable until official code is ported in-tree.",
    )
    IN_TREE_BACKEND = False

    def __init__(self, status: HunyuanWorldIntegrationStatus | None = None) -> None:
        """Initializes the HunyuanWorldPlanSynthesis with a specific integration status.

        Args:
            status: An optional HunyuanWorldIntegrationStatus object. If None,
                    the default class-level STATUS is used.
        """
        super().__init__()
        # Assign the provided status or fall back to the class-level default status.
        self.status = status or self.STATUS
        self.model_id = self.status.model_id
        self.model_name = self.status.display_name

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> HunyuanWorldPlanSynthesis:
        """Returns a plan-only synthesis instance without actual executable synthesis.

        This method is provided for API compatibility but ignores all input
        parameters as no actual model loading or execution setup is performed.

        Args:
            pretrained_model_path: Ignored checkpoint or repository path.
            args: Ignored runtime arguments.
            device: Ignored execution device.
            kwargs: Ignored additional keyword arguments.

        Returns:
            An instance of HunyuanWorldPlanSynthesis.
        """
        # These parameters are intentionally ignored for plan-only synthesis.
        del pretrained_model_path, args, device, kwargs
        return cls()

    def plan(self) -> dict[str, Any]:
        """Returns the integration status and provenance evidence for the model.

        This method provides detailed metadata about the model's in-tree
        integration status, including recommendations and reasons.

        Returns:
            A dictionary containing the model's integration status, display name,
            source repository, evidence, recommendation, and whether it's an
            in-tree backend.
        """
        return {
            "model_id": self.status.model_id,
            "display_name": self.status.display_name,
            "status": self.status.status,
            "source_repo": self.status.source_repo,
            "evidence": list(self.status.evidence),  # Convert tuple to list for potential mutability in consumers.
            "recommendation": self.status.recommendation,
            "in_tree_backend": self.IN_TREE_BACKEND,
        }

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Rejects any inference requests for non-runnable Hunyuan World plan surfaces.

        This method raises an error because the model represented by this
        synthesis class is not intended for actual inference execution.

        Args:
            args: Ignored positional inference inputs.
            kwargs: Ignored keyword inference inputs.

        Raises:
            HunyuanWorldIntegrationError: Always raised, indicating that the
                                         model is not runnable for prediction.
        """
        # These parameters are intentionally ignored as prediction is not supported.
        del args, kwargs
        raise HunyuanWorldIntegrationError(
            f"{self.status.model_id} is {self.status.status}: {self.status.recommendation}"
        )

    def api_init(self, api_key: str, endpoint: str) -> None:
        """Rejects API initialization for this module.

        This module is designed to track in-tree integration only and does not
        support external API-based synthesis.

        Args:
            api_key: Ignored API credential.
            endpoint: Ignored API endpoint.

        Raises:
            HunyuanWorldIntegrationError: Always raised, indicating that no
                                         in-tree API synthesis backend exists.
        """
        # These parameters are intentionally ignored as API initialization is not supported.
        del api_key, endpoint
        raise HunyuanWorldIntegrationError(f"{self.status.model_id} has no in-tree API synthesis backend.")


class HunyuanWorld1Synthesis(HunyuanWorldPlanSynthesis):
    """Represents the plan-only synthesis for the HunyuanWorld 1.0 model.

    This class extends `HunyuanWorldPlanSynthesis` to specifically detail the
    integration status of HunyuanWorld 1.0, which is currently blocked due to
    license and third-party dependency requirements.
    """

    STATUS = HunyuanWorldIntegrationStatus(
        model_id="hunyuanworld-1",
        display_name="HunyuanWorld 1.0",
        status="blocked-license-and-deps",
        source_repo="tencent/HunyuanWorld-1",
        evidence=(
            "Official setup requires cloning Real-ESRGAN, ZIM, and draco.",
            "Official license is Tencent HunyuanWorld-1.0 Community License with territory restrictions.",
            "Runtime imports include basicsr, realesrgan, zim_anything, MoGe, open3d, and draco-dependent export paths.",
        ),
        recommendation=(
            "Keep metadata-only or plan-only until third-party runtime dependencies and license obligations "
            "are reviewed and vendored under an approved in-tree layout."
        ),
    )


class HunyuanWorldMirrorSynthesis(HunyuanWorldPlanSynthesis):
    """Provides an integrated synthesis surface for the HunyuanWorld-Mirror model.

    Unlike its base class, this model is fully integrated in-tree and provides
    executable synthesis capabilities through an internal pipeline.
    """

    STATUS = HunyuanWorldIntegrationStatus(
        model_id="hunyuanworld-mirror",
        display_name="HunyuanWorld-Mirror",
        status="integrated",
        source_repo="tencent/HunyuanWorld-Mirror",
        evidence=(
            "HunyuanWorld-Mirror inference is resolved through WorldFoundry's in-tree runtime.",
            "WorldFoundry wraps the Mirror model through HunyuanMirrorPipeline for framework inference.",
            "Third-party runtime dependencies are expected to be provided by the configured runtime environment.",
        ),
        recommendation=(
            "Use HunyuanMirrorPipeline or this synthesis wrapper for image-sequence 3D reconstruction inference."
        ),
    )
    IN_TREE_BACKEND = True

    def __init__(self, pipeline: Any | None = None, status: HunyuanWorldIntegrationStatus | None = None) -> None:
        """Initializes the HunyuanWorldMirrorSynthesis.

        Args:
            pipeline: An optional pre-initialized pipeline object for inference.
                      If not provided, it will be initialized by `from_pretrained`.
            status: An optional HunyuanWorldIntegrationStatus object. If None,
                    the default class-level STATUS is used.
        """
        # Call the base class constructor, passing the status.
        super().__init__(status=status or self.STATUS)
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = "tencent/HunyuanWorld-Mirror",
        args: Any = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> HunyuanWorldMirrorSynthesis:
        """Loads and initializes the HunyuanWorld-Mirror model pipeline.

        This method retrieves the HunyuanMirrorPipeline from `worldfoundry` and
        configures it with the provided model path and device.

        Args:
            pretrained_model_path: The path or identifier for the pretrained model.
                                   Defaults to "tencent/HunyuanWorld-Mirror".
            args: Ignored runtime arguments.
            device: The device to load the model onto (e.g., "cuda", "cpu").
                    Defaults to "cuda" if not specified.
            kwargs: Additional keyword arguments passed to `HunyuanMirrorPipeline.from_pretrained`.

        Returns:
            An instance of HunyuanWorldMirrorSynthesis with an initialized pipeline.
        """
        # This parameter is intentionally ignored.
        del args
        # Dynamically import the HunyuanMirrorPipeline to avoid circular dependencies
        # or unnecessary imports if this class is not used.
        from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_mirror import HunyuanMirrorPipeline

        # Initialize the HunyuanMirrorPipeline from the specified path and device.
        pipeline = HunyuanMirrorPipeline.from_pretrained(
            model_path=pretrained_model_path,
            device=device or "cuda",
            **kwargs,
        )
        return cls(pipeline=pipeline)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Performs inference using the initialized HunyuanMirrorPipeline.

        Args:
            args: Positional arguments passed directly to the underlying pipeline's call method.
            kwargs: Keyword arguments passed directly to the underlying pipeline's call method.
                    If 'data' is present and a dict, its contents are merged into kwargs.

        Returns:
            The output from the HunyuanMirrorPipeline's inference.

        Raises:
            HunyuanWorldIntegrationError: If the pipeline has not been initialized.
        """
        if self.pipeline is None:
            raise HunyuanWorldIntegrationError("hunyuanworld-mirror synthesis was not initialized with a pipeline.")
        # Extract 'data' if present and merge its contents into kwargs for pipeline compatibility.
        data = kwargs.pop("data", None)
        if isinstance(data, dict):
            kwargs = {**data, **kwargs}
        output_path = kwargs.get("output_path")
        run_dir = None
        if output_path:
            requested = Path(str(output_path)).expanduser()
            run_dir = (
                requested.with_suffix("")
                if requested.suffix.lower() in {".mp4", ".mov", ".webm", ".ply", ".zip"}
                else requested
            )
            kwargs["output_path"] = str(run_dir)

        results = self.pipeline(*args, **kwargs)
        saved = self.pipeline.save_results(
            results,
            save_pointmap=bool(kwargs.get("save_pointmap", True)),
            save_depth=bool(kwargs.get("save_depth", True)),
            save_normal=bool(kwargs.get("save_normal", True)),
            save_gs=bool(kwargs.get("save_gs", True)),
            save_rendered=bool(kwargs.get("save_rendered", True)),
            save_colmap=bool(kwargs.get("save_colmap", True)),
        )
        run_dir = Path(str(run_dir or self.pipeline.output_path)).expanduser()
        preview_image = None
        for candidate in (
            run_dir / "images_resized" / "image_0001.png",
            run_dir / "images" / "image_0001.png",
            run_dir / "depth" / "depth_0000.png",
            run_dir / "normal" / "normal_0000.png",
        ):
            if candidate.is_file():
                preview_image = str(candidate)
                break
        preview_video = saved.get("rendered_video_path")
        preview_model = saved.get("gaussians_path") or saved.get("pointmap_path")
        return {
            "status": "succeeded",
            "runtime": "hunyuanworld-mirror",
            "artifact_kind": "generated_3d_asset",
            "artifact_path": str(run_dir),
            "run_dir": str(run_dir),
            "preview_video": preview_video
            if isinstance(preview_video, str) and Path(preview_video).is_file()
            else None,
            "preview_image": preview_image,
            "model_path": preview_model if isinstance(preview_model, str) and Path(preview_model).is_file() else None,
            "metadata": {
                "saved_outputs": saved,
                "image_count": int(results.get("S", 0)) if isinstance(results, dict) else 0,
            },
        }

    def api_init(self, api_key: str, endpoint: str) -> None:
        """Handles API initialization for HunyuanWorld-Mirror.

        This model is primarily an in-tree backend, and this method currently
        does nothing beyond discarding the parameters, as it doesn't represent
        an external API.

        Args:
            api_key: Ignored API credential.
            endpoint: Ignored API endpoint.
        """
        # These parameters are intentionally ignored as API initialization is not handled here.
        del api_key, endpoint
        return None


class HYWorld2Synthesis(HunyuanWorldPlanSynthesis):
    """Represents the component-only in-tree synthesis for HY-World 2.0.

    This class extends `HunyuanWorldPlanSynthesis` to record the integration
    status of HY-World 2.0, whose WorldMirror 2.0 reconstruction component runs
    in-tree. HY-Pano 2.0 and the multi-stage world generation are not integrated.
    """

    STATUS = HunyuanWorldIntegrationStatus(
        model_id="hy-world-2.0",
        display_name="HY-World 2.0",
        status="partial-in-tree-component-only",
        source_repo="tencent/HY-World-2.0",
        evidence=(
            "WorldFoundry wraps WorldMirror 2.0 reconstruction.",
            "Official worldgen requires five separate runners: traj_generate.py, traj_render.py, video_gen.py, gen_gs_data.py, and world_gs_trainer.py.",
            "Official worldgen also requires custom gsplat_maskgaussian/navmesh native extensions, a vLLM VLM service, and distributed GPU orchestration.",
        ),
        recommendation=("Use task='worldrecon'. HY-Pano 2.0 and world generation are not part of WorldFoundry."),
    )


__all__ = [
    "HYWorld2Synthesis",
    "HunyuanWorld1Synthesis",
    "HunyuanWorldIntegrationError",
    "HunyuanWorldIntegrationStatus",
    "HunyuanWorldMirrorSynthesis",
    "HunyuanWorldPlanSynthesis",
]
