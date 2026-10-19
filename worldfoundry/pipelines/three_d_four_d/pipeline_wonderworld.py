# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Public native WonderWorld single-image to layered Gaussian scene pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from worldfoundry.runtime.wonderworld_inference_spec import WONDERWORLD_NEGATIVE_PROMPT

from ..pipeline_utils import PipelineABC

if TYPE_CHECKING:
    from worldfoundry.core.contracts import PipelineInvocation


class WonderWorldPipeline(PipelineABC):
    """Generate an isolated 512x512 seed world and optional novel viewpoints."""

    MODEL_ID = "wonderworld"
    MODEL_PATH_OPTION = "checkpoint_root"
    generation_type = "three_d_four_d"

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> WonderWorldPipeline:
        """Stage component paths without loading weights or an external repo."""
        from worldfoundry.operators.wonderworld_operator import WonderWorldOperator
        from worldfoundry.synthesis.visual_generation.wonderworld.runtime import WonderWorldSynthesis

        del model_id
        options = cls._strip_framework_loading_options(cls._runtime_options(model_path, required_components, kwargs))
        synthesis = WonderWorldSynthesis.from_pretrained(options, device=device)
        return cls(model_id="wonderworld", operator=WonderWorldOperator(), synthesis_model=synthesis, device=device)

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        *,
        input_path: str | Path | None = None,
        camera_path: Any = None,
        interactions: Any = None,
        video: Any = None,
        output_path: str | Path | None = None,
        run_dir: str | Path | None = None,
        seed: int = 42,
        depth_steps: int = 30,
        inpaint_steps: int = 50,
        normal_steps: int = 10,
        refinement_steps: int = 100,
        focal_length: float = 500,
        gen_layer: bool = True,
        negative_prompt: str = WONDERWORLD_NEGATIVE_PROMPT,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        """Return a standard .splat artifact or its structured runtime result.

        camera_path is JSON/[N,4,4] OpenCV world-to-camera in exported units.
        interactions also accepts named upstream viewer matrices. Prompts
        guide occlusion completion and newly visible regions, not the seed
        image itself. This route is offline, not a realtime browser session.
        """
        for name, value, minimum, maximum in (
            ("depth_steps", depth_steps, 1, 1000),
            ("inpaint_steps", inpaint_steps, 1, 999),
            ("normal_steps", normal_steps, 1, 999),
            ("refinement_steps", refinement_steps, 0, None),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                interval = f"[{minimum}, {maximum}]" if maximum is not None else "[0, infinity)"
                raise ValueError(f"WonderWorld {name} must be an integer in {interval}")
        if not math.isfinite(focal_length) or focal_length <= 0:
            raise ValueError("WonderWorld focal_length must be positive and finite")
        if output_path is not None and Path(output_path).suffix != ".splat":
            raise ValueError("WonderWorld output_path must end in .splat")
        image, matrices = self.operator.prepare(
            images=images, input_path=input_path, video=video, camera_path=camera_path, interactions=interactions
        )
        result = self.synthesis_model.predict(
            image=image,
            camera_path=matrices,
            prompt=prompt,
            output_path=output_path,
            run_dir=run_dir,
            seed=seed,
            depth_steps=depth_steps,
            inpaint_steps=inpaint_steps,
            normal_steps=normal_steps,
            refinement_steps=refinement_steps,
            focal_length=focal_length,
            gen_layer=gen_layer,
            negative_prompt=negative_prompt,
        )
        return result if return_dict else result["artifact_path"]

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> dict[str, Any]:
        """Adapt normalized evaluation inputs and controls to the native call."""
        options = {
            key: value for key, value in invocation.operator_kwargs.items() if key not in {"sample_id", "task_name"}
        }
        options.update(invocation.pipeline_kwargs)
        viewer = options.pop("camera_view_matrix_sequence", None)
        if viewer is not None:
            options.setdefault("camera_path", {"camera_view_matrix_sequence": viewer})
        options.setdefault("prompt", invocation.prompt)
        options.setdefault("images", invocation.image)
        options.setdefault("input_path", invocation.ref_image_path)
        options.setdefault("video", invocation.video)
        options.setdefault("interactions", invocation.interactions)
        options.setdefault("output_path", invocation.output_path)
        options["return_dict"] = True
        return cast(dict[str, Any], self(**options))

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Reject realtime streaming; use a finite camera path per offline call."""
        del args, kwargs
        raise NotImplementedError("WonderWorld supports offline scene generation, not realtime streaming")
