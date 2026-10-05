"""Matrix Game 3 visual generation pipeline module."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from PIL import Image

from ...operators.matrix_game_3_operator import MatrixGame3Operator
from ...synthesis.visual_generation.matrix_game.matrix_game_3_synthesis import MatrixGame3Synthesis
from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ..pipeline_utils import PipelineABC
from worldfoundry.core.io.paths import scratch_directory

DEFAULT_MATRIX_GAME3_COMPONENTS = {
    # Mirrors the upstream lightweight inference recipe while staying single-GPU friendly.
    "compile_vae": True,
    "lightvae_pruning_rate": 0.5,
    "use_int8": True,
    "vae_type": "mg_lightvae",
}


def _looks_like_matrix_game3_code_path(path_value: Optional[str]) -> bool:
    """Looks like matrix game3 code path helper function."""
    if path_value is None:
        return False
    candidate = Path(str(path_value)).expanduser()
    return (
        (candidate / "generate.py").is_file()
        and (candidate / "pipeline" / "inference_pipeline.py").is_file()
        and (candidate / "wan" / "__init__.py").is_file()
    )


class MatrixGame3Pipeline(PipelineABC):
    """Pipeline implementation for MatrixGame3 visual generation."""
    def __init__(
        self,
        operators: Optional[MatrixGame3Operator] = None,
        synthesis_model: Optional[MatrixGame3Synthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.operators = operators or MatrixGame3Operator()
        self.memory_module = memory_module or VisualFrameMemory(model_id="matrix-game-3")
        self.device = device
        self.weight_dtype = weight_dtype
        self._realtime_session: Any = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[dict] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> "MatrixGame3Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = {**DEFAULT_MATRIX_GAME3_COMPONENTS, **(required_components or {})}
        checkpoint_dir = required_components.get("checkpoint_dir")
        if model_path is not None and not _looks_like_matrix_game3_code_path(model_path):
            checkpoint_dir = model_path
        synthesis_model = MatrixGame3Synthesis.from_pretrained(
            pretrained_model_path=checkpoint_dir,
            device=device,
            checkpoint_dir=checkpoint_dir,
            **{
                key: value
                for key, value in {**required_components, **kwargs}.items()
                if key not in {"repo_root", "checkpoint_dir"}
            },
        )
        return cls(
            operators=MatrixGame3Operator(),
            synthesis_model=synthesis_model,
            memory_module=VisualFrameMemory(model_id="matrix-game-3"),
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        input_image,
        interactions: Sequence[Union[str, Dict[str, Any]]],
        num_frames: Optional[int] = None,
        num_iterations: Optional[int] = None,
    ):
        """Process and normalize input arguments and conditions for inference."""
        image = self.operators.process_perception(input_image)
        self.operators.get_interaction(interactions)
        operator_condition = self.operators.process_interaction(
            num_frames=num_frames,
            num_iterations=num_iterations,
        )
        self.operators.delete_last_interaction()
        return {
            "image": image,
            **operator_condition,
        }

    def __call__(
        self,
        images,
        interactions: Optional[List[str]] = None,
        prompt: str = "",
        num_frames: Optional[int] = None,
        num_iterations: Optional[int] = None,
        size=(704, 1280),
        fps: int = 17,
        output_path: str | Path | None = None,
        output_dir: Optional[str] = None,
        save_name: str = "matrix_game_3",
        visualize_ops: bool = True,
        show_progress: bool = True,
        seed: int = 42,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if isinstance(images, (str, Path)):
            images = Image.open(str(images)).convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("Unsupported image type. Expected PIL.Image.")
        if interactions is None:
            interactions = ["forward", "camera_r"]

        output_dict = self.process(
            input_image=images,
            interactions=interactions,
            num_frames=num_frames,
            num_iterations=num_iterations,
        )
        save_root = Path(output_dir).expanduser().resolve() if output_dir else scratch_directory("matrix_game_3_")
        save_root.mkdir(parents=True, exist_ok=True)
        synthesis_result = self.synthesis_model.predict(
            image=output_dict["image"],
            prompt=prompt or "",
            keyboard_condition=output_dict["keyboard_condition"],
            mouse_condition=output_dict["mouse_condition"],
            num_iterations=output_dict["num_iterations"],
            output_dir=str(save_root),
            save_name=save_name,
            size=size,
            fps=fps,
            seed=seed,
            visualize_ops=visualize_ops,
            show_progress=show_progress,
            **kwargs,
        )
        generated_path = Path(synthesis_result["generated_video_path"]).expanduser().resolve()
        artifact_path = generated_path
        if output_path is not None:
            artifact_path = Path(output_path).expanduser().resolve()
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if artifact_path != generated_path:
                shutil.copyfile(generated_path, artifact_path)

        result = {
            **synthesis_result,
            "artifact_path": str(artifact_path),
            "status": "succeeded",
        }
        if return_dict or output_path is not None:
            return result
        return synthesis_result["video"]

    def stream(
        self,
        images: Optional[Union[Image.Image, str, Path]],
        interactions: List[str],
        prompt: str = "",
        num_frames: Optional[int] = None,
        num_iterations: Optional[int] = None,
        size=(704, 1280),
        fps: int = 17,
        output_dir: Optional[str] = None,
        save_name: str = "matrix_game_3_stream",
        visualize_ops: bool = True,
        show_progress: bool = True,
        reset_memory: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if isinstance(images, (str, Path)):
            images = Image.open(str(images)).convert("RGB")
        if reset_memory:
            self.memory_module.manage(action="reset")
        if images is not None:
            self.memory_module.record(images)

        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No image in storage. Provide 'images' first.")

        turn_index = len(self.memory_module.storage)
        video_output = self.__call__(
            images=current_image,
            interactions=interactions,
            prompt=prompt,
            num_frames=num_frames,
            num_iterations=num_iterations,
            size=size,
            fps=fps,
            output_dir=output_dir,
            save_name=f"{save_name}_{turn_index}",
            visualize_ops=visualize_ops,
            show_progress=show_progress,
            seed=seed,
            **kwargs,
        )
        self.memory_module.record(video_output)
        return video_output

    def _ensure_realtime_session(self) -> Any:
        """Construct one model-owned resident rollout without spawning a runner."""

        if self._realtime_session is None:
            if self.synthesis_model is None:
                raise RuntimeError("Matrix-Game 3 pipeline is not initialized.")
            from ...synthesis.visual_generation.matrix_game.matrix_game_3_runtime.realtime import (
                MatrixGame3RealtimeSession,
            )

            self._realtime_session = MatrixGame3RealtimeSession(
                self.synthesis_model.runtime,
                self.operators,
            )
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        """Load resident weights and expose the native playback cadence."""

        session = self._ensure_realtime_session()
        return {"realtime_spec": session.realtime_spec().to_payload()}

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 17,
        size: str | Sequence[int] = "704*1280",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode user-provided session conditions while retaining model weights."""

        if isinstance(images, (str, Path)):
            with Image.open(images) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("Matrix-Game 3 realtime requires a PIL image or image path.")
        if self.memory_module is not None:
            self.memory_module.manage(action="reset")
            self.memory_module.record(images)
        return self._ensure_realtime_session().configure(
            image=images,
            prompt=str(prompt or ""),
            seed=seed,
            fps=fps,
            size=size,
            num_inference_steps=kwargs.get("num_inference_steps"),
            sample_shift=kwargs.get("sample_shift"),
            sample_guide_scale=kwargs.get("sample_guide_scale"),
            use_base_model=kwargs.get("use_base_model"),
        )

    def stream_realtime(
        self,
        prompt: str | None = None,
        interactions: Sequence[str] | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Advance the resident camera-aware rollout by one native window."""

        session = self._ensure_realtime_session()
        if prompt is not None:
            session.update_prompt(prompt)
        return session.generate(
            interactions=list(interactions or ()),
            control_segments=realtime_segments,
        )

    def realtime_next_output_frames(self) -> int:
        return int(self._ensure_realtime_session().next_output_frames())

    def reset_realtime(self) -> None:
        """Release session state but keep native model weights loaded."""

        if self._realtime_session is not None:
            self._realtime_session.reset()
        if self.memory_module is not None:
            self.memory_module.manage(action="reset")
