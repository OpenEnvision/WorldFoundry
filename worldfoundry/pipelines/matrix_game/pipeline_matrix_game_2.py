"""Matrix Game 2 visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import torch
import os
import inspect
from PIL import Image
from typing import Optional, Any, List, Mapping, Sequence, Union, TYPE_CHECKING
from worldfoundry.core.io import write_video
from ...operators.matrix_game_2_operator import MatrixGame2Operator
from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
import logging

from worldfoundry.runtime.env import resolve_ckpt_dir

if TYPE_CHECKING:
    from ...synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis

logger = logging.getLogger(__name__)


class MatrixGame2Pipeline(PipelineABC):
    """Pipeline implementation for MatrixGame2 visual generation."""
    def __init__(self,
                 operators: Optional[MatrixGame2Operator] = None,
                 synthesis_model: Optional["MatrixGame2Synthesis"] = None,
                 memory_module: Optional[Any] = None,
                 device: str = "cuda",
                 # Use bfloat16 precision to balance memory efficiency and numeric range
                 weight_dtype = torch.bfloat16,
                 ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.current_image = None
        self._realtime_session: Any = None

    @classmethod
    def from_pretrained(cls,
                        model_path: Any = None,
                        required_components: Optional[Mapping[str, Any]] = None,
                        device: str = "cuda",
                        # Use bfloat16 precision to balance memory efficiency and numeric range
                        weight_dtype = torch.bfloat16,
                        mode = "universal",
                        **kwargs) -> "MatrixGame2Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        runtime_kwargs = dict(model_path) if isinstance(model_path, Mapping) else {}
        if isinstance(model_path, Mapping):
            synthesis_model_path = runtime_kwargs.pop("model_path", None) or runtime_kwargs.pop(
                "pretrained_model_path", None
            )
        else:
            synthesis_model_path = model_path
        nested_components = runtime_kwargs.pop("required_components", None)
        if nested_components is not None:
            if not isinstance(nested_components, Mapping):
                raise TypeError("required_components must be a mapping")
            runtime_kwargs.update(nested_components)
        if required_components is not None:
            if not isinstance(required_components, Mapping):
                raise TypeError("required_components must be a mapping")
            runtime_kwargs.update(required_components)
        runtime_kwargs.update(kwargs)
        mode = runtime_kwargs.pop("mode", mode)
        if synthesis_model_path in {None, ""}:
            synthesis_model_path = str(resolve_ckpt_dir() / "Matrix-Game-2.0")
        
        logger.info("Loading MatrixGame2 synthesis model from %s...", synthesis_model_path)
        from ...synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis

        synthesis_model = MatrixGame2Synthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device,
            mode=mode,
            weight_dtype=weight_dtype,
            **runtime_kwargs
        )
        operators = MatrixGame2Operator(mode=mode)
        memory_module = VisualFrameMemory(model_id="matrix-game-2")

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline
    
    def process(self,
                input_image,
                num_output_frames,
                resize_H=352,
                resize_W=640,
                interaction_signal=None,
                official_bench_actions: bool = False,
                seed: Optional[int] = None):
        """
        the input_image is PIL image
        """
        if interaction_signal is None:
            interaction_signal = [
                "forward", "left", "right",
                "forward_left", "forward_right",
                "camera_l", "camera_r",
            ]
        preception_dict = self.operators.process_perception(input_image, num_output_frames, resize_H, resize_W,
                                                            device=self.device, weight_dtype=self.weight_dtype)

        img_cond = self.synthesis_model.vae.encode(preception_dict["img_cond"], device=self.device,
                                                   **preception_dict["tiler_kwargs"]).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.synthesis_model.vae.clip.encode_video(preception_dict["image"])

        output_dict = {
            "cond_concat": cond_concat,
            "visual_context": visual_context
        }

        num_frames = (num_output_frames - 1) * 4 + 1
        if official_bench_actions:
            operator_condition = self.operators.process_official_bench_actions(num_frames=num_frames, seed=seed)
        else:
            # define the interaction
            self.operators.get_interaction(interaction_signal)
            operator_condition = self.operators.process_interaction(num_frames=num_frames)
            self.operators.delete_last_interaction()

        output_dict['operator_condition'] = operator_condition

        return output_dict

    def __call__(self,
                 images,
                 interactions=None,
                 num_frames=None,
                 size = (352, 640),
                 seed: Optional[int] = None,
                 visualize_ops=True,
                 visualize_warning=False,
                 official_bench_actions: bool = False,
                 return_dict: bool = False,
                 output_path: Optional[Union[str, os.PathLike]] = None,
                 fps: Optional[int] = None,
                 **kwds):
        """Execute the complete pipeline generation flow."""
        if not visualize_warning:
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
            logging.getLogger("torch._inductor").setLevel(logging.WARNING)
        if isinstance(images, Image.Image):
            input_image = images.convert("RGB")
        elif isinstance(images, (str, os.PathLike)):
            input_image = Image.open(os.fspath(images)).convert("RGB")
        else:
            raise ValueError("Unsupported image type. Expected PIL.Image or image path.")
        if interactions is None and not official_bench_actions:
            interactions = [
                "forward", "left", "right",
                "forward_left", "forward_right",
                "camera_l", "camera_r",
            ]
        if num_frames is None:
            num_output_frames = 150 if official_bench_actions else len(interactions) * 12
        else:
            num_output_frames = num_frames
        runtime = getattr(self.synthesis_model, "runtime", None)
        runtime_pipeline = getattr(runtime, "pipeline", None)
        frame_block = int(getattr(runtime_pipeline, "num_frame_per_block", 1) or 1)
        if frame_block > 1 and num_output_frames % frame_block:
            num_output_frames = ((num_output_frames + frame_block - 1) // frame_block) * frame_block
        resize_H, resize_W = size

        output_dict = self.process(
            input_image=input_image,
            num_output_frames=num_output_frames,
            resize_H=resize_H,
            resize_W=resize_W,
            interaction_signal=interactions,
            official_bench_actions=official_bench_actions,
            seed=seed,
        )
        predict_kwargs = dict(kwds)
        runtime_predict = getattr(getattr(self.synthesis_model, "runtime", None), "predict", None)
        try:
            predict_signature = inspect.signature(runtime_predict or self.synthesis_model.predict)
        except (TypeError, ValueError):
            predict_signature = None
        if predict_signature is not None and not any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in predict_signature.parameters.values()
        ):
            predict_kwargs = {
                key: value
                for key, value in predict_kwargs.items()
                if key in predict_signature.parameters
            }

        output_video = self.synthesis_model.predict(
            cond_concat=output_dict['cond_concat'],
            visual_context=output_dict['visual_context'],
            operator_condition=output_dict['operator_condition'],
            num_output_frames=num_output_frames,
            seed=seed,
            operation_visualization=visualize_ops,
            **predict_kwargs
        )
        if output_path is not None:
            output_path = os.fspath(output_path)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            write_video(output_video, output_path, fps=fps if fps is not None else 12)
        if return_dict:
            return {
                "status": "ok",
                "model_id": "matrix-game-2",
                "artifact_kind": "generated_video",
                "artifact_path": str(output_path) if output_path is not None else "",
                "num_output_frames": num_output_frames,
                "fps": fps if fps is not None else 12,
                "mode": getattr(self.operators, "mode", None),
                "official_bench_actions": bool(official_bench_actions),
                "backend_quality": "in_tree_runtime",
                "video": output_video,
            }
        return output_video
    
    def stream(self,
               images: Optional[Image.Image],
               interactions: List[str],
               num_frames: int = 15,
               size = (352, 640),
               seed: Optional[int] = None,
               visualize_ops: bool = False,
               visualize_warning=False,
               **kwds) -> torch.Tensor:
        """Stream visual generation outputs chunk by chunk."""
        if not visualize_warning:
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        if images is not None:
            logger.info("--- Stream Started ---")
            self.memory_module.record(images)
        
        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No image in storage. Provide 'images' first.")

        video_output = self.__call__(
            images=current_image,
            interactions=interactions,
            num_frames=num_frames,
            size=size,
            seed=seed,
            visualize_ops=visualize_ops,
            **kwds
        )

        self.memory_module.record(video_output)

        return video_output

    def _ensure_realtime_session(self) -> Any:
        """Construct the model-owned resident rollout adapter once."""

        if self._realtime_session is None:
            if self.synthesis_model is None or self.operators is None:
                raise RuntimeError("Matrix-Game 2 pipeline is not initialized.")
            from ...synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
                MatrixGame2RealtimeSession,
            )

            self._realtime_session = MatrixGame2RealtimeSession(
                self.synthesis_model.runtime,
                self.operators,
            )
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        """Load the resident adapter and report the model's native cadence."""

        session = self._ensure_realtime_session()
        return {"realtime_spec": session.realtime_spec().to_payload()}

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 12,
        **_: Any,
    ) -> dict[str, Any]:
        """Encode a seed image and allocate rollout caches without generating."""

        del prompt, fps
        if isinstance(images, (str, os.PathLike)):
            with Image.open(os.fspath(images)) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("Matrix-Game 2 realtime requires a PIL image or image path.")
        if self.memory_module is not None:
            self.memory_module.manage(action="reset")
            self.memory_module.record(images, record_frames=False, type="image")
        return self._ensure_realtime_session().configure(images, seed=seed)

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int = 42,
        prompt: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        """Advance the existing causal rollout by exactly one native block."""

        # The rollout owns one continuous RNG stream. Re-seeding every control
        # chunk would repeat diffusion noise and create visible temporal beats.
        del seed, prompt
        return self._ensure_realtime_session().generate(
            interactions=list(interactions or ()),
            control_segments=realtime_segments,
        )

    def realtime_next_output_frames(self) -> int:
        return int(self._ensure_realtime_session().next_output_frames())

    def reset_realtime(self) -> None:
        """Drop rollout caches while retaining decoder and model weights."""

        if self._realtime_session is not None:
            self._realtime_session.reset()
        if self.memory_module is not None:
            self.memory_module.manage(action="reset")
