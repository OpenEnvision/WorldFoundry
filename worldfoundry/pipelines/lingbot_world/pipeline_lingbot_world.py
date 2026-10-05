"""Lingbot World visual generation pipeline module."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from ...operators.lingbot_world_operator import LingBotOperator
from ...synthesis.visual_generation.lingbot_world.lingbot_world_runtime.utils.wasd_ijkl_to_c2ws import (
    generate_and_save_trajectory,
)
from ...synthesis.visual_generation.lingbot_world.lingbot_world_synthesis import LingBotSynthesis
from ...synthesis.visual_generation.lingbot_world.runtime import (
    DEFAULT_LINGBOT_ACT_REPO,
    DEFAULT_LINGBOT_BASE_REPO,
)
from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ..pipeline_utils import PipelineABC

WMFACTORY_BASE_INTRINSICS = np.asarray(
    [502.9116, 503.10812, 415.77786, 239.77779],
    dtype=np.float32,
)
WMFACTORY_INTERACTION_KEYS: Dict[str, tuple[str, ...]] = {
    "forward": ("w",),
    "backward": ("s",),
    "left": ("a",),
    "right": ("d",),
    "forward_left": ("w", "a"),
    "forward_right": ("w", "d"),
    "backward_left": ("s", "a"),
    "backward_right": ("s", "d"),
    "camera_up": ("i",),
    "camera_down": ("k",),
    "camera_l": ("j",),
    "camera_r": ("l",),
    "camera_ul": ("i", "j"),
    "camera_ur": ("i", "l"),
    "camera_dl": ("k", "j"),
    "camera_dr": ("k", "l"),
}

logger = logging.getLogger(__name__)


def _wmfactory_trajectory_from_interactions(
    interactions: Optional[list[str]],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wmfactory trajectory from interactions helper function."""
    keys: list[str] = []
    unknown: list[str] = []
    for raw_token in interactions or []:
        token = str(raw_token).strip().lower()
        if not token:
            continue
        mapped = WMFACTORY_INTERACTION_KEYS.get(token)
        if mapped is None:
            unknown.append(token)
            continue
        for key in mapped:
            if key not in keys:
                keys.append(key)

    if unknown:
        unsupported = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unsupported LingBot WMFactory interaction token(s): {unsupported}")

    repeat_frames = max(int(num_frames) - 1, 1)
    frame_keys = [keys[:] for _ in range(repeat_frames)]
    c2ws = np.asarray(generate_and_save_trajectory(frame_keys), dtype=np.float32)
    intrinsics = np.repeat(WMFACTORY_BASE_INTRINSICS[None, :], c2ws.shape[0], axis=0).astype(np.float32)
    return c2ws, intrinsics


class LingBotPipeline(PipelineABC):
    """Pipeline implementation for LingBot visual generation."""
    def __init__(self,
                 operators: Optional[LingBotOperator] = None,
                 synthesis_model: Optional[LingBotSynthesis] = None,
                 memory_module: Optional[Any] = None,
                 device: str = "cuda",
                 model_id: str = "lingbot-world",
                 ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.model_id = model_id
        self._realtime_session = None

    @classmethod
    def from_pretrained(cls,
                        model_path: Union[str, Dict[str, Any], None] = None,
                        required_components: Optional[Dict[str, Any]] = None,
                        mode: str = "i2v-A14B",
                        device: str = "cuda",
                        **kwargs) -> "LingBotPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop("model_path", None)
        resolved_model_id = str(
            component_options.get("model_id")
            or component_options.get("profile_id")
            or kwargs.get("model_id")
            or "lingbot-world"
        )
        mode = component_options.pop("mode", mode)
        kwargs = cls._strip_framework_loading_options({**component_options, **kwargs})
        model_path = model_path or (
            DEFAULT_LINGBOT_ACT_REPO if resolved_model_id == "lingbot-world-act" else DEFAULT_LINGBOT_BASE_REPO
        )

        logger.info("Loading LingBot World Model from %s...", model_path)

        synthesis_model = LingBotSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            task=mode,
            device=device,
            control_type="act" if resolved_model_id == "lingbot-world-act" else "cam",
            **kwargs
        )

        operators = LingBotOperator()
        memory_module = VisualFrameMemory(model_id=resolved_model_id)

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            model_id=resolved_model_id,
        )
        return pipeline

    def process(self,
                images: Any = None,
                prompt: Optional[str] = None,
                interactions: Optional[list[str]] = None,
                action_path: Optional[str] = None,
                resize_H: int = 480,
                resize_W: int = 832,
                num_frames: Optional[int] = 81,
                wmfactory_action_controls: bool = False,
                poses_c2ws: Any = None,
                poses_Ks: Any = None):

        """Process and normalize input arguments and conditions for inference."""
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("Input image must be a PIL image or an image path.")

        output_dict = {
            "pil_image": images,
            "prompt": prompt if prompt is not None else "",
            "action_path": action_path,
            "c2ws": None,
            "Ks": None,
        }

        if action_path is None and poses_c2ws is not None:
            if poses_Ks is None:
                raise ValueError("LingBot direct pose input requires both poses_c2ws and poses_Ks.")
            output_dict["c2ws"] = np.asarray(poses_c2ws, dtype=np.float32)
            output_dict["Ks"] = np.asarray(poses_Ks, dtype=np.float32)
        elif action_path is None and interactions:
            if wmfactory_action_controls:
                c2ws, Ks = _wmfactory_trajectory_from_interactions(interactions, num_frames or 81)
            else:
                c2ws, Ks = self.operators.traj_generator.generate(
                    interactions,
                    num_frames,
                    resize_H,
                    resize_W,
                )
            output_dict["c2ws"] = c2ws
            output_dict["Ks"] = Ks

        return output_dict

    def __call__(self,
                  images: Any = None,
                  num_frames: Optional[int] = 81,
                  prompt: Optional[str] = None,
                  interactions: Optional[list[str]] = None,
                  action_path: Optional[str] = None,
                  resize_H: int = 480,
                  resize_W: int = 832,
                  max_area: int = 720 * 1280,
                  vis_ui: bool = False,
                  allow_act2cam: bool = False,
                  action_string: Optional[str] = None,
                  wmfactory_action_controls: bool = False,
                  poses_c2ws: Any = None,
                  poses_Ks: Any = None,
                  action_matrix: Any = None,
                  output_path: str | Path | None = None,
                  fps: int = 16,
                  return_dict: bool = False,
                  operator_kwargs: Any = None,
                  seed: int = 42,
                  **kwds):

        """Execute the complete pipeline generation flow."""
        # PipelineInvocation carries model-specific inputs separately from the
        # common image/prompt/action fields. Keep direct-call arguments as the
        # priority, while accepting the same pose and action payload through
        # WorldFoundryPipelineRunner's request.inputs.
        if isinstance(operator_kwargs, dict):
            if action_path is None:
                action_path = operator_kwargs.get("action_path")
            if poses_c2ws is None:
                poses_c2ws = operator_kwargs.get("poses_c2ws")
            if poses_Ks is None:
                poses_Ks = operator_kwargs.get("poses_Ks")
            if action_matrix is None:
                action_matrix = operator_kwargs.get("action_matrix")
        processed_inputs = self.process(
            images=images,
            prompt=prompt,
            interactions=interactions,
            action_path=action_path,
            resize_H=resize_H,
            resize_W=resize_W,
            num_frames=num_frames,
            wmfactory_action_controls=wmfactory_action_controls,
            poses_c2ws=poses_c2ws,
            poses_Ks=poses_Ks,
        )

        output_video = self.synthesis_model.predict(
            prompt=processed_inputs["prompt"],
            pil_image=processed_inputs["pil_image"],
            action_path=processed_inputs["action_path"],
            c2ws=processed_inputs["c2ws"],
            Ks=processed_inputs["Ks"],
            num_output_frames=num_frames,
            max_area=max_area,
            vis_ui=vis_ui,
            allow_act2cam=allow_act2cam,
            action_string=action_string,
            action_matrix=action_matrix,
            seed=seed,
            **kwds
        )

        if output_video is None:
            # The distributed LingBot runtime decodes only on rank zero.  The
            # remaining ranks still have to execute the complete denoising
            # graph and participate in Studio's status gather, but an empty
            # local payload is expected there (the V2 pipeline follows the
            # same contract).  Treat ``None`` as an error only for rank zero
            # or a non-distributed invocation, where it really does mean that
            # no video was produced.
            import torch.distributed as dist

            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
                and dist.get_rank() != 0
            ):
                return None
            raise RuntimeError("LingBot did not return video frames on the current rank.")
        artifact_path = None
        if output_path is not None:
            from worldfoundry.core.io import save_video_frames

            output = Path(output_path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            save_video_frames(output_video, output, fps=int(fps))
            artifact_path = str(output)
        if return_dict:
            return {
                "status": "success",
                "model_id": self.model_id,
                "artifact_kind": "generated_video",
                "artifact_path": artifact_path,
                "generated_video_path": artifact_path,
                "video": output_video,
            }
        return output_video

    def stream(self,
                prompt: Optional[str] = None,
                interactions: Optional[list[str]] = None,
                images: Any = None,
                action_path: Optional[str] = None,
                num_frames: Optional[int] = 81,
                resize_H: int = 480,
                resize_W: int = 832,
                max_area: int = 720 * 1280,
                vis_ui: bool = False,
                allow_act2cam: bool = False,
                action_string: Optional[str] = None,
                wmfactory_action_controls: bool = False,
                seed: int = 42,
                **kwds) -> np.ndarray:

        # 1. Initialize Memory if images provided (First Turn)
        """Stream visual generation outputs chunk by chunk."""
        if images is not None:
            logger.info("--- Stream Started ---")
            self.memory_module.manage(action="reset") # Clear old memory
            self.memory_module.record(images, type="image")

        # 2. Retrieve Context (Input for this turn)
        current_img = self.memory_module.select()
        if current_img is None:
            raise ValueError("No image in storage. Provide 'images' first.")

        # 3. Generate Video
        video_output = self.__call__(
            images=current_img,
            num_frames=num_frames,
            prompt=prompt,
            interactions=interactions,
            action_path=action_path,
            resize_H=resize_H,
            resize_W=resize_W,
            max_area=max_area,
            vis_ui=vis_ui,
            allow_act2cam=allow_act2cam,
            action_string=action_string,
            wmfactory_action_controls=wmfactory_action_controls,
            seed=seed,
            **kwds
        ) # Returns numpy array [T, H, W, C]

        # 4. Record Result (Updates context for next turn)
        if video_output is not None:
            self.memory_module.record(video_output, type="video_chunk")

        return video_output

    def _ensure_realtime_session(self):
        """Build the resident fast session once per loaded pipeline."""

        if self._realtime_session is None:
            from ...synthesis.visual_generation.lingbot_world.realtime import (
                LingBotRealtimeSession,
            )

            core_model = getattr(self.synthesis_model, "core_model", None)
            self._realtime_session = LingBotRealtimeSession(core_model)
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        """Load realtime-only components without creating a rollout."""

        session = self._ensure_realtime_session()
        return {"realtime_spec": session.realtime_spec().to_payload()}

    def configure_realtime(
        self,
        images: Any,
        prompt: Optional[str] = None,
        seed: int = 42,
        fps: int = 16,
        **_: Any,
    ) -> dict[str, Any]:
        """Initialize prompt, image, VAE, and transformer state once."""

        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("LingBot realtime configuration requires a PIL image or image path.")
        self.memory_module.manage(action="reset")
        self.memory_module.record(images, type="image")
        return self._ensure_realtime_session().configure(
            image=images,
            prompt=prompt or "",
            seed=seed,
            fps=fps,
        )

    def stream_realtime(
        self,
        prompt: Optional[str] = None,
        interactions: Optional[list[str]] = None,
        realtime_segments: Optional[list[dict[str, Any]]] = None,
        seed: int = 42,
        **_: Any,
    ) -> Optional[dict[str, Any]]:
        """Advance the existing autoregressive rollout by one model chunk."""

        session = self._ensure_realtime_session()
        if prompt is not None:
            session.update_prompt(prompt)
        result = session.generate(
            interactions=interactions or [],
            control_segments=realtime_segments,
            seed=seed,
        )
        if result is not None and result.get("video") is not None:
            self.memory_module.record(result["video"], type="video_chunk")
        return result

    def realtime_next_output_frames(self) -> int:
        """Return 9 for the causal seed chunk and 12 thereafter."""

        return int(self._ensure_realtime_session().next_output_frames())

    def reset_realtime(self) -> None:
        """Drop rollout state while retaining model and decoder weights."""

        if self._realtime_session is not None:
            self._realtime_session.reset()
        self.memory_module.manage(action="reset")
