"""Astra visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import logging
import torch
import os
import numpy as np
from PIL import Image
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence, Union

from ...operators.astra_operator import AstraOperator
from ...synthesis.visual_generation.kling.astra_synthesis import AstraSynthesis
from ...synthesis.visual_generation.memory.kling.astra import AstraMemory

logger = logging.getLogger(__name__)


@dataclass
class AstraConfig:
    # pretrained model paths
    """AstraConfig implementation for WorldFoundry pipelines."""
    astra_path: str = ""
    wan_model_path: str = ""
    # default generation parameters
    start_frame: int = 0
    initial_condition_frames: int = 1
    frames_per_generation: int = 8
    total_frames_to_generate: int = 24  # frames per direction step; actual total = this * num_directions
    max_history_frames: int = 100       # sliding window size for memory
    num_inference_steps: int = 50
    # guidance parameters
    use_camera_cfg: bool = False
    camera_guidance_scale: float = 2.0
    text_guidance_scale: float = 1.0
    # MoE models parameters
    moe_num_experts: int = 3
    moe_top_k: int = 1
    moe_hidden_dim: Optional[int] = None
    # data parameters
    modality_type: str = "sekai"
    use_real_poses: bool = False
    scene_info_path: Optional[str] = None
    use_gt_prompt: bool = False
    # inference parameters
    device: str = "cuda"
    add_icons: bool = False


class AstraPipeline(PipelineABC):
    """Pipeline implementation for Astra visual generation."""
    def __init__(self, operator, synthesis, memory, config):
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator
        self.synthesis = synthesis
        self.memory = memory
        self.config = config
        self.device = synthesis.device

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        required_components: dict,
                        device: str = "cuda",
                        **kwargs):
        """
        Load pretrained models and initialize the pipeline.
        """
        if isinstance(required_components, dict) and "wan_model_path" in required_components.keys():
            wan_model_path = required_components.get("wan_model_path", "Wan-AI/Wan2.1-T2V-1.3B")
        else:
            wan_model_path = "Wan-AI/Wan2.1-T2V-1.3B"
        config = AstraConfig(
            astra_path=model_path,
            wan_model_path=wan_model_path,
            device=device
        )
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        logger.info("Loading Astra components on %s...", device)
        synthesis = AstraSynthesis.from_pretrained(config, device=device)
        operator = AstraOperator(device=device)
        memory = AstraMemory(capacity=config.max_history_frames)

        return cls(operator, synthesis, memory, config)

    def _stitch_camera_embeddings(
        self,
        direction_embeddings: List[torch.Tensor],
        initial_condition_frames: int,
        per_direction_frames: int,
        total_frames: int
    ) -> torch.Tensor:
        """
        Stitch per-direction camera embeddings into a single continuous embedding
        that supports absolute-position indexing used by the memory module.

        The memory's select() method reads camera_embedding_full[T : T + n] where
        T is the current history length. After the first direction's frames are
        generated, T advances past them and the next direction's poses must be
        at the correct absolute positions. This method achieves that by extracting
        each direction's motion portion and placing them sequentially.

        Layout of the combined embedding (example with 2 directions):
            [0, cond)                            : condition frame(s) from direction 0
            [cond, cond + per_dir)               : direction 0 motion poses
            [cond + per_dir, cond + 2 * per_dir) : direction 1 motion poses
            [cond + 2 * per_dir, ...)            : repeated final pose for padded chunks

        Note: the mask column (last dim) is included as-is, but the memory module
        overrides it in prepare_framepack, so its values here do not matter.

        Args:
            direction_embeddings: Camera embedding per direction, each [T_i, emb_dim].
            initial_condition_frames: Number of condition frames at the beginning.
            per_direction_frames: Number of new motion frames per direction step.
            total_frames: Total motion frames across all directions.

        Returns:
            Combined camera embedding of shape [needed_length, emb_dim].
        """
        emb_dim = direction_embeddings[0].shape[1]
        dtype = direction_embeddings[0].dtype
        device = direction_embeddings[0].device

        # Allocate with sufficient padding for FramePack indexing
        needed_length = initial_condition_frames + total_frames + 30
        combined = torch.zeros(needed_length, emb_dim, dtype=dtype, device=device)

        # Copy condition frame(s) from the first direction's embedding
        cond_len = min(initial_condition_frames, direction_embeddings[0].shape[0])
        combined[:cond_len, :] = direction_embeddings[0][:cond_len, :]

        # Copy motion frames from each direction sequentially
        dst_offset = initial_condition_frames
        for emb in direction_embeddings:
            # In each individual embedding, motion occupies frames
            # [initial_condition_frames, initial_condition_frames + per_direction_frames)
            src_start = initial_condition_frames
            src_end = min(initial_condition_frames + per_direction_frames, emb.shape[0])
            n_motion = src_end - src_start
            combined[dst_offset:dst_offset + n_motion, :] = emb[src_start:src_end, :]
            dst_offset += n_motion

        if dst_offset > 0:
            combined[dst_offset:] = combined[dst_offset - 1].clone()
        return combined

    def process(self, input_: str, interaction: Dict[str, Any], args: Optional[AstraConfig] = None):
        """
        Process the input image and interaction signals to prepare all data
        needed for autoregressive generation.

        Args:
            input_: Path to the condition image.
            interaction: Dict with keys:
                - 'prompt' (str): Text description of the scene.
                - 'direction' (str | List[str]): Camera direction(s).

        Returns:
            Dict containing initial_latents, prompt embeddings, stitched camera
            embeddings, and the computed total_frames.
        """
        args = args or self.config
        condition_image = input_
        prompt = interaction.get("prompt", "")
        directions = interaction.get("direction", "forward")

        # Normalize direction to a list for unified handling (backward compatible)
        if isinstance(directions, str):
            directions = [directions]

        num_directions = len(directions)
        per_direction_frames = args.total_frames_to_generate
        total_frames = per_direction_frames * num_directions

        # 1. Load and encode the condition image (Perception -> Representation)
        logger.info("Processing image: %s", condition_image)
        frames = self.operator.process_perception(condition_image=condition_image)
        latents = self.synthesis.encode_frames(frames)

        target_height, target_width = 60, 104
        C, T, H, W = latents.shape
        if H > target_height or W > target_width:
            h_start = (H - target_height) // 2
            w_start = (W - target_width) // 2
            latents = latents[:, :, h_start:h_start+target_height, w_start:w_start+target_width]

        model_dtype = next(self.synthesis.pipe.dit.parameters()).dtype
        history_latents = latents[:, :args.initial_condition_frames, :, :].to(self.device, dtype=model_dtype)
        initial_latents = history_latents  # backup for final concatenation

        # 2. Encode the text prompt (Interaction -> Representation)
        logger.info("Encoding prompt: %s", prompt)
        prompt_emb_pos = self.synthesis.pipe.encode_prompt(prompt)
        prompt_emb_neg = None
        if args.text_guidance_scale > 1.0:
            prompt_emb_neg = self.synthesis.pipe.encode_prompt("")

        # 3. Generate camera embeddings for each direction separately, then stitch.
        #
        #    Why separate calls: operator.process_interaction() only reads the LAST
        #    entry of self.current_interaction as the direction. So we must call it
        #    once per direction to obtain each direction's camera embedding.
        #
        #    Why stitch: memory.select() indexes into camera_embedding_full using the
        #    absolute history length T. After direction 0's frames are generated, T
        #    has advanced, so direction 1's poses must sit at the correct offset.
        logger.info(
            "Generating camera embeddings for directions: %s (%s step(s) x %s frames = %s total frames)...",
            directions, num_directions, per_direction_frames, total_frames,
        )

        direction_embeddings = []
        for direction in directions:
            # Set operator to target a single direction for this call
            self.operator.current_interaction = [direction]
            cam_emb = self.operator.process_interaction(
                modality_type=args.modality_type,
                start_frame=args.start_frame,
                initial_condition_frames=args.initial_condition_frames,
                total_frames_to_generate=per_direction_frames,
                use_real_poses=args.use_real_poses,
                scene_info_path=args.scene_info_path
            )
            direction_embeddings.append(cam_emb)

        # Stitch into one continuous embedding aligned with absolute frame positions
        camera_embedding_full = self._stitch_camera_embeddings(
            direction_embeddings,
            args.initial_condition_frames,
            per_direction_frames,
            total_frames
        )

        camera_embedding_uncond = None
        if args.use_camera_cfg:
            camera_embedding_uncond = torch.zeros_like(camera_embedding_full)

        return {
            "initial_latents": initial_latents,
            "prompt_emb_pos": prompt_emb_pos,
            "prompt_emb_neg": prompt_emb_neg,
            "camera_embedding_full": camera_embedding_full,
            "camera_embedding_uncond": camera_embedding_uncond,
            "total_frames": total_frames,
        }

    @staticmethod
    def _normalise_workspace_interaction(
        *,
        prompt: str = "",
        interactions: Dict[str, Any] | Sequence[Any] | str | None = None,
    ) -> Dict[str, Any]:
        """Normalise workspace interaction for AstraPipeline."""
        if isinstance(interactions, dict):
            payload = dict(interactions)
            payload.setdefault("prompt", prompt)
            payload.setdefault("direction", "forward")
            return payload
        if isinstance(interactions, str):
            direction: str | list[str] = interactions or "forward"
        elif interactions:
            direction = [str(item) for item in interactions]
        else:
            direction = "forward"
        return {"prompt": prompt, "direction": direction}

    def __call__(self,
                 image_path: str = "",
                 interactions: Dict[str, Any] | Sequence[Any] | str | None = None,
                 images: Optional[List[Image.Image]] = None,
                 prompt: str = "",
                 output_path: str | Path | None = None,
                 fps: int | None = None,
                 return_dict: bool = False,
                 num_frames: int | None = None,
                 frames_per_generation: int | None = None,
                 total_frames_to_generate: int | None = None,
                 num_inference_steps: int | None = None,
                 seed: int | None = None,
                 height: int | None = None,
                 width: int | None = None,
                 **kwargs: Any,
                ) -> List[Image.Image] | Dict[str, Any]:
        """
        Main entry point. Supports multi-step direction interaction.

        ``num_frames`` is the total decoded video length. The native
        ``total_frames_to_generate`` option remains latent frames per direction.
        Camera directions advance at complete native chunk boundaries; shorter
        outputs crop the generated sequence. ``seed`` uses a request-local
        generator without changing global RNG state.

        When 'direction' is a list of N directions, the pipeline generates
        total_frames_to_generate * N frames, with camera motion transitioning
        through each direction sequentially. frames_per_generation (the
        autoregressive step size) remains unchanged.

        Args:
            image_path: Path to the condition image.
            interactions: Dict with keys:
                - 'prompt' (str): Text description of the scene.
                - 'direction' (str | List[str]): One or more camera directions.
                  Supported: forward, backward, camera_l, camera_r,
                  forward_left, forward_right, s_curve, left_right.

        Returns:
            List of PIL.Image frames comprising the generated video.
        """
        if height not in (None, 480) or width not in (None, 832):
            raise ValueError("Astra currently generates 480x832 frames.")
        if fps is not None and fps <= 0:
            raise ValueError("Astra fps must be positive.")
        if num_frames is not None and num_frames < 1:
            raise ValueError("Astra num_frames must be positive.")
        args = replace(self.config)
        step_value = num_inference_steps
        for alias in ("steps", "sample_steps", "infer_steps", "sampling_steps", "num_steps"):
            if step_value is None and alias in kwargs:
                step_value = kwargs.pop(alias)
        if total_frames_to_generate is not None:
            args.total_frames_to_generate = max(1, int(total_frames_to_generate))
        if frames_per_generation is not None:
            args.frames_per_generation = max(1, int(frames_per_generation))
        if step_value is not None:
            args.num_inference_steps = max(1, int(step_value))
        if args.frames_per_generation != 8:
            raise ValueError("Astra requires 8 latent frames per denoising chunk.")
        if not image_path:
            if not images:
                raise ValueError("Astra requires image_path or images.")
            if isinstance(images, (str, Path)):
                image_path = str(images)
            else:
                raise ValueError("Astra workspace inference currently requires a filesystem image path.")
        interaction_payload = self._normalise_workspace_interaction(prompt=prompt, interactions=interactions)
        directions = interaction_payload["direction"]
        direction_count = 1 if isinstance(directions, str) else len(directions)
        if direction_count < 1:
            raise ValueError("Astra requires at least one camera direction.")
        if num_frames is not None:
            # The native count is latents per direction; the public count is
            # decoded frames for the whole video, including the condition frame.
            initial_frames = 4 * (args.initial_condition_frames - 1) + 1
            stride = 4 * direction_count
            per_direction = max(1, (int(num_frames) - initial_frames + stride - 1) // stride)
            chunk = args.frames_per_generation
            args.total_frames_to_generate = ((per_direction + chunk - 1) // chunk) * chunk
        elif args.total_frames_to_generate % args.frames_per_generation:
            raise ValueError("Astra total_frames_to_generate must be a multiple of 8 latent frames per direction.")
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))

        # Process input and interaction signals
        processed_data = self.process(str(image_path), interaction_payload, args=args)

        initial_latents = processed_data["initial_latents"]
        camera_embedding_full = processed_data["camera_embedding_full"]
        prompt_emb_pos = processed_data["prompt_emb_pos"]
        prompt_emb_neg = processed_data["prompt_emb_neg"]
        camera_embedding_uncond = processed_data["camera_embedding_uncond"]
        total_frames = processed_data["total_frames"]

        # Initialize memory with the condition frame
        self.memory.history_latents = None
        self.memory.record(initial_latents)

        total_generated = 0
        all_generated_frames = []

        # Autoregressive generation loop over all direction steps
        while total_generated < total_frames:
            # Astra predicts complete latent chunks. A shorter last denoising
            # chunk degrades its tail; generate the full chunk and crop below.
            current_generation = args.frames_per_generation
            logger.info("Generation step: %s/%s", total_generated, total_frames)

            framepack_data = self.memory.select(
                current_generation,
                camera_embedding_full,
                args.start_frame,
                args.modality_type
            )

            new_latents = self.synthesis.predict(
                framepack_data,
                current_generation,
                prompt_emb_pos,
                prompt_emb_neg,
                args,
                camera_embedding_uncond,
                generator=generator,
            )

            new_latents_squeezed = new_latents.squeeze(0)
            self.memory.record(new_latents_squeezed)

            all_generated_frames.append(new_latents_squeezed)
            total_generated += current_generation

        # Concatenate all generated frames and prepend the initial condition frame
        all_generated = torch.cat(all_generated_frames, dim=1)[:, :total_frames]
        final_video_latents = torch.cat(
            [initial_latents.to(all_generated.device), all_generated], dim=1
        ).unsqueeze(0)

        logger.info("Decoding video...")
        video_np = self.synthesis.decode_video(final_video_latents)
        if num_frames is not None:
            video_np = video_np[: int(num_frames)]
        pil_frames = [Image.fromarray(frame) for frame in video_np]
        if output_path is not None:
            from worldfoundry.core.io import write_video

            write_video(pil_frames, output_path, fps=16 if fps is None else fps)
        if return_dict:
            return {
                "status": "success",
                "model_id": "astra",
                "artifact_kind": "generated_video",
                "frames": pil_frames,
                "num_frames": len(pil_frames),
                "output_path": str(output_path) if output_path is not None else None,
            }
        return pil_frames
