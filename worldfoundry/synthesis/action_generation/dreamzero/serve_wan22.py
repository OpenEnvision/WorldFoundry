# Inference-only DreamZero runtime retained in-tree.
"""
Serve the DreamZero 5B implementation (Wan2.2-TI2V-5B) over the websocket policy server.

This is the 5B model: Wan2.2 diffusion backbone, 48-channel VAE38, frame_seqlen=50 (160×320
latent 10×20). Inference is causal with KV caching: first request in a session uses 1 frame
and warms the cache; subsequent requests use FRAMES_PER_CHUNK=4 frames and append to the cache.
On session_id change (or explicit reset), buffers and action_head.current_start_frame are cleared.

The checkpoint at model_path should be DreamZero with Wan22 5B (model/dreamzero/action_head=
wan_flow_matching_action_tf_wan22, data droid_relative_wan22 → 160×320). GrootSimPolicy loads
that checkpoint and runs inference; it is the correct policy class for DreamZero.

Usage (single GPU):

  torchrun --nproc_per_node=1 -m worldfoundry.synthesis.action_generation.dreamzero.serve_wan22 --model_path ./checkpoints/dreamzero_droid_wan22_validation --port 8000

  # Or single process:
  python -m worldfoundry.synthesis.action_generation.dreamzero.serve_wan22 --model_path ./checkpoints/dreamzero_droid_wan22_validation --port 8000

Client: send observations per PolicyServerConfig (policy_server.py). Video is resized to the
checkpoint's expected resolution (e.g. 180×320) so the eval transform accepts it; the 5B action
head resizes to 160×320 internally. Override with --image_height/--image_width if needed.
Response is an action chunk (N, 8). Use session_id for episode boundaries.
"""

import datetime
import logging
import os

import imageio

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import tyro

# Avoid FailOnRecompileLimitHit when serving: the flow scheduler's torch.compile'd
# multistep_uni_p_bh_update recompiles under varying shapes/inputs (e.g. batch size,
# step_index, order). Increase limits so the server doesn't hit the default cap.
_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000
from tianshou.data import Batch

from .protocol import BasePolicy

from .websocket import WebsocketPolicyServer, PolicyServerConfig
from .modeling.policy import GrootSimPolicy
from .preprocessing import EmbodimentTag, ComposedModalityTransform


# DreamZero Wan 5B is trained with 160×320 (droid_relative_wan22). Fallback if we cannot read from policy.
DEFAULT_IMAGE_HEIGHT = 160
DEFAULT_IMAGE_WIDTH = 320
FRAMES_PER_CHUNK = 4  # matches 5B num_frame_per_block for causal chunked inference


def _get_expected_video_resolution(policy: GrootSimPolicy) -> tuple[int, int]:
    """Get (height, width) the policy's eval_transform expects for video (from checkpoint
    metadata). Resolution in metadata is (width, height); we return (height, width) for resize.
    DreamZero Wan 5B (droid_relative_wan22) uses 160×320; other configs may use e.g. 180×320.
    """
    eval_transform = getattr(policy, "eval_transform", None)
    if eval_transform is None:
        return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    if not isinstance(eval_transform, ComposedModalityTransform):
        return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    for t in eval_transform.transforms:
        if hasattr(t, "original_resolutions") and getattr(t, "original_resolutions", None):
            res = t.original_resolutions
            if res:
                # original_resolutions values are (width, height)
                w, h = next(iter(res.values()))
                return (int(h), int(w))
    return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)


def _resize_frames_to_resolution(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize video frames to (target_h, target_w). Accepts (H,W,C) or (T,H,W,C)."""
    if frames.ndim == 3:
        if (frames.shape[0], frames.shape[1]) != (target_h, target_w):
            frames = cv2.resize(frames, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return frames
    out = np.stack(
        [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in frames],
        axis=0,
    )
    return out


def _maybe_init_distributed():
    """Initialize process group for single-GPU or multi-GPU. Required by GrootSimPolicy."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


# Modality key mappings: client observation keys -> model input keys per embodiment.
# Client sends: observation/exterior_image_0_left, exterior_image_1_left, wrist_image_left.
VIDEO_KEY_MAPPING = {
    "oxe_droid": {
        "observation/exterior_image_0_left": "video.exterior_image_1_left",
        "observation/exterior_image_1_left": "video.exterior_image_2_left",
        "observation/wrist_image_left": "video.wrist_image_left",
    },
}
STATE_KEY_MAPPING = {
    "oxe_droid": ("state.joint_position", "state.gripper_position"),
}
LANGUAGE_KEY_MAPPING = {
    "oxe_droid": "annotation.language.action_text",
}


class DreamZeroWan225BPolicy(BasePolicy):
    """
    Wraps GrootSimPolicy for the DreamZero 5B implementation (Wan2.2-TI2V-5B).

    Converts roboarena observation/action format to DROID/Batch. Video is resized to the
    resolution expected by the policy's eval_transform (from checkpoint metadata) so
    VideoToTensor validation passes. The 5B action head then resizes to 160×320 internally.
    First call in a session uses 1 frame; later calls use 4 frames (FRAMES_PER_CHUNK).
    Session reset clears frame buffers and action_head.current_start_frame.
    """

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        image_height: int,
        image_width: int,
        embodiment_tag: str = "oxe_droid",
        save_video_pred: bool = False,
        video_output_dir: str = "./video_pred_output",
    ):
        super().__init__()
        self._policy = groot_policy
        self._image_height = image_height
        self._image_width = image_width
        self._embodiment_tag = (
            embodiment_tag if embodiment_tag in VIDEO_KEY_MAPPING else "oxe_droid"
        )
        video_keys = list(VIDEO_KEY_MAPPING[self._embodiment_tag].values())
        self._frame_buffers = {k: [] for k in video_keys}
        self._is_first_call = True
        self._current_session_id = None
        self._save_video_pred = save_video_pred
        self._video_output_dir = video_output_dir
        self._video_pred_latents: list[torch.Tensor] = []
        self._current_prompt: str = ""

    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to model Batch format.
        Incoming frames are resized to the policy's expected (height, width) so
        eval_transform's VideoToTensor check passes.
        """
        image_key_mapping = VIDEO_KEY_MAPPING[self._embodiment_tag]
        for roboarena_key, model_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    data = _resize_frames_to_resolution(
                        data, self._image_height, self._image_width
                    )
                    if data.ndim == 4:
                        self._frame_buffers[model_key].extend(list(data))
                    else:
                        self._frame_buffers[model_key].append(data)

        num_frames = 1 if self._is_first_call else FRAMES_PER_CHUNK
        converted = {}
        for model_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                video = np.stack(frames_to_use, axis=0)
                converted[model_key] = video

        state_joint_key, state_gripper_key = STATE_KEY_MAPPING[self._embodiment_tag]
        if "observation/joint_position" in obs:
            joint_pos = np.asarray(obs["observation/joint_position"])
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted[state_joint_key] = joint_pos.astype(np.float64)
        else:
            converted[state_joint_key] = np.zeros((1, 7), dtype=np.float64)

        if "observation/gripper_position" in obs:
            gripper_pos = np.asarray(obs["observation/gripper_position"])
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted[state_gripper_key] = gripper_pos.astype(np.float64)
        else:
            converted[state_gripper_key] = np.zeros((1,1), dtype=np.float64)

        text_prompt = obs.get("prompt", "")
        logger.info("Text prompt: %s", text_prompt)
        if text_prompt:
            self._current_prompt = text_prompt
        lang_key = LANGUAGE_KEY_MAPPING[self._embodiment_tag]
        converted[lang_key] = text_prompt
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert model action dict to (N, 8) array (7 joint + 1 gripper)."""
        joint_action = None
        gripper_action = None
        for key, value in action_dict.items():
            if ("joint_position" in key or "joint_pos" in key) and "gripper" not in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        if joint_action is None:
            return np.zeros((1, 8), dtype=np.float32)
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        N = joint_action.shape[0]
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            if gripper_action.shape[-1] > 1:
                gripper_action = gripper_action[..., :1]
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        return np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)

    def infer(self, obs: dict) -> np.ndarray:
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                self.reset({})
            self._current_session_id = session_id

        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        if self._save_video_pred and video_pred is not None:
            self._video_pred_latents.append(video_pred.detach())
        action_dict = {}
        action_chunk_dict = result_batch.act
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        action = self._convert_action(action_dict)
        if self._is_first_call:
            self._is_first_call = False
        return action

    def _save_predicted_video(self) -> None:
        """Decode accumulated video prediction latents through the VAE and save as mp4."""
        if not self._video_pred_latents:
            return
        try:
            from einops import rearrange

            action_head = self._policy.trained_model.action_head
            latents = torch.cat(self._video_pred_latents, dim=2)
            with torch.no_grad():
                frames = action_head.vae.decode(
                    latents,
                    device=latents.device,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

            os.makedirs(self._video_output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            n_latent_frames = latents.shape[2]
            existing = [f for f in os.listdir(self._video_output_dir) if f.endswith(".mp4")]
            safe_prompt = self._current_prompt.replace(" ", "_")
            safe_prompt = "".join(c for c in safe_prompt if c.isalnum() or c in "_-.")
            if len(safe_prompt) > 80:
                safe_prompt = safe_prompt[:80]
            if not safe_prompt:
                safe_prompt = "no_prompt"
            output_path = os.path.join(
                self._video_output_dir,
                f"{len(existing):06}_{safe_prompt}_{timestamp}.mp4",
            )
            imageio.mimsave(output_path, list(frames), fps=5, codec="libx264")
            logger.info("Saved video prediction (%d frames) to %s", len(frames), output_path)
        except Exception as e:
            logger.warning("Failed to save video prediction: %s", e)

    def reset(self, reset_info: dict) -> None:
        if self._save_video_pred:
            self._save_predicted_video()
        self._video_pred_latents.clear()
        self._current_prompt = ""
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        self._current_session_id = None
        if hasattr(self._policy.trained_model, "action_head") and hasattr(
            self._policy.trained_model.action_head, "current_start_frame"
        ):
            self._policy.trained_model.action_head.current_start_frame = 0


def main(
    model_path: str = "./checkpoints/dreamzero_droid_wan22_validation",
    embodiment_tag: str = "oxe_droid",
    tokenizer_path: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
    image_height: int | None = None,
    image_width: int | None = None,
    save_video_pred: bool = False,
    video_output_dir: str = "./video_pred_output",
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    _maybe_init_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    logger.info("Loading DreamZero Wan22 policy from %s (embodiment=%s)", model_path, embodiment_tag)
    checkpoint_name = os.path.basename(model_path.rstrip("/"))
    video_output_dir = os.path.join(video_output_dir, checkpoint_name)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        tokenizer_path_override=tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    if image_height is not None and image_width is not None:
        h, w = image_height, image_width
        logger.info("Using CLI video resolution: %dx%d", h, w)
    else:
        h, w = _get_expected_video_resolution(policy)
        logger.info("Using checkpoint video resolution: %dx%d (HxW)", h, w)
    wrapper = DreamZeroWan225BPolicy(
        groot_policy=policy,
        image_height=h,
        image_width=w,
        embodiment_tag=embodiment_tag,
        save_video_pred=save_video_pred,
        video_output_dir=video_output_dir,
    )

    server_config = PolicyServerConfig(
        image_resolution=(h, w),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )
    logger.info("Starting WebsocketPolicyServer on %s:%d (DreamZero 5B, %dx%d)", host, port, h, w)
    server = WebsocketPolicyServer(
        policy=wrapper,
        server_config=server_config,
        host=host,
        port=port,
    )
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
