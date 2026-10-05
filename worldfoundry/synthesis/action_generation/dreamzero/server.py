# Inference-only DreamZero runtime retained in-tree.
import asyncio
import dataclasses
import datetime
import http
import logging
import os
import socket
import traceback
import time

import torch
import tyro
from einops import rearrange

from .modeling.policy import GrootSimPolicy
from .preprocessing import EmbodimentTag
import imageio
import numpy as np

from . import protocol as _base_policy
from . import protocol as msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# Use roboarena policy server interface
from .websocket import WebsocketPolicyServer as RoboarenaServer
from .websocket import PolicyServerConfig

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000  # 10 hours default, configurable
    model_path: str = "./checkpoints/dreamzero"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None  # If None, use config value. Otherwise override max_chunk_size for inference.


class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.

    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Frame accumulation across calls (roboarena sends single frames, AR_droid expects multi-frame video)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination
    """

    # Number of frames to accumulate after the first call
    FRAMES_PER_CHUNK = 4

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir

        # Frame buffers for accumulation (per camera view)
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": [],
        }
        self._call_count = 0
        self._is_first_call = True

        # Session tracking - reset state when new session starts
        self._current_session_id: str | None = None

        # Video across time for saving (similar to original server)
        self.video_across_time = []
        self._msg_index = 0

        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.

        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str

        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}

        # Map image keys (roboarena uses 0-indexed, AR_droid uses 1-indexed)
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }

        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # Multiple frames (T, H, W, 3)
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        # Single frame (H, W, 3)
                        self._frame_buffers[droid_key].append(data)

        # Determine how many frames to use
        if self._is_first_call:
            # First call: use only 1 frame
            num_frames = 1
        else:
            # Subsequent calls: use exactly FRAMES_PER_CHUNK frames
            num_frames = self.FRAMES_PER_CHUNK

        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # Take the last num_frames frames
                    frames_to_use = buffer[-num_frames:]
                else:
                    # Pad by repeating the first frame to reach num_frames
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        # Prepend the first frame to pad
                        frames_to_use.insert(0, buffer[0])
                # Stack to (T, H, W, C)
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video

        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            # Reshape to (1, 7) if needed
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)

        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            # Reshape to (1, 1) if needed
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

        # Convert prompt
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""

        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.

        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)

        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None

        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value

        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)

        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()

        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)

        N = joint_action.shape[0]

        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)

        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)

        return action

    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        """Broadcast batch data from rank 0 to all other ranks."""
        serialized = msgpack_numpy.packb(obs)
        data_size = len(serialized)

        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        # Broadcast data
        data_tensor = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations.

        Args:
            obs: Observation dict in roboarena format

        Returns:
            action: (N, 8) action array
        """
        # Check for session change - reset state if new session
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                # Reset state for new session
                self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id

        self._msg_index += 1
        self._call_count += 1

        # Convert observation format
        converted_obs = self._convert_observation(obs)

        # Signal workers to continue (0 = continue)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        # Broadcast obs to workers
        self._broadcast_batch_to_workers(converted_obs)

        # Create batch for policy
        batch = Batch(obs=converted_obs)

        # Distributed forward pass
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        # Store video predictions for potential saving
        self.video_across_time.append(video_pred)

        # Extract and convert action
        action_chunk_dict = result_batch.act

        # Convert Batch to dict
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)

        action = self._convert_action(action_dict)

        # Update first call flag
        if self._is_first_call:
            self._is_first_call = False

        return action

    def _reset_state(self, save_video: bool = True) -> None:
        """Internal method to reset policy state.

        Args:
            save_video: Whether to save accumulated video before reset.
        """
        # Optionally save accumulated video before reset
        if save_video and len(self.video_across_time) > 0 and self._output_dir:
            try:
                frame_list = []
                video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                frames = self._policy.trained_model.action_head.vae.decode(
                    video_across_time_cat,
                    device=video_across_time_cat.device,
                    tiled=self._policy.trained_model.action_head.tiled,
                    tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                    tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                )
                frames = rearrange(frames, "B C T H W -> B T H W C")
                frames = frames[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                for frame in frames:
                    frame_list.append(frame)

                if len(frame_list) > 0:
                    sample_frame = frame_list[0]
                    if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                        save_dir = self._output_dir
                        os.makedirs(save_dir, exist_ok=True)
                        all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                        num_frames = len(frame_list)
                        n = (num_frames - 1) // 8
                        output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                        imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                        logger.info(f"Saved video on reset to: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save video on reset: {e}")

        # Clear frame buffers
        for key in self._frame_buffers:
            self._frame_buffers[key] = []

        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode.

        Clears frame buffers and resets call count.
        """
        self._reset_state(save_video=True)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.video_across_time = []
        self._msg_index = 0
        self._signal_group = signal_group
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)

    def _save_input_obs(self, obs: dict) -> None:
        """Save incoming observation images per message.

        Expected format: THWC (Time, Height, Width, Channel) with 4 frames.
        Saves each frame as a separate PNG image: HWC format (uint8).

        Directory structure:
        output_dir/inputs/{msg_index:06d}_{timestamp}/{obs_key}/f{frame_idx:02d}.png
        """
        if not self._output_dir:
            return
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        base_dir = os.path.join(self._output_dir, "inputs", f"{self._msg_index:06d}_{timestamp}")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            return

        for key in ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"):
            if key not in obs:
                continue
            value = obs[key]
            try:
                # Convert to numpy if tensor
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = np.asarray(value)

                # Expected format: THWC (Time, Height, Width, Channel)
                if arr.ndim != 4:
                    logger.warning(f"obs key '{key}' has shape {arr.shape}, expected 4D (T,H,W,C)")
                    continue

                # arr is (T, H, W, C)
                T, H, W, C = arr.shape

                # Normalize to uint8
                if arr.dtype == np.uint8:
                    frames_u8 = arr
                else:
                    f = arr.astype(np.float32)
                    # Common conventions: [-1,1] or [0,1]
                    min_val = float(np.nanmin(f))
                    max_val = float(np.nanmax(f))
                    if min_val >= -1.1 and max_val <= 1.1:
                        # Assume [-1,1] range
                        frames_u8 = ((f + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        # Min-max scaling
                        denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
                        frames_u8 = ((f - min_val) / denom * 255.0).clip(0, 255).astype(np.uint8)

                # Save each frame: frames_u8[i] is (H, W, C)
                key_dir = os.path.join(base_dir, key.replace("/", "_"))
                os.makedirs(key_dir, exist_ok=True)
                for frame_idx in range(T):
                    frame = frames_u8[frame_idx]  # (H, W, C)
                    # Handle grayscale (H, W) -> (H, W, 1)
                    if frame.ndim == 2:
                        frame = np.expand_dims(frame, axis=-1)
                    imageio.imwrite(os.path.join(key_dir, f"f{frame_idx:02d}.png"), frame)

            except Exception as e:
                logger.warning(f"Failed to save obs key '{key}': {e}")
                continue



    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    async def run(self, rank: int = 0):
        if rank == 0:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            # Non-rank-0 processes run a worker loop
            await self._worker_loop()

    async def _worker_loop(self):
        """Worker loop for non-rank-0 processes to participate in distributed inference."""
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                # Wait for obs broadcast from rank 0
                # Create a dummy obs dict structure - will be filled by broadcast
                # obs = {}

                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                signal = signal_tensor.item()
                if signal == 1:
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break

                # --- ADD THIS ELIF BLOCK ---
                elif signal == 2:
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    # Loop back to the top and wait for the next signal
                    continue

                # Receive the batch data via broadcast/gather mechanism
                # This is a simplified version - the actual obs structure needs to be broadcasted
                batch = self._receive_batch_from_rank0()
                # Participate in distributed forward pass
                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    def _receive_batch_from_rank0(self):
        """Receive batch data from rank 0 using torch.distributed primitives."""
        # Receive the size of the msgpack data first.
        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        # Receive the actual data
        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)

        obs = msgpack_numpy.unpackb(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    def _broadcast_batch_to_workers(self, obs):
        """Broadcast batch data from rank 0 to all other ranks."""
        serialized = msgpack_numpy.packb(obs)
        data_size = len(serialized)

        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        # Broadcast data
        data_tensor = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')

        try:
            while True:
                try:
                    start_time = time.perf_counter()
                    data = await websocket.recv()
                    recv_done = time.perf_counter()
                    obs = msgpack_numpy.unpackb(data)
                    print(f"Wait Time: {recv_done - start_time:.2f} seconds")
                    self._msg_index += 1

                    infer_start_time = time.perf_counter()

                    # Signal other ranks to continue (0 = continue)
                    signal_tensor.zero_()
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group) # <-- USE GLOO GROUP

                    # Broadcast the obs to all ranks for distributed inference
                    self._broadcast_batch_to_workers(obs)
                    batch = Batch(obs=obs)

                    # All ranks need to participate in the forward pass
                    dist.barrier()
                    forward_start_time = time.perf_counter()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()
                    print(f"Forward Time: {time.perf_counter() - forward_start_time:.2f} seconds")

                    action_chunk_dict = result_batch.act
                    video_chunk = video_pred

                    print(f"Inference Time: {time.perf_counter() - infer_start_time:.2f} seconds")

                    self.video_across_time.append(video_chunk)

                    if len(self.video_across_time) > 10:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            device=video_across_time_cat.device,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                        self.video_across_time = []
                    elif self._policy.trained_model.action_head.current_start_frame == 1 + self._policy.trained_model.action_head.num_frame_per_block and len(self.video_across_time) > 1:
                        print("current_start_frame == 1 + num_frame_per_block and len(self.video_across_time) > 1")
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time[:-1], dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            device=video_across_time_cat.device,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)
                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        self.video_across_time = [video_chunk]


                    def batch_to_dict(batch):
                        out = {}
                        for k in dir(batch):
                            if not k.startswith("action."):
                                continue
                            out[k] = getattr(batch, k)
                        return out
                    action_chunk_dict = batch_to_dict(action_chunk_dict)
                    await websocket.send(packer.pack(action_chunk_dict))

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    if len(self.video_across_time) > 0:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            device=video_across_time_cat.device,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                    self.video_across_time = []
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            logger.info(f"Rank 0: Client session ended. Sending idle signal (2) to workers.")
            signal_tensor.fill_(2)  # Set tensor value to 2
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            # When connection closes, signal other ranks to continue waiting for next connection
            # (or implement proper shutdown if needed)


def init_mesh() -> DeviceMesh:
    # env vars set by torchrun
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def main(args: Args) -> None:
    # Set environment variable for DIT cache.
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"

    # Use TE cuDNN backend for attention.
    os.environ["ATTENTION_BACKEND"] = "TE"

    # Increase the recompile limit to 100 for inference due
    # to autoregressive nature of the model (several possible shapes).
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "oxe_droid"
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    # Create server for all ranks - rank 0 handles websocket, others run worker loop
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        # Create output directory for videos
        # Extract parent directory and checkpoint name from model_path
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")

    # Create wrapper policy that converts between roboarena and AR_droid formats
    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
    )

    # Configure server for AR_droid (2 external cameras, wrist camera, joint position actions)
    server_config = PolicyServerConfig(
        image_resolution=(180, 320),  # AR_droid expects 180x320 images
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,  # Track session to reset state for new clients
        action_space="joint_position",
    )

    if rank == 0:
        logging.info("Using roboarena policy server interface")
        logging.info(f"Server config: {server_config}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        # Non-rank-0 processes need to run worker loop for distributed inference
        # We'll use the existing WebsocketPolicyServer's worker loop mechanism
        server = WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
