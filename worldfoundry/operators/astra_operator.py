"""Module for the Astra operator implementation."""

import logging

import torch
import numpy as np
from PIL import Image
from einops import rearrange
import json
import imageio
from pathlib import Path
from torchvision.transforms import v2

from .base_operator import BaseOperator

logger = logging.getLogger(__name__)


IMAGE_CONDITION_FRAME_COUNT = 10
FRAMEPACK_CONTEXT_FRAMES = 1 + 16 + 2 + 1
SEKAI_SYNTHETIC_DIRECTIONS = {
    "camera_l",
    "camera_r",
    "forward",
    "backward",
    "forward_left",
    "forward_right",
    "s_curve",
    "left_right",
}


class AstraOperator(BaseOperator):
    """Operator class for the Astra model integration."""
    def __init__(self, device="cuda"):
        """Initialize the operator with specific configurations."""
        super().__init__(operation_types=["visual_instruction", "action_instruction"])
        self.device = device
        self.interaction_template = sorted(SEKAI_SYNTHETIC_DIRECTIONS)
        self.interaction_template_init()

        self.frame_process = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def check_interaction(self, interaction):
        """
        检验输入是否符合astra输入格式
        """
        if interaction not in self.interaction_template:
            raise ValueError(f"Invalid interaction: {interaction}. Allowed: {self.interaction_template}")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, modality_type="sekai", start_frame=0, initial_condition_frames=1, total_frames_to_generate=8, use_real_poses=False, scene_info_path=None, encoded_data=None):
        """
        根据当前的 interaction (direction) 生成对应的 Camera Embedding。
        """
        direction = self.current_interaction[-1] if self.current_interaction else "forward"
        model_dtype = torch.bfloat16 
        
        scene_info = None
        if modality_type == "nuscenes" and scene_info_path:
            scene_path = Path(scene_info_path).expanduser()
            if scene_path.exists():
                scene_info = json.loads(scene_path.read_text(encoding="utf-8"))

        if modality_type == "sekai":
            camera_embedding_full = self.generate_sekai_camera_embeddings_sliding(
                encoded_data.get('cam_emb', None) if encoded_data else None,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate,
                0,
                use_real_poses=use_real_poses,
                direction=direction
            )
        elif modality_type == "nuscenes":
            camera_embedding_full = self.generate_nuscenes_camera_embeddings_sliding(
                scene_info,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate
            )
        elif modality_type == "openx":
            camera_embedding_full = self.generate_openx_camera_embeddings_sliding(
                encoded_data,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate,
                use_real_poses=use_real_poses
            )
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
            
        return camera_embedding_full.to(self.device, dtype=model_dtype)

    def process_perception(self, condition_video=None, condition_image=None):
        """
        加载图像或视频文件，预处理为 Tensor Frames，供 Synthesis 编码使用。
        """
        frames = None
        
        if condition_video:
            video_path = Path(condition_video).expanduser().resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"File not Found: {video_path}")
                
            reader = imageio.get_reader(str(video_path))
            frame_list = []
            for frame_data in reader:
                frame = Image.fromarray(frame_data)
                # Crop and Resize
                frame = self._crop_and_resize(frame)
                frame_list.append(self.frame_process(frame))
            reader.close()
            
            if frame_list:
                frames = torch.stack(frame_list, dim=0)
                frames = rearrange(frames, "T C H W -> C T H W")
                
        elif condition_image:
            image_path = Path(condition_image).expanduser().resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"File not Found: {image_path}")
            
            image = Image.open(str(image_path)).convert("RGB")
            image = self._crop_and_resize(image)
            frame = self.frame_process(image)
            frames = torch.stack([frame for _ in range(IMAGE_CONDITION_FRAME_COUNT)], dim=0)
            frames = rearrange(frames, "T C H W -> C T H W")
        
        return frames

    @staticmethod
    def _crop_and_resize(image: Image.Image) -> Image.Image:
        """Crop and resize implementation."""
        target_w, target_h = 832, 480
        return v2.functional.resize(
            image,
            (round(target_h), round(target_w)),
            interpolation=v2.InterpolationMode.BILINEAR,
        )

    def compute_relative_pose(self, pose_a, pose_b, use_torch=False):
        """Compute relative pose matrix of camera B with respect to camera A"""
        assert pose_a.shape == (4, 4), f"Camera A extrinsic matrix should be (4,4), got {pose_a.shape}"
        assert pose_b.shape == (4, 4), f"Camera B extrinsic matrix should be (4,4), got {pose_b.shape}"
        if use_torch:
            if not isinstance(pose_a, torch.Tensor): pose_a = torch.from_numpy(pose_a).float()
            if not isinstance(pose_b, torch.Tensor): pose_b = torch.from_numpy(pose_b).float()
            pose_a_inv = torch.inverse(pose_a)
            relative_pose = torch.matmul(pose_b, pose_a_inv)
        else:
            if not isinstance(pose_a, np.ndarray): pose_a = np.array(pose_a, dtype=np.float32)
            if not isinstance(pose_b, np.ndarray): pose_b = np.array(pose_b, dtype=np.float32)
            pose_a_inv = np.linalg.inv(pose_a)
            relative_pose = np.matmul(pose_b, pose_a_inv)
        return relative_pose

    @staticmethod
    def _needed_frame_count(start_frame, initial_condition_frames, new_frames):
        """Needed frame count implementation."""
        framepack_needed_frames = FRAMEPACK_CONTEXT_FRAMES + new_frames
        return max(start_frame + initial_condition_frames + new_frames, framepack_needed_frames, 30)

    @staticmethod
    def _camera_embedding_from_rows(rows, start_frame, condition_end):
        """Camera embedding from rows implementation."""
        pose_embedding = torch.stack(rows, dim=0)
        if pose_embedding.dim() == 3:
            pose_embedding = rearrange(pose_embedding, "b c d -> b (c d)")
        mask = torch.zeros(pose_embedding.shape[0], 1, dtype=torch.float32)
        mask[start_frame:min(condition_end, pose_embedding.shape[0])] = 1.0
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        return camera_embedding.to(torch.bfloat16)

    @staticmethod
    def _relative_pose_rows_from_extrinsics(cam_extrinsic, max_needed_frames, time_compression_ratio):
        """Relative pose rows from extrinsics implementation."""
        relative_poses = []
        for i in range(max_needed_frames):
            frame_idx = i * time_compression_ratio
            next_frame_idx = frame_idx + time_compression_ratio
            if next_frame_idx < len(cam_extrinsic):
                cam_prev = cam_extrinsic[frame_idx]
                cam_next = cam_extrinsic[next_frame_idx]
                relative_pose = AstraOperator.compute_relative_pose(cam_prev, cam_next)
                relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
            else:
                relative_poses.append(torch.zeros(3, 4))
        return relative_poses

    @staticmethod
    def _apply_yaw(pose, yaw_per_frame):
        """Apply yaw implementation."""
        pose[0, 0] = np.cos(yaw_per_frame)
        pose[0, 2] = np.sin(yaw_per_frame)
        pose[2, 0] = -np.sin(yaw_per_frame)
        pose[2, 2] = np.cos(yaw_per_frame)

    @classmethod
    def _sekai_synthetic_pose(cls, frame_idx, condition_frames, stage_1, stage_2, direction):
        """Sekai synthetic pose implementation."""
        active_start = condition_frames
        active_end = condition_frames + stage_1 + stage_2
        pose = np.eye(4, dtype=np.float32)
        if frame_idx < active_start or frame_idx >= active_end:
            return pose[:3, :]
        if direction == "forward":
            pose[2, 3] = -0.03
        elif direction == "backward":
            pose[2, 3] = 0.03
        elif direction == "camera_l":
            cls._apply_yaw(pose, 0.03)
        elif direction == "camera_r":
            cls._apply_yaw(pose, -0.03)
        elif direction == "forward_left":
            cls._apply_yaw(pose, 0.03)
            pose[2, 3] = -0.03
        elif direction == "forward_right":
            cls._apply_yaw(pose, -0.03)
            pose[2, 3] = -0.03
        elif direction == "s_curve":
            yaw_per_frame = 0.03 if frame_idx < condition_frames + stage_1 else -0.03
            cls._apply_yaw(pose, yaw_per_frame)
            pose[2, 3] = -0.03
            if condition_frames + stage_1 <= frame_idx < condition_frames + stage_1 + stage_2 // 3:
                pose[0, 3] = -0.01
        elif direction == "left_right":
            yaw_per_frame = 0.03 if frame_idx < condition_frames + stage_1 else -0.03
            cls._apply_yaw(pose, yaw_per_frame)
        else:
            raise ValueError(f"Unsupported Sekai synthetic direction: {direction}")
        return pose[:3, :]

    @staticmethod
    def _openx_synthetic_pose():
        """Openx synthetic pose implementation."""
        roll_per_frame = 0.02
        pitch_per_frame = 0.01
        yaw_per_frame = 0.015
        forward_speed = 0.003
        pose = np.eye(4, dtype=np.float32)
        cos_roll = np.cos(roll_per_frame)
        sin_roll = np.sin(roll_per_frame)
        cos_pitch = np.cos(pitch_per_frame)
        sin_pitch = np.sin(pitch_per_frame)
        cos_yaw = np.cos(yaw_per_frame)
        sin_yaw = np.sin(yaw_per_frame)
        pose[0, 0] = cos_yaw * cos_pitch
        pose[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
        pose[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
        pose[1, 0] = sin_yaw * cos_pitch
        pose[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
        pose[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
        pose[2, 0] = -sin_pitch
        pose[2, 1] = cos_pitch * sin_roll
        pose[2, 2] = cos_pitch * cos_roll
        pose[0, 3] = forward_speed * 0.5
        pose[1, 3] = forward_speed * 0.3
        pose[2, 3] = -forward_speed
        return pose[:3, :]

    def generate_sekai_camera_embeddings_sliding(self, cam_data, start_frame, initial_condition_frames, new_frames, total_generated, use_real_poses=True, direction="left"):
        """Generate camera embeddings for Sekai dataset - sliding window version"""
        time_compression_ratio = 4
        max_needed_frames = self._needed_frame_count(start_frame, initial_condition_frames, new_frames)
        
        if use_real_poses and cam_data is not None and 'extrinsic' in cam_data:
            logger.info("Using real Sekai camera data")
            cam_extrinsic = cam_data['extrinsic']
            relative_poses = self._relative_pose_rows_from_extrinsics(
                cam_extrinsic,
                max_needed_frames,
                time_compression_ratio,
            )
            return self._camera_embedding_from_rows(
                relative_poses,
                start_frame,
                start_frame + initial_condition_frames,
            )
        else:
            logger.info("Generating Sekai synthetic camera frames: %s, direction: %s", max_needed_frames, direction)
            stage_1 = new_frames // 2
            stage_2 = new_frames - stage_1
            relative_poses = [
                torch.as_tensor(
                    self._sekai_synthetic_pose(
                        i,
                        initial_condition_frames,
                        stage_1,
                        stage_2,
                        direction,
                    )
                )
                for i in range(max_needed_frames)
            ]
            return self._camera_embedding_from_rows(
                relative_poses,
                start_frame,
                start_frame + initial_condition_frames + 1,
            )

    def generate_openx_camera_embeddings_sliding(self, encoded_data, start_frame, initial_condition_frames, new_frames, use_real_poses):
        """Generate camera embeddings for OpenX dataset"""
        time_compression_ratio = 4
        max_needed_frames = self._needed_frame_count(start_frame, initial_condition_frames, new_frames)
        
        if use_real_poses and encoded_data is not None and 'cam_emb' in encoded_data and 'extrinsic' in encoded_data['cam_emb']:
            logger.info("Using OpenX real camera data")
            cam_extrinsic = encoded_data['cam_emb']['extrinsic']
            relative_poses = self._relative_pose_rows_from_extrinsics(
                cam_extrinsic,
                max_needed_frames,
                time_compression_ratio,
            )
            return self._camera_embedding_from_rows(
                relative_poses,
                start_frame,
                start_frame + initial_condition_frames,
            )
        else:
            logger.info("Using OpenX synthetic camera data")
            relative_poses = [
                torch.as_tensor(self._openx_synthetic_pose())
                for _ in range(max_needed_frames)
            ]
            return self._camera_embedding_from_rows(
                relative_poses,
                start_frame,
                start_frame + initial_condition_frames,
            )

    def generate_nuscenes_camera_embeddings_sliding(self, scene_info, start_frame, initial_condition_frames, new_frames):
        """Generate camera embeddings for NuScenes dataset"""
        max_needed_frames = max(FRAMEPACK_CONTEXT_FRAMES + new_frames, 30)
        
        if scene_info is not None and 'keyframe_poses' in scene_info:
            logger.info("Using NuScenes real pose data")
            keyframe_poses = scene_info['keyframe_poses']
            if len(keyframe_poses) == 0:
                pose_sequence = torch.zeros(max_needed_frames, 7, dtype=torch.float32)
                return self._camera_embedding_from_rows(
                    [row for row in pose_sequence],
                    start_frame,
                    start_frame + initial_condition_frames,
                )
            
            reference_pose = keyframe_poses[0]
            pose_vecs = []
            for i in range(max_needed_frames):
                if i < len(keyframe_poses):
                    current_pose = keyframe_poses[i]
                    translation = torch.tensor(np.array(current_pose['translation']) - np.array(reference_pose['translation']), dtype=torch.float32)
                    rotation = torch.tensor(current_pose['rotation'], dtype=torch.float32)
                    pose_vec = torch.cat([translation, rotation], dim=0)
                else:
                    pose_vec = torch.cat([torch.zeros(3, dtype=torch.float32), torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)], dim=0)
                pose_vecs.append(pose_vec)
            return self._camera_embedding_from_rows(
                pose_vecs,
                start_frame,
                start_frame + initial_condition_frames,
            )
        else:
            logger.info("Using NuScenes synthetic pose data")
            pose_vecs = []
            for i in range(max_needed_frames):
                angle = i * 0.04; radius = 15.0
                x = radius * np.sin(angle); y = 0.0; z = radius * (1 - np.cos(angle))
                translation = torch.tensor([x, y, z], dtype=torch.float32)
                yaw = angle + np.pi/2
                rotation = torch.tensor([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)], dtype=torch.float32)
                pose_vec = torch.cat([translation, rotation], dim=0)
                pose_vecs.append(pose_vec)
            return self._camera_embedding_from_rows(
                pose_vecs,
                start_frame,
                start_frame + initial_condition_frames,
            )
