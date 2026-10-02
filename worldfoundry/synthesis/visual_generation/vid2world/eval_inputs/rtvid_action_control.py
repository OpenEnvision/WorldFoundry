"""
RT-1 Action Control Test Dataset

This dataset loads RT-1 episodes and creates 6 variants with different actions:
- X+, X-, Y+, Y-, Z+, Z- (world_vector, dims 10-12)

Action concatenation order (from oxe_data_converter.py):
[0:2]   base_displacement_vector
[2:3]   base_displacement_vertical_rotation
[3:4]   gripper_closedness_action
[4:7]   rotation_delta
[7:10]  terminate_episode
[10:13] world_vector <- X/Y/Z control (TESTED)

Each batch returns 6 samples with the same video but different world_vector actions.
"""
import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import json

from worldfoundry.synthesis.visual_generation.vid2world.config_paths import (
    resolve_runtime_asset,
)


class RTVidActionControl(Dataset):
    """
    RT Action Control Dataset for Fractal20220817 (RT-1).
    
    Action structure (13-dim), as concatenated in oxe_data_converter.py:
    [0:2]   base_displacement_vector (x, y)
    [2:3]   base_displacement_vertical_rotation
    [3:4]   gripper_closedness_action
    [4:7]   rotation_delta (roll, pitch, yaw)
    [7:10]  terminate_episode
    [10:13] world_vector (x, y, z) <- TESTED (dims 10-12)
    
    For each episode, generates 6 samples with different action controls:
    1. X+ (world_vector x+, dim 10)
    2. X- (world_vector x-, dim 10)
    3. Y+ (world_vector y+, dim 11)
    4. Y- (world_vector y-, dim 11)
    5. Z+ (world_vector z+, dim 12)
    6. Z- (world_vector z-, dim 12)
    """
    
    def __init__(self,
                 data_dir,
                 resolution,
                 video_length=16,
                 spatial_transform=None,
                 crop_resolution=None,
                 subsample=False,
                 augment_data=False,  # Typically False for testing
                 mode='validation',
                 strong_augmentation=False,
                 val_file_list_path=None,
                 action_magnitude=0.1,
                 action_dim=13,
                 samples_per_episode=6,  # 6 action variants: X+/-, Y+/-, Z+/-
                 single_file=True,  # Only use one episode file for action control test
                 ):
        """Init.

        Args:
            data_dir: The data dir.
            resolution: The resolution.
            video_length: The video length.
            spatial_transform: The spatial transform.
            crop_resolution: The crop resolution.
            subsample: The subsample.
            augment_data: The augment data.
            mode: The mode.
            strong_augmentation: The strong augmentation.
            val_file_list_path: The val file list path.
            action_magnitude: The action magnitude.
            action_dim: The action dim.
            samples_per_episode: The samples per episode.
            single_file: The single file.
        """
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.subsample = subsample
        self.mode = mode
        self.augment_data = augment_data
        self.strong_augmentation = strong_augmentation
        self.action_magnitude = action_magnitude
        self.action_dim = action_dim
        self.samples_per_episode = samples_per_episode
        self.single_file = single_file
        
        # Data augmentation settings (typically not used for testing)
        if self.strong_augmentation:
            self.brightness = [0.6, 1.4]
            self.contrast = [0.6, 1.4]
            self.saturation = [0.6, 1.4]
            self.hue = [-0.5, 0.5]
        else:
            self.brightness = [0.9, 1.1]
            self.contrast = [0.9, 1.1]
            self.saturation = [0.9, 1.1]
            self.hue = [-0.05, 0.05]
        
        # Load validation file list
        self.val_file_list_path = (
            str(resolve_runtime_asset(val_file_list_path)) if val_file_list_path else val_file_list_path
        )
        self.val_file_set = set()
        if self.val_file_list_path and os.path.exists(self.val_file_list_path):
            with open(self.val_file_list_path, 'r') as f:
                self.val_file_set = set(json.load(f))
        
        print(f'Current directory: {os.getcwd()}')
        print(f'Does val_file_list_path exist? {os.path.exists(self.val_file_list_path) if self.val_file_list_path else False}')
        print(f'val_file_list_path: {self.val_file_list_path}')
        
        # Load metadata
        self._load_metadata()
        
        # Setup spatial transform
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                ])
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
        
        if self.mode == 'validation':
            assert self.augment_data == False, "validation mode, augment_data should be set False"
    
    def _load_metadata(self):
        """Load all npz files from data directory"""
        self.file_list = []
        for i, file in enumerate(os.listdir(self.data_dir)):
            if not file.endswith('.npz'):
                continue
            if self.mode == 'training':
                if file not in self.val_file_set:
                    self.file_list.append(os.path.join(self.data_dir, file))
            elif self.mode == 'validation':
                if file in self.val_file_set:
                    self.file_list.append(os.path.join(self.data_dir, file))
        
        self.file_list.sort()  # Sort for consistency
        
        # Filter files with enough frames
        if self.mode == 'validation':
            valid_files = []
            for file_path in self.file_list:
                with np.load(file_path) as data:
                    frames = data['action']
                    if len(frames) >= self.video_length:
                        valid_files.append(file_path)
            self.file_list = valid_files
        
        if self.subsample:
            self.file_list = self.file_list[:16]
        
        # Use only one file for action control test
        if self.single_file and len(self.file_list) > 0:
            self.file_list = [self.file_list[0]]
            print(f'Using single file for action control test: {self.file_list[0]}')
        
        print(f'{self.mode} set size: {len(self.file_list)} episode files.')
        print(f'Each episode generates {self.samples_per_episode} action control samples')
        print(f'Total samples: {len(self.file_list) * self.samples_per_episode}')
    
    def define_test_actions(self):
        """
        Define 6 test actions for RT-1 (Fractal20220817).
        
        RT-1 action format (13-dim), concatenated as per oxe_data_converter.py:
        [0:2]   base_displacement_vector (x, y)
        [2:3]   base_displacement_vertical_rotation
        [3:4]   gripper_closedness_action
        [4:7]   rotation_delta (roll, pitch, yaw)
        [7:10]  terminate_episode
        [10:13] world_vector (x, y, z)
        
        Returns:
            List of (action_vector, action_label) tuples
        """
        actions = []
        base_action = np.zeros(self.action_dim)
        
        # Set baseline: dim 8 is often 1.0 in real data
        base_action[8] = 1.0
        
        # 1. X+ (world_vector x positive)
        action = base_action.copy()
        action[10] = self.action_magnitude  # world_vector x
        actions.append((action, 'X+'))
        
        # 2. X- (world_vector x negative)
        action = base_action.copy()
        action[10] = -self.action_magnitude  # world_vector x
        actions.append((action, 'X-'))
        
        # 3. Y+ (world_vector y positive)
        action = base_action.copy()
        action[11] = self.action_magnitude  # world_vector y
        actions.append((action, 'Y+'))
        
        # 4. Y- (world_vector y negative)
        action = base_action.copy()
        action[11] = -self.action_magnitude  # world_vector y
        actions.append((action, 'Y-'))
        
        # 5. Z+ (world_vector z positive)
        action = base_action.copy()
        action[12] = self.action_magnitude  # world_vector z
        actions.append((action, 'Z+'))
        
        # 6. Z- (world_vector z negative)
        action = base_action.copy()
        action[12] = -self.action_magnitude  # world_vector z
        actions.append((action, 'Z-'))
        
        return actions
    
    def __getitem__(self, index):
        """Getitem.

        Args:
            index: The index.
        """
        # Determine which action variant to return
        if self.single_file:
            # When using single file, index directly maps to action variant (0-7)
            episode_idx = 0
            action_variant = index % self.samples_per_episode
        else:
            # Multiple files: index // samples_per_episode gives episode, index % samples_per_episode gives action
            episode_idx = index // self.samples_per_episode
            action_variant = index % self.samples_per_episode
            # Handle wrap-around if needed
            episode_idx = episode_idx % len(self.file_list)
        
        file_path = self.file_list[episode_idx]
        
        # Load the npz file
        with np.load(file_path) as data:
            frames = data['image']  # [T, H, W, C]
            actions = data['action']  # [T, M]
            
            if len(frames) < self.video_length:
                # Skip to next valid sample
                return self.__getitem__(index + 1)
            
            # Select segment
            if self.mode == 'validation':
                # Use deterministic start index based on file hash
                random_state = random.getstate()
                file_hash = hash(file_path.split('/')[-1])
                random.seed(file_hash)
                random_range = len(frames) - self.video_length
                start_idx = random.randint(0, random_range) if random_range > 0 else 0
                random.setstate(random_state)
            else:
                random_range = len(frames) - self.video_length
                start_idx = random.randint(0, random_range) if random_range > 0 else 0
            
            selected_frames = frames[start_idx:start_idx+self.video_length]
            selected_actions = actions[start_idx:start_idx+self.video_length]
        
        # Convert to torch tensor
        frames_tensor = torch.tensor(selected_frames).permute(3, 0, 1, 2).float()  # [C, T, H, W]
        
        # Apply spatial transform
        if self.spatial_transform is not None:
            frames_tensor = self.spatial_transform(frames_tensor)
        
        # Apply data augmentation (typically False for testing)
        if self.augment_data:
            frames_tensor = self.augment_video_clip(frames_tensor)
        
        # Check resolution
        if self.resolution is not None:
            assert (frames_tensor.shape[2], frames_tensor.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames={frames_tensor.shape}, self.resolution={self.resolution}'
        
        # Normalize frames to [-1, 1]
        frames_tensor = (frames_tensor / 255 - 0.5) * 2
        
        # Generate test actions
        test_actions = self.define_test_actions()
        test_action_vector, test_action_label = test_actions[action_variant]
        
        # Create action tensor: repeat the test action for all timesteps
        actions_tensor = torch.tensor(test_action_vector).float()
        actions_tensor = actions_tensor.unsqueeze(0).repeat(self.video_length, 1)  # [T, action_dim]
        
        # Create data dict
        # Format path to ensure action label is before .npz extension
        # This is important for MetricsLogger to create separate folders for each action
        base_path = os.path.splitext(file_path)[0]  # Remove .npz
        ext = os.path.splitext(file_path)[1]  # Get .npz
        action_path = f"{base_path}_{test_action_label}{ext}"  # e.g., train_eps_00000003_X+.npz
        
        data = {
            'video': frames_tensor,
            'caption': f"{test_action_label}",  # Label with action name
            'action': actions_tensor,
            'path': action_path,
            'fps': 3,  # RT-1 fps
            'frame_stride': 1
        }
        
        return data
    
    def __len__(self):
        """Len."""
        # Return samples_per_episode * number of episodes
        if self.single_file:
            # When using single file, return only samples_per_episode samples
            return self.samples_per_episode if len(self.file_list) > 0 else 0
        else:
            return len(self.file_list) * self.samples_per_episode
    
    def augment_video_clip(self, video):
        """
        Augment video clip with color jittering and random crop
        video: [C, T, H, W]
        """
        C, T, H, W = video.shape
        assert C == 3, f"Expected 3 channels, got {C} channels"
        
        new_frames = []
        transform = transforms.ToPILImage()
        brightness = random.uniform(*self.brightness)
        contrast = random.uniform(*self.contrast)
        saturation = random.uniform(*self.saturation)
        hue = random.uniform(*self.hue)
        padding = 2
        i = random.randint(0, 2 * padding)
        j = random.randint(0, 2 * padding)
        
        for t in range(T):
            frame = video[:, t, :, :]  # [C, H, W]
            
            # Random crop
            frame = F.pad(frame, [padding] * 4, padding_mode='reflect')
            frame = frame[:, i:i+H, j:j+W]
            
            # Color augmentation
            frame = transform(frame / 255)
            tensor = transforms.ToTensor()
            
            for fn_id in range(4):
                if fn_id == 0:
                    frame = F.adjust_brightness(frame, brightness)
                elif fn_id == 1:
                    frame = F.adjust_contrast(frame, contrast)
                elif fn_id == 2:
                    frame = F.adjust_saturation(frame, saturation)
                elif fn_id == 3:
                    frame = F.adjust_hue(frame, hue)
            
            frame = tensor(frame)
            frame = frame * 255.0
            new_frames.append(frame)
        
        return torch.stack(new_frames, dim=1)  # [C, T, H, W]
