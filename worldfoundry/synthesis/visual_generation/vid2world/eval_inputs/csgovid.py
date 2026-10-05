"""Module for base_models -> diffusion_model -> video -> lvdm -> variants -> vid2world -> eval_inputs -> csgovid.py functionality."""

import os
import random
from importlib.resources import files
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torch import Tensor
from typing import Optional, List, Tuple
import math

from worldfoundry.synthesis.visual_generation.vid2world.config_paths import (
    resolve_runtime_asset,
)
from worldfoundry.synthesis.visual_generation.vid2world.csgo_utils.data import (
    CSGOHdf5Dataset,
    SegmentId,
)


_VID2WORLD_RUNTIME_ROOT = Path(
    str(files("worldfoundry.synthesis.visual_generation.vid2world"))
)

class CSGOVID(Dataset):
    """
    CSGOVID Dataset.
    Assumes CSGOVID data is structured as follows:
    data_dir/
        1-200/
            hdf5_dm_july2021_1.hdf5
            ...
        201-400/
            hdf5_dm_july2021_201.hdf5
            ...
        ...
    Each .hdf5 file contains:
        - image: shape [T, H, W, C]
        - action: shape [T, M]
    """
    def __init__(self,
                 data_dir,
                 resolution,
                 video_length=16,
                 spatial_transform=None,
                 crop_resolution=None,
                 subsample=False,
                 augment_data=True,
                 mode = 'training',
                 strong_augmentation=False,
                 val_file_list_path=None,
                 val_start_idx_path=None,
                 use_random_action=False,
                 cond_frame=4,
                 ): # training or validation
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
            val_start_idx_path: The val start idx path.
            use_random_action: The use random action.
            cond_frame: The cond frame.
        """
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.subsample = subsample
        self.mode = mode
        self.use_random_action = use_random_action
        self.cond_frame = cond_frame
        
        self.augment_data = augment_data
        self.strong_augmentation = strong_augmentation
        if self.strong_augmentation:
            self.brightness = [0.6, 1.4]
            self.contrast = [0.6, 1.4]
            self.saturation = [0.6, 1.4]
            self.hue = [-0.5, 0.5]
            self.random_resized_crop_scale = (0.6, 1.0)
            self.random_resized_crop_ratio = (0.75, 1.3333)
        else:
            self.brightness = [0.9, 1.1]
            self.contrast = [0.9, 1.1]
            self.saturation = [0.9, 1.1]
            self.hue = [-0.05, 0.05]
            self.random_resized_crop_scale = (0.8, 1.0)
            self.random_resized_crop_ratio = (0.9, 1.1)

        self.full_dataset = CSGOHdf5Dataset(Path(data_dir))
        # Initialize val_file_set before _load_metadata
        self.val_file_list_path = val_file_list_path
        self.val_start_idx_path = val_start_idx_path
        self.val_file_set = set()
        
        if self.val_file_list_path and not os.path.isabs(self.val_file_list_path):
            self.val_file_list_path = str(
                resolve_runtime_asset(self.val_file_list_path, _VID2WORLD_RUNTIME_ROOT)
            )
        
        if self.val_file_list_path and os.path.exists(self.val_file_list_path):
            with open(self.val_file_list_path, 'r') as f: # val_file_list_path is a txt file, each line is a file path
                self.val_file_set = set(line.strip() for line in f)
        print(f'Current directory: {os.getcwd()}')
        print(f'Does val_file_list_path exist? {os.path.exists(self.val_file_list_path)}')
        print(f'val_file_list_path: {self.val_file_list_path}')
        self._load_metadata()
        
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
            elif spatial_transform == "resize_center_crop_csgo":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize((275,512)), ##[150,280] -> [275, 512] 275 ~ 150 * 512/280
                    transforms.CenterCrop(self.resolution), # [275, 512] -> [320, 512]
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
        """Helper function to load metadata."""
        # Get all hdf5 files in the directory recursively
        self.start_indices=[]
        if self.mode == 'validation':
            # Convert relative path to absolute path based on project root
            if not os.path.isabs(self.val_start_idx_path):
                val_start_idx_path = str(_VID2WORLD_RUNTIME_ROOT / self.val_start_idx_path)
            else:
                val_start_idx_path = self.val_start_idx_path
            self.start_indices = np.load(val_start_idx_path)
        self.file_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith('.hdf5'):
                    continue
                file_path = os.path.join(root, file)
                if self.mode == 'training':
                    if file not in self.val_file_set:
                        self.file_list.append(file_path)
                elif self.mode == 'validation':
                    if file in self.val_file_set:
                        self.file_list.append(file_path)
        self.file_list.sort()  # Sort files for consistency
        # print(f"file_list_length: {len(self.file_list)}")
        if self.subsample: # default is False, for debugging purpose
            self.file_list = self.file_list[:16] # when training, we only use 64 videos for sake of time
        print(f'{self.mode} set size: {len(self.file_list)} episode files.')
    
    def __getitem__(self, index):
        """Getitem.

        Args:
            index: The index.
        """
        # Find which file to load based on index
        file_idx = index % len(self.file_list)
        file_path = self.file_list[file_idx]
        if self.mode == 'training':
            start_idx = random.randint(0, 1000 - self.video_length)
        else: # on validation, we use the pre-generated start indices (see gen_csgo_start_idx.py)
            start_idx = self.start_indices[index]
        
        # Convert full path to relative path for episode_id
        relative_path = os.path.relpath(file_path, self.data_dir)
        data = self.full_dataset[SegmentId(relative_path, start_idx, start_idx+self.video_length)]
        frames_tensor = data.obs.detach().clone().permute(1, 0, 2, 3) # [T, C, H, W]
        actions_tensor = data.act.detach().clone().float() # [T, M]

        if self.spatial_transform is not None:
            frames_tensor = self.spatial_transform(frames_tensor)

        if self.augment_data:
            frames_tensor = self.augment_video_clip(frames_tensor)
        
        if self.resolution is not None:
            assert (frames_tensor.shape[2], frames_tensor.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames={frames_tensor.shape}, self.resolution={self.resolution}'
        
        # Normalize frames to [-1, 1]
        frames_tensor = (frames_tensor / 255 - 0.5) * 2

        # Use random actions if specified (keeping first cond_frame actions real)
        if self.use_random_action:
            actions_tensor = self.generate_random_actions(self.video_length, actions_tensor)

        data = {
            'video': frames_tensor, 
            'caption': "",
            'action': actions_tensor,
            'path': file_path,
            'fps': 3, # for csgo dataset, fps is fixed to 3
            'frame_stride': 1 # for rt dataset, frame_stride is fixed to 1
        }
        return data
    
    def __len__(self):
        """Len."""
        return len(self.file_list)
    
    def generate_random_actions(self, T, real_actions):
        """
        Generate random actions for T timesteps, preserving the first cond_frame actions.
        
        CSGO Action Space (51 dimensions):
        - 11 dimensions: keyboard keys (multi-hot): w, a, s, d, space, ctrl, shift, 1, 2, 3, r [RANDOMIZED]
        - 1 dimension: left click (0 or 1) [ALWAYS 0 - no click]
        - 1 dimension: right click (0 or 1) [ALWAYS 0 - no click]
        - 23 dimensions: mouse x movement (one-hot) [ALWAYS at index 11 = 0, no horizontal movement]
        - 15 dimensions: mouse y movement (one-hot) [ALWAYS at index 7 = 0, no vertical movement]
        
        Random action strategy:
        - Preserve first cond_frame actions (real from dataset)
        - For t >= cond_frame: only randomize keyboard keys, mouse is static (no clicks, no movement)
        - Keyboard: 50% chance of no key, 50% chance of one random key
        
        Note: In the model, actions are shifted by 1 timestep: [a_0, a_1, ..., a_15] -> [0, a_0, a_1, ..., a_14]
        So we preserve the first cond_frame actions which will correspond to frames 0 to cond_frame-1 after shifting.
        
        Args:
            T: number of timesteps
            real_actions: [T, 51] tensor of real actions from dataset
        
        Returns:
            actions_tensor: [T, 51] tensor with real actions for first cond_frame timesteps, 
                           random keyboard + static mouse for the rest
        """
        N_KEYS = 11
        N_CLICKS = 2
        N_MOUSE_X = 23
        N_MOUSE_Y = 15
        
        # Start with real actions
        actions_tensor = real_actions.clone()
        
        # Only randomize actions after cond_frame
        for t in range(self.cond_frame-1, T):
            # Sample ONE random key (more realistic than multi-hot)
            # 50% chance of no key, 50% chance of one random key
            if torch.rand(1).item() < 0.5:
                key_idx = torch.randint(0, N_KEYS, (1,)).item()
                keys = torch.zeros(N_KEYS)
                keys[key_idx] = 1.0
            else:
                keys = torch.zeros(N_KEYS)  # No key pressed
            
            # Mouse clicks: always 0 (never click)
            l_click = torch.zeros(1)
            r_click = torch.zeros(1)
            
            # Mouse movement: always at center (no movement)
            # mouse_x: index 11 = 0 (no horizontal movement)
            mouse_x = torch.zeros(N_MOUSE_X)
            mouse_x[11] = 1.0
            
            # mouse_y: index 7 = 0 (no vertical movement)
            mouse_y = torch.zeros(N_MOUSE_Y)
            mouse_y[7] = 1.0
            
            # Concatenate all action components and replace
            actions_tensor[t] = torch.cat([keys, l_click, r_click, mouse_x, mouse_y])
        
        return actions_tensor
    
    def data_augmentation(self, images):
        """Data augmentation.

        Args:
            images: The images.
        """
        # Compute random crop parameters
        i, j, h, w = self.get_crop_params(
            images,
            self.random_resized_crop_scale,
            self.random_resized_crop_ratio
        )
        # Generate random color adjustment parameters
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_jittor_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue
        )

        new_images = []
        transform = transforms.ToPILImage() # Tool to convert Tensor to PIL Image
        tensor = transforms.ToTensor()      # Convert PIL Image to Tensor (automatically normalizes to [0, 1])
        
        for image in images:
            image = F.resized_crop(image, i, j, h, w, self.resolution)
            image = transform(image / 255)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    image = F.adjust_brightness(image, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    image = F.adjust_contrast(image, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    image = F.adjust_saturation(image, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    image = F.adjust_hue(image, hue_factor)
            # image = F.resized_crop(image, i, j, h, w, [self.image_size, self.image_size])
            image = tensor(image)
            image = image * 255.0
            new_images.append(image)
        
        res = torch.stack(new_images)
        res = res.permute(1, 0, 2, 3)   # [C, T, H, W]
        return torch.stack(new_images)

    def augment_video_clip(self, video):
        """Augment video clip.

        Args:
            video: The video.
        """
        # video: [C, T, H, W]
        C, T, H, W = video.shape
        # print(video.shape)
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
            # Get frame t and keep it as [C, H, W]
            frame = video[:, t, :, :]  # [C, H, W]
            
            frame = F.pad(frame, [padding] * 4, padding_mode='reflect')  # Pad all sides
            frame = frame[:, i:i+H, j:j+W]  # Random crop back to original size
            
            # Convert to PIL for color augmentation (requires channels last)
            frame = transform(frame / 255)  # Convert to PIL Image
            tensor = transforms.ToTensor()
            # Color augmentation
            for fn_id in range(4):
                if fn_id == 0:
                    frame = F.adjust_brightness(frame, brightness)
                elif fn_id == 1:
                    frame = F.adjust_contrast(frame, contrast)
                elif fn_id == 2:
                    frame = F.adjust_saturation(frame, saturation)
                elif fn_id == 3:
                    frame = F.adjust_hue(frame, hue)

            # Convert back to Tensor
            frame = tensor(frame)
            frame = frame * 255.0
            new_frames.append(frame)

        return torch.stack(new_frames, dim=1)  # [C, T, H, W]
    
    @staticmethod
    def get_crop_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = F.get_dimensions(img)
        # area = height * width
        area = min(height, width) ** 2

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    @staticmethod
    def get_jittor_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h
