"""Module for base_models -> diffusion_model -> video -> lvdm -> variants -> vid2world -> lvdm -> eval_inputs -> valorant.py functionality."""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from typing import Optional, List, Tuple
import glob

class CSGOVIDStyleDataset(Dataset):
    """
    Csgo-style Dataset, with start images loaded from local file.
    Some typical examples of such games are: Valorant and Delta Force.
    Assumes Valorant data is structured as follows:
    data_dir/
        0.png
        1.png
        2.png
        ...
    Each .png file corresponds to the start of a trajectory.
    For each sample, we return:
    - video: [C, T, H, W] where T frames are all the same (repeated from the input image)
    - action: [T, 51] where all actions are static (no movement, no clicks, center mouse)
    """
    def __init__(self,
                 data_dir,
                 resolution=[320, 512],
                 video_length=16,
                 spatial_transform=None,
                 augment_data=False,
                 mode='validation',
                 ):
        """Init.

        Args:
            data_dir: The data dir.
            resolution: The resolution.
            video_length: The video length.
            spatial_transform: The spatial transform.
            augment_data: The augment data.
            mode: The mode.
        """
        # Convert relative path to absolute path based on project root
        if not os.path.isabs(data_dir):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
            data_dir = os.path.join(project_root, data_dir)
        
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.mode = mode
        self.augment_data = augment_data
        
        # Load all PNG files
        self.image_files = sorted(glob.glob(os.path.join(data_dir, '*.png')), 
                                  key=lambda x: int(os.path.basename(x).split('.')[0]))
        print(f'{self.mode} set size: {len(self.image_files)} images.')
        
        # Setup spatial transform (same as CSGO)
        if spatial_transform is not None:
            if spatial_transform == "resize_center_crop_csgo":
                # Resize to (275, 512), then pad to (320, 512)
                self.spatial_transform = "resize_pad_csgo"
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError(f"spatial_transform {spatial_transform} not implemented")
        else:
            self.spatial_transform = None
        
        if self.mode == 'validation':
            assert self.augment_data == False, "validation mode, augment_data should be set False"
    
    def __len__(self):
        """Len."""
        return len(self.image_files)
    
    def _resize_and_pad(self, image_tensor):
        """
        Resize image to (275, 512), then pad to (320, 512).
        This mimics the CSGO resize_center_crop_csgo transform.
        image_tensor: [C, H, W] in range [0, 255]
        Returns: [C, 320, 512] in range [0, 255]
        """
        # Resize to (275, 512)
        image_tensor = F.resize(image_tensor, (275, 512))
        
        # Pad to (320, 512) - pad top and bottom with black pixels
        # Current size: [C, 275, 512], target: [C, 320, 512]
        # F.pad padding order: [left, top, right, bottom]
        pad_top = (320 - 275) // 2  # 22
        pad_bottom = 320 - 275 - pad_top  # 23
        image_tensor = F.pad(image_tensor, [0, pad_top, 0, pad_bottom], fill=0, padding_mode='constant')
        
        return image_tensor
    
    def _generate_static_action(self, T):
        """
        Generate static action for T timesteps.
        Action space (51 dimensions):
        - 11 dimensions: keyboard keys (all zeros - no keys pressed)
        - 1 dimension: left click (0)
        - 1 dimension: right click (0)
        - 23 dimensions: mouse x movement (one-hot at index 11 = 0, center)
        - 15 dimensions: mouse y movement (one-hot at index 7 = 0, center)
        
        Returns: [T, 51] tensor
        """
        N_KEYS = 11
        N_CLICKS = 2
        N_MOUSE_X = 23
        N_MOUSE_Y = 15
        
        # Create one static action
        keys = torch.zeros(N_KEYS)
        l_click = torch.zeros(1)
        r_click = torch.zeros(1)
        mouse_x = torch.zeros(N_MOUSE_X)
        mouse_x[11] = 1.0  # Center position (index 11 = 0)
        mouse_y = torch.zeros(N_MOUSE_Y)
        mouse_y[7] = 1.0  # Center position (index 7 = 0)
        
        static_action = torch.cat([keys, l_click, r_click, mouse_x, mouse_y])
        
        # Repeat for T timesteps
        actions_tensor = static_action.unsqueeze(0).repeat(T, 1)  # [T, 51]
        
        return actions_tensor
    
    def __getitem__(self, index):
        """Getitem.

        Args:
            index: The index.
        """
        # Load image
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')
        
        # Convert to tensor [C, H, W] in range [0, 255]
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image) * 255.0  # [C, H, W], range [0, 255]
        
        # Apply spatial transform
        if self.spatial_transform == "resize_pad_csgo":
            image_tensor = self._resize_and_pad(image_tensor)
        elif self.spatial_transform is not None:
            image_tensor = self.spatial_transform(image_tensor)
        
        # Verify resolution
        if self.resolution is not None:
            assert (image_tensor.shape[1], image_tensor.shape[2]) == (self.resolution[0], self.resolution[1]), \
                f'image={image_tensor.shape}, self.resolution={self.resolution}'
        
        # Repeat frame for video_length frames: [C, H, W] -> [C, T, H, W]
        frames_tensor = image_tensor.unsqueeze(1).repeat(1, self.video_length, 1, 1)  # [C, T, H, W]
        
        # Normalize frames to [-1, 1] (same as CSGO)
        frames_tensor = (frames_tensor / 255.0 - 0.5) * 2.0
        
        # Generate static actions
        actions_tensor = self._generate_static_action(self.video_length)  # [T, 51]
        
        data = {
            'video': frames_tensor,  # [C, T, H, W]
            'caption': "",
            'action': actions_tensor,  # [T, 51]
            'path': image_path,
            'fps': 3,  # Same as CSGO
            'frame_stride': 1  # Same as CSGO
        }
        return data
