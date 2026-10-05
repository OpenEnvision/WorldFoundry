"""Module for base_models -> diffusion_model -> video -> lvdm -> variants -> vid2world -> lvdm -> eval_inputs -> rtvid.py functionality."""

import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import json
from torch import Tensor
from typing import Optional, List, Tuple
import math

from worldfoundry.synthesis.visual_generation.vid2world.config_paths import (
    resolve_runtime_asset,
)

class RTVid(Dataset):
    """
    RT Dataset.
    Assumes RT data is structured as follows:
    data_dir/
        train_eps_00000000.npz
        train_eps_00000001.npz
        ...
    Each .npz file contains:
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
                 mode = 'training', # training or validation
                 strong_augmentation=False,
                 val_file_list_path=None,
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
        """
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.subsample = subsample
        self.mode = mode
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
        
        # Initialize val_file_set before _load_metadata
        self.val_file_list_path = (
            str(resolve_runtime_asset(val_file_list_path)) if val_file_list_path else val_file_list_path
        )
        self.val_file_set = set()
        if self.val_file_list_path and os.path.exists(self.val_file_list_path):
            with open(self.val_file_list_path, 'r') as f:
                self.val_file_set = set(json.load(f))
        print(f'Current directory: {os.getcwd()}')
        print(f'Does val_file_list_path exist? {os.path.exists(self.val_file_list_path)}')
        print(f'val_file_list_path: {self.val_file_list_path}')
        # Load all npz files
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
        # Get all npz files in the directory
        self.file_list = []
        for i,file in enumerate(os.listdir(self.data_dir)):
            if not file.endswith('.npz'):
                continue
            if self.mode == 'training':
                if file not in self.val_file_set:
                    self.file_list.append(os.path.join(self.data_dir, file))
            elif self.mode == 'validation':
                if file in self.val_file_set:
                    self.file_list.append(os.path.join(self.data_dir, file))
        self.file_list.sort()  # Sort files for consistency
        if self.mode == 'validation':
            if len(self.file_list) > 1024:
                # Random sample from valid files
                rand_state = random.getstate()
                random.seed(0)
                valid_files = []
                for file_path in self.file_list:
                    with np.load(file_path) as data:
                        frames = data['action']
                        if len(frames) >= self.video_length:
                            valid_files.append((file_path, len(frames)))
                valid_file_paths = [x[0] for x in valid_files]
                assert len(valid_file_paths) >= 1024, f"Only {len(valid_file_paths)} files with enough frames"
                self.file_list = random.sample(valid_file_paths, 1024) # same as AVID
                random.setstate(rand_state)
        if self.subsample:
            self.file_list = self.file_list[:16] # when training, we only use 16 videos for sake of time
        print(f'{self.mode} set size: {len(self.file_list)} episode files.')
    
    def __getitem__(self, index):
        """Getitem.

        Args:
            index: The index.
        """
        # Find which file to load based on index
        file_idx = index % len(self.file_list)
        file_path = self.file_list[file_idx]
        
        # Load the npz file
        with np.load(file_path) as data:
            frames = data['image']  # [T, H, W, C] # [115, 256, 320, 3]
            actions = data['action']  # [T, M]
            # print(f'{len(frames)} frames in {file_path}')
            if len(frames) < self.video_length:
                index+=1
                return self.__getitem__(index)
            else:
                random_range = len(frames) - self.video_length
                if self.mode == 'validation':
                    random_state = random.getstate()
                    # Use file path hash as seed for consistent but different start_idx per video
                    file_hash = hash(file_path.split('/')[-1])
                    random.seed(file_hash)
                    start_idx = random.randint(0, random_range) if random_range > 0 else 0            
                    selected_frames = frames[start_idx:start_idx+self.video_length]  # [video_length, H, W, C]
                    selected_actions = actions[start_idx:start_idx+self.video_length]  # [video_length, M]
                    random.setstate(random_state)
                else:
                    start_idx = random.randint(0, random_range) if random_range > 0 else 0
                    selected_frames = frames[start_idx:start_idx+self.video_length]  # [video_length, H, W, C]
                    selected_actions = actions[start_idx:start_idx+self.video_length]  # [video_length, M]
            # Convert to torch tensor and adjust dimensions
            frames_tensor = torch.tensor(selected_frames).permute(3, 0, 1, 2).float()  # [C, T, H, W]
            actions_tensor = torch.tensor(selected_actions).float()  # [T, M]
            
            
            if self.spatial_transform is not None:
                frames_tensor = self.spatial_transform(frames_tensor)

            if self.augment_data:
                frames_tensor = self.augment_video_clip(frames_tensor)
            
            if self.resolution is not None:
                assert (frames_tensor.shape[2], frames_tensor.shape[3]) == (self.resolution[0], self.resolution[1]), \
                    f'frames={frames_tensor.shape}, self.resolution={self.resolution}'
            
            # Normalize frames to [-1, 1]
            frames_tensor = (frames_tensor / 255 - 0.5) * 2

            data = {
                'video': frames_tensor, 
                'caption': "",
                'action': actions_tensor,
                'path': file_path,
                'fps': 3, # for rt dataset, fps is fixed to 3
                'frame_stride': 1 # for rt dataset, frame_stride is fixed to 1
            }
            return data
    
    def __len__(self):
        """Len."""
        return len(self.file_list)
    
    def data_augmentation(self, images):
        """Data augmentation.

        Args:
            images: The images.
        """
        # Compute the random crop params
        i, j, h, w = self.get_crop_params(
            images,
            self.random_resized_crop_scale,
            self.random_resized_crop_ratio
        )
        # Generate the random color adjustment params
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_jittor_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue
        )

        new_images = []
        transform = transforms.ToPILImage() # Tool to convert Tensor to PIL Image
        tensor = transforms.ToTensor()      # Convert PIL Image to Tensor (will be normalized to [0, 1])
        
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
