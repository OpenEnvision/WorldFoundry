# adapted from https://github.com/facebookresearch/nwm/tree/main
"""Module for base_models -> diffusion_model -> video -> lvdm -> variants -> vid2world -> eval_inputs -> reconvid.py functionality."""

import os
import random
from importlib.resources import files

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
from worldfoundry.synthesis.visual_generation.vid2world.nvm_utils.eval_inputs import (
    RECONEvalDataset,
    RECONVIDDataset,
)
from worldfoundry.synthesis.visual_generation.vid2world.nvm_utils.misc import (
    transform,
)


_VID2WORLD_RUNTIME_ROOT = str(
    files("worldfoundry.synthesis.visual_generation.vid2world")
)

class RECONVID(Dataset):
    """
    RECON Dataset.
    Assumes RECON data is structured as follows:
    data_dir/
        jackal_2019-10-24-15-23-52_13_r01/
            traj_data.pkl
            0.jpg
            1.jpg
            ...
        jackal_2019-10-24-15-23-52_13_r02/
        ...
    Each traj_data.pkl file contains:
        - position: shape [T, 2]
        - yaw: shape [T,]
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
                 data_split_folder=None,
                 train_predefined_index=None,
                 val_predefined_index=None,
                 processed_data_dir=None,
                 skip_processed=False,
                 exp=None,
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
            data_split_folder: The data split folder.
            train_predefined_index: The train predefined index.
            val_predefined_index: The val predefined index.
            processed_data_dir: The processed data dir.
            skip_processed: The skip processed.
            exp: The exp.
        """
        self.data_dir = data_dir
        self.exp = exp
        
        # Convert relative path to absolute path based on project root
        if data_split_folder and not os.path.isabs(data_split_folder):
            self.data_split_folder = os.path.join(_VID2WORLD_RUNTIME_ROOT, data_split_folder)
        else:
            self.data_split_folder = data_split_folder
            
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.subsample = subsample
        self.mode = mode
        self.augment_data = augment_data
        self.strong_augmentation = strong_augmentation
        self.train_predefined_index = train_predefined_index
        self.val_predefined_index = val_predefined_index
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
        self.data_split_folder_training = os.path.join(self.data_split_folder, 'train')
        self.data_split_folder_validation = os.path.join(self.data_split_folder, 'test')
        if self.mode == 'training':
            self.dataset = RECONVIDDataset(
                data_folder=self.data_dir,
                data_split_folder=self.data_split_folder_training,
                min_dist_cat=0,
                max_dist_cat=100000,
                dataset_name='recon',
                image_size=(320,320), # 224x224 for input
                len_traj_pred=self.video_length,
                normalize=True,
                context_size=1,
                transform=transform,
                predefined_index=self.train_predefined_index,
                video_length=self.video_length,
                traj_stride=1,
                traj_names='traj_names.txt',
                mode='training',
            )
            print(f"train_dataset: {len(self.dataset)}")
        if self.mode == 'validation':
            if self.subsample:
                # When subsample=True, use a large traj_stride to ensure each trajectory only generates one sample
                # This limits the total number of validation samples
                self.dataset = RECONVIDDataset(
                    data_folder=self.data_dir,
                    data_split_folder=self.data_split_folder_validation,
                    dataset_name='recon',
                    image_size=(320,320), # 224x224 for input
                    min_dist_cat=0,
                    max_dist_cat=100000,
                    len_traj_pred=self.video_length,
                    traj_stride=100000,  # Large stride to ensure each trajectory only generates one sample
                    context_size=1,
                    transform=transform,
                    traj_names='traj_names_subsample.txt',
                    normalize=True,
                    predefined_index=self.val_predefined_index,
                    video_length=self.video_length,
                    mode='validation',
                )
            else:
                # this branch is currently deprecated
                assert self.exp is not None, "exp is not set, choices [rollout, single_step]"
                if self.exp == "rollout":
                    predefined_index = os.path.join(_VID2WORLD_RUNTIME_ROOT, "nvm_utils/data_splits/recon/test/rollout.pkl")
                elif self.exp == "single_step":
                    predefined_index = os.path.join(_VID2WORLD_RUNTIME_ROOT, "nvm_utils/data_splits/recon/test/time.pkl")
                else:
                    raise ValueError(f"Invalid exp: {self.exp}")
                self.dataset = RECONEvalDataset(
                    data_folder=self.data_dir,
                    data_split_folder=self.data_split_folder_validation,
                    dataset_name='recon',
                    image_size=(320,320), # 224x224 for input
                    min_dist_cat=-64,
                    max_dist_cat=64,
                    traj_stride=1,
                    len_traj_pred=self.video_length -4, # self.video_length should be 20
                    context_size = 4,
                    normalize=True,
                    transform=transform,
                    predefined_index=predefined_index,
                    traj_names='traj_names.txt',
                    mode='validation',
                )
            print(f"val_dataset: {len(self.dataset)}")
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
    
    def __getitem__(self, index):
        """Getitem.

        Args:
            index: The index.
        """
        
        # Load data
        if self.mode == 'training':
            frames = self.dataset[index][0]  # [T, H, W, C] # [115, 256, 320, 3]
            actions = self.dataset[index][1]  # [T, M]
        elif self.mode == 'validation':
            frames = self.dataset[index][0]  # [T, H, W, C] # [115, 256, 320, 3]
            actions = self.dataset[index][1]  # [T, M]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        frames_tensor = torch.tensor(frames).clone().detach().permute(1,0,2,3).float()  # [C, T, H, W], range:[-1,1]
        frames_tensor = (frames_tensor+1.0)/2.0 * 255.0
        actions_tensor = torch.tensor(actions).clone().detach().float()  # [T, M]
            
        
        if self.spatial_transform is not None:
            frames_tensor = self.spatial_transform(frames_tensor)

        if self.augment_data:
            frames_tensor = self.augment_video_clip(frames_tensor)
        
        if self.resolution is not None:
            assert (frames_tensor.shape[2], frames_tensor.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames={frames_tensor.shape}, self.resolution={self.resolution}'
        
        # Normalize frames to [-1, 1]
        frames_tensor = (frames_tensor / 255.0 - 0.5) * 2

        data = {
            'video': frames_tensor, 
            'caption': "",
            'action': actions_tensor,
            'path': self.dataset[index][2],
            'fps': 3, # for rt dataset, fps is fixed to 3
            'frame_stride': 1, # for rt dataset, frame_stride is fixed to 1
            'curr_time': self.dataset[index][3],
        }
        return data
    
    def __len__(self):
        """Len."""
        return len(self.dataset)
    
    def data_augmentation(self, images):
        """Data augmentation.

        Args:
            images: The images.
        """
        # Calculate random crop parameters
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
