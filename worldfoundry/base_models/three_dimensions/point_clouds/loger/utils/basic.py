"""Module for base_models -> three_dimensions -> point_clouds -> loger -> utils -> basic.py functionality."""

import os
import os.path as osp
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
from plyfile import PlyData, PlyElement
import numpy as np


def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000, Target_W=None, Target_H=None):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith('.mp4'):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    if Target_W is None and Target_H is None:
        first_img = sources[0]
        W_orig, H_orig = first_img.size
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = Target_W, Target_H
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)
