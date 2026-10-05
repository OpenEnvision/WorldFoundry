import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import json
import os
import pdb
import glob
from torchvision.io import read_video
import torch.nn.functional as F

def change_fps(video_tensor, fps, target_fps):
    if target_fps == -1:
        return video_tensor
    
    # 获取输入视频的帧率和帧大小
    num_frames, height, width, _ = video_tensor.shape

    # 计算采样间隔或插帧间隔
    if fps > target_fps:
        # 采样间隔
        interval = int(fps / target_fps)
        # 对高帧率视频进行采样
        output_tensor = video_tensor[::interval]
    else:
        # 插帧间隔
        interval = int(target_fps / fps)
        # 对低帧率视频进行插帧
        output_frames = []
        for frame in video_tensor:
            output_frames.append(frame)
            for _ in range(interval - 1):
                output_frames.append(frame)
        output_tensor = torch.stack(output_frames)
    return output_tensor


def resize_long_side(image_tensor, target_long_side):
    """
    Resizes a tensor image to have the specified size for its longest side while maintaining the aspect ratio.

    Args:
        image_tensor (torch.Tens
        or): Input image tensor of dtype torch.uint8 with shape (C, H, W).
        target_long_side (int): Desired size of the longest side of the resized image.

    Returns:
        torch.Tensor: The resized image tensor.
    """
    # Ensure that the input is a 3D tensor and the data type is uint8
    
    # Get the original dimensions of the image
    height, width = image_tensor.shape[-2:]

    # Determine whether the width or height is longer, and compute the new dimensions accordingly
    if width > height:
        scale = target_long_side / width
        new_width = target_long_side
        new_height = int(height * scale)
    else:
        scale = target_long_side / height
        new_height = target_long_side
        new_width = int(width * scale)

    # Reshape the image using bilinear interpolation (align_corners=False is recommended)
    resized_image = F.interpolate(
        image_tensor.float(),  
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    )  # Remove the batch dimension and convert back to uint8

    return resized_image

def find_video_files(root_dir):
    # 存储视频文件的完整路径
    video_files = []

    # 遍历指定的根目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前目录下的每个文件
        for filename in filenames:
            # 检查文件扩展名是否为 mp4 或 avi
            if filename.endswith('.mp4') or filename.endswith('.avi'):
                # 构建文件的完整路径并添加到列表中
                full_path = os.path.join(dirpath, filename)
                video_files.append(full_path)
    
    # 返回视频文件路径列表
    return video_files

class StandardVidoDataset(Dataset):
    def __init__(self, video_dir, transform=None, transform_no_resize=None, return_video=True, target_fps=8.0):
        """
        Args:
            annotations_file (string): 描述和时间戳的 JSON 文件路径。
            video_dir (string): 包含视频文件的目录路径。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.video_dir = video_dir
        self.transform = transform
        self.transform_no_resize = transform_no_resize
        video_paths = find_video_files(video_dir)
        self.video_paths = [v for v in video_paths if not os.path.basename(v).startswith('._')]
        self.return_video = return_video
        self.target_fps = target_fps

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        try:
            video, _, video_metas = read_video(video_path, pts_unit='sec')
            video = change_fps(video, video_metas['video_fps'],self.target_fps)
            # interval = round(video_metas['video_fps'] / self.target_fps)
            # if interval > 1:
            #     video = video[::interval]
        except BaseException as e:
            print(video_path)
            print(f'{e}')
            return None
        org_video = video.clone()

        sample = {
            'video_name':os.path.basename(video_path),
            'length':video.shape[0],
            'video_path': video_path,
            }

        if self.return_video:
            video_no_resize = resize_long_side(video.permute(0,3,1,2).float() / 255, 512)
            sample.update({'video_no_resize': video_no_resize})
            video = self.transform(video.permute(0,3,1,2).float() / 255)
            sample.update({'video': video})
            sample.update({'org_video': org_video})

        return sample

def standard_collate_fn(batch):
    video_names = []
    videos = []
    videos_no_resize = []
    video_lengths = []
    org_videos = []
    video_paths = []
    for item in batch:
        if item is not None:  # 确保读取视频没有出错
            video_names.append(item['video_name'])
            video_lengths.append(item['length'])
            video_paths.append(item['video_path'])

            if 'video' in item:
                videos.append(item['video'])
            if 'video_no_resize' in item:
                videos_no_resize.append(item['video_no_resize'])
            if 'org_video' in item:
                org_videos.append(item['org_video'])

    # 使用torch.cat在时间维度上连接视频，假设每个视频是shape [C, T, H, W]
    # 如果你的视频shape是[T, H, W, C]，可能需要先转置
    if 'video' in item:
        videos = torch.cat(videos, dim=0)  # 这里假设dim=1是时间维度，根据你的具体情况调整
    
    # 返回字典，包含视频名称、合并后的视频和描述
    return {
        'video_names': video_names,
        'videos': videos,
        'video_lengths': video_lengths,
        'org_videos': org_videos,
        'videos_no_resize': videos_no_resize,
        'video_paths': video_paths,
    }



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Inspect videos with StandardVidoDataset")
    parser.add_argument("video_folder", help="Directory containing input videos")
    args = parser.parse_args()
    video_folder = args.video_folder
    transform = Compose([
        Resize((224, 224)),  # 根据需要调整大小
    ])

    dataset = StandardVidoDataset(video_folder, transform=transform)
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=standard_collate_fn)

    # 测试数据加载
    for i, data in enumerate(dataloader):
        videos = data['videos']
        org_videos = data['org_videos']
        pdb.set_trace()
        print(f"Batch {i+1}:")
        print(f"Videos shape: {videos.shape}")  # 打印视频张量的形状
        print(f'len(org_videos): {len(org_videos)}')
        if i == 1:  # 为了测试，我们只迭代两个批次
            break
