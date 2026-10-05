"""
HunyuanMirror Operator

This operator handles user interactions for the HunyuanMirror pipeline.
It processes user commands like camera control, object manipulation, etc.
"""

import numpy as np
import torch
from .base_operator import BaseOperator
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.camera_utils import vector_to_camera_matrices

class HunyuanMirrorOperator(BaseOperator):
    """Operator class for the HunyuanMirror model integration."""
    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["action_instruction", "textual_instruction"]
        if interaction_template is None:
            interaction_template = [
                "camera_look_up",
                "camera_look_down",
                "camera_look_left",
                "camera_look_right",
                "camera_zoom_in",
                "camera_zoom_out",
                "object_move_forward",
                "object_move_backward",
                "object_move_left",
                "object_move_right",
                "reset_camera",
            ]
        super(HunyuanMirrorOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()

        self.current_interaction = []  # 保存用户的交互指令序列

    def check_interaction(self, interaction):
        """
        检查用户交互是否在模板范围内
        """
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in interaction template: {self.interaction_template}")
        return True
    
    def get_interaction(self, interaction):
        """
        获得用户的交互指令
        """
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, num_frames, image_hw=(518, 518)):
        """
        处理用户交互指令，将其转换为representation和synthesis可以处理的形式
        
        Args:
            num_frames: 生成的帧数
            image_hw: 图像的高度和宽度
            
        Returns:
            dict: 包含处理后的相机参数或其他控制信息
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        
        # 处理最近的交互指令
        latest_interaction = self.current_interaction[-1]
        
        # 将交互记录到历史
        self.interaction_history.append(latest_interaction)
        
        # 根据不同的交互指令生成相机参数
        h, w = image_hw
        
        # 默认相机参数
        default_cam_vec = torch.tensor([
            0.0, 0.0, 0.0,  # 位置 (x, y, z)
            0.0, 0.0, 0.0, 1.0,  # 旋转 (quaternion: x, y, z, w; scalar-last)
            torch.pi / 3,  # 垂直FOV
            torch.pi / 3   # 水平FOV
        ])

        def with_axis_angle(vec, axis, angle):
            """Return ``vec`` with a normalized scalar-last axis-angle quaternion."""
            axis = torch.as_tensor(axis, dtype=vec.dtype, device=vec.device)
            axis = axis / axis.norm().clamp_min(torch.finfo(vec.dtype).eps)
            half_angle = torch.as_tensor(angle * 0.5, dtype=vec.dtype, device=vec.device)
            quaternion = torch.cat((axis * torch.sin(half_angle), torch.cos(half_angle)[None]))
            adjusted = vec.clone()
            adjusted[3:7] = quaternion
            return adjusted
        
        # 相机参数调整量
        cam_adjustments = {
            "camera_look_up": lambda vec: with_axis_angle(vec, (1.0, 0.0, 0.0), 0.2),
            "camera_look_down": lambda vec: with_axis_angle(vec, (1.0, 0.0, 0.0), -0.2),
            "camera_look_left": lambda vec: with_axis_angle(vec, (0.0, 0.0, 1.0), 0.2),
            "camera_look_right": lambda vec: with_axis_angle(vec, (0.0, 0.0, 1.0), -0.2),
            "camera_zoom_in": lambda vec: torch.cat([vec[:2], vec[2:3] - 0.1, vec[3:7], vec[7:]]),  # 靠近物体
            "camera_zoom_out": lambda vec: torch.cat([vec[:2], vec[2:3] + 0.1, vec[3:7], vec[7:]]),  # 远离物体
            "reset_camera": lambda vec: default_cam_vec.clone(),  # 重置相机
            # 物体移动的处理可以在这里扩展
            "object_move_forward": lambda vec: vec,  # 示例：暂时不处理物体移动
            "object_move_backward": lambda vec: vec,
            "object_move_left": lambda vec: vec,
            "object_move_right": lambda vec: vec,
        }
        
        # 应用相机调整
        if latest_interaction in cam_adjustments:
            adjusted_cam_vec = cam_adjustments[latest_interaction](default_cam_vec)
        else:
            adjusted_cam_vec = default_cam_vec.clone()
        
        # 生成相机矩阵
        ext, intr = vector_to_camera_matrices(adjusted_cam_vec, image_hw=image_hw)
        
        # 重复参数以匹配帧数
        ext = ext.unsqueeze(0).repeat(num_frames, 1, 1)
        intr = intr.unsqueeze(0).repeat(num_frames, 1, 1)
        
        # 返回处理后的相机参数
        return {
            "interaction": latest_interaction,
            "num_frames": num_frames,
            "camera_poses": ext,  # 相机姿态矩阵 (num_frames, 3, 4)
            "camera_intrinsics": intr,  # 相机内参 (num_frames, 3, 3)
            "image_hw": image_hw,
        }
