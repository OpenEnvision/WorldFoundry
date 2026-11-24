"""Module for the Camera operator implementation."""

import logging

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms import functional as TF

from .base_operator import BaseOperator

logger = logging.getLogger(__name__)


class Camera:
    """Operator class for the Camera model integration."""

    def __init__(self, c2w):
        """Initialize the operator with specific configurations."""
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class ReCamMasterOperator(BaseOperator):
    """Operator class for the ReCamMaster model integration."""

    def __init__(self, operation_types=None, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = []
        super().__init__(operation_types=operation_types)
        self.camera_init = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height, self.width = height, width

    def rotation_matrix_z(self, theta: float) -> np.ndarray:
        """Return the 4x4 homogeneous rotation by ``theta`` radians about the z axis."""
        return np.array(
            [[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

    def rotation_matrix_x(self, theta: float) -> np.ndarray:
        """Return the 4x4 homogeneous rotation by ``theta`` radians about the x axis."""
        return np.array(
            [[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]]
        )

    def translation_matrix(self, dx: float, dy: float, dz: float) -> np.ndarray:
        """Return a 4x4 matrix whose last row is ``(dx + 3390, dy + 1380, dz + 240, 0)``, added to a pose before it is transposed."""
        return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [dx + 3390, dy + 1380, dz + 240, 0]])

    def crop_and_resize(self, image):
        """Resize ``image`` bilinearly, keeping its aspect ratio, so that it covers ``(self.height, self.width)``."""
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = TF.resize(
            image,
            [round(height * scale), round(width * scale)],
            interpolation=InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(
        self,
        file_path,
        start_frame_id=0,
    ):
        """Read ``num_frames`` frames every ``frame_interval`` frames from ``start_frame_id`` as a ``[1, C, T, H, W]`` tensor.

        Returns None and logs a warning when the video is too short.
        """
        reader = imageio.get_reader(file_path)
        # The FFmpeg reader that opens these videos defines count_frames; ImageIO's generic Reader stub does not.
        frame_count = reader.count_frames()  # ty: ignore[unresolved-attribute]
        if (
            frame_count < self.max_num_frames
            or frame_count - 1 < start_frame_id + (self.num_frames - 1) * self.frame_interval
        ):
            reader.close()
            logger.warning(
                "Your video clip is too short. The length of video clip need longer than %s", self.max_num_frames
            )
            return None

        frames = []
        first_frame = None
        for frame_id in range(self.num_frames):
            frame = reader.get_data(start_frame_id + frame_id * self.frame_interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = self.frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        # if is_i2v:
        #     return frames, first_frame
        return frames.unsqueeze(0)

    def get_relative_pose(self, cam_params):
        """Return the camera-to-world poses of ``cam_params`` relative to the first camera as a float32 ``[N, 4, 4]`` array."""
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

        cam_to_origin = 0
        target_cam_c2w = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [
            target_cam_c2w,
        ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def check_interaction(self, interaction):
        """
        the input interaction should be a dict:
        {"camera_viewpoint": [dx, dy, dz, theta_x, theta_z], "textual_instruction": str,}
        """
        if "camera_viewpoint" not in interaction:
            raise ValueError("interaction need include camera_viewpoint.")
        if "textual_instruction" not in interaction:
            raise ValueError("interaction need include textual instruction.")
        if len(interaction["camera_viewpoint"]) != 5:
            raise ValueError("the length of interaction need include dx, dy, dz, theta_x, theta_z.")
        if type(interaction["textual_instruction"]) is not str:
            raise ValueError("the type of textual_instruction should be string.")
        return True

    def get_interaction(
        self,
        camera_trajectory,
        textual_instruction,
    ):
        """Process and append the interaction to the current sequence."""
        interaction = {"camera_viewpoint": camera_trajectory, "textual_instruction": textual_instruction}
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_perception(self, video_path, start_frame_id=0, **kwargs):
        """Process perception inputs like images, videos, and reference frames."""
        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(self.height, self.width)),
                v2.Resize(size=(self.height, self.width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return self.load_frames_using_imageio(file_path=video_path, start_frame_id=start_frame_id)

    def process_interaction(
        self,
    ):
        """
        Turn the latest interaction into a smooth camera trajectory.

        Input: ``cur_interaction["camera_viewpoint"] = [dx, dy, dz, theta_x, theta_z]`` with angles in degrees.
        Output: camera pose embedding of shape (1, len(range(0, num_frames, 4)), 12), for example
        (1, 21, 12) for 81 frames.
        """
        cur_interaction = self.current_interaction[-1]
        camera_change = cur_interaction["camera_viewpoint"]  # [dx, dy, dz, theta_x, theta_z]

        target_dx, target_dy, target_dz, target_theta_x, target_theta_z = camera_change
        num_frames = self.num_frames  # default = 81

        # construct num_frames linspace
        t_vals = np.linspace(0, 1, num_frames)[::4]

        dx_steps = t_vals * target_dx
        dy_steps = t_vals * target_dy
        dz_steps = t_vals * target_dz
        theta_x_steps = t_vals * target_theta_x
        theta_z_steps = t_vals * target_theta_z

        # camera view init
        init_c2w = np.array(self.camera_init, dtype=np.float32).reshape(4, 4)

        all_c2ws = []
        for i in range(len(t_vals)):
            # translation matrix
            trans_mat = self.translation_matrix(dx_steps[i], dy_steps[i], dz_steps[i])
            # rotation matrix
            rot_z_mat = self.rotation_matrix_z(np.radians(theta_z_steps[i]))
            rot_x_mat = self.rotation_matrix_x(np.radians(theta_x_steps[i]))

            # composition matrix
            incremental_transform = rot_z_mat @ rot_x_mat

            # application to the camera view
            current_c2w = init_c2w @ incremental_transform + trans_mat
            all_c2ws.append(current_c2w)

        # reference to the recammaster code
        processed_poses = []
        for c2w in all_c2ws:
            c2w = c2w.T  # need check
            c2w_rearranged = c2w[:, [1, 2, 0, 3]].copy()
            c2w_rearranged[:3, 1] *= -1.0
            c2w_rearranged[:3, 3] /= 100.0
            processed_poses.append(c2w_rearranged)

        # construct camera list
        cam_params = [Camera(pose) for pose in processed_poses]
        relative_poses = []
        for i in range(len(cam_params)):
            relative_pose = self.get_relative_pose([cam_params[0], cam_params[i]])
            relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])

        pose_embedding = torch.stack(relative_poses, dim=0)  # shape: (21, 3, 4)
        pose_embedding = rearrange(pose_embedding, "b c d -> b (c d)")  # shape: (21, 12)

        return pose_embedding.unsqueeze(0)
