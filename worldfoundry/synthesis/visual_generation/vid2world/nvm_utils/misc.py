# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from worldfoundry.synthesis.visual_generation.vid2world.config_paths import (
    load_runtime_yaml,
)

# Get the absolute path to the DynamiCrafter directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_ASPECT_RATIO = (4 / 3)  # all images are centered cropped to a 4:3 aspect ratio in training

data_config = load_runtime_yaml("data_config.yaml")


def get_action_torch(diffusion_output, action_stats):
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = torch.cumsum(ndeltas, dim=1)
    return actions.to(ndeltas)

def log_viz_single(dataset_name, obs_image, goal_image, preds, deltas, loss, min_idx, actions, action_stats, plan_iter=0, output_dir='plot.png'):
    '''
    Visualize a single instance
    actions is gt actions
    '''
    viz_obs_image = unnormalize(obs_image.detach().cpu())[-1] # take last img 
    viz_goal_image = unnormalize(goal_image.detach().cpu())
    deltas = deltas.detach().cpu()
    loss = loss.detach().cpu()
    actions = actions.detach().cpu()
    pred_actions = get_action_torch(deltas[:, :, :2], action_stats)
    plot_array = plot_images_and_actions(dataset_name, viz_obs_image, viz_goal_image, pred_actions, actions, min_idx, loss=loss)

    plt.imshow(plot_array)
    plt.axis('off')  # Hide axes for a cleaner image

    # Save the plot array as a PNG file locally
    plt.savefig(output_dir, format='png', dpi=300, bbox_inches='tight')

def plot_images_and_actions(dataset_name, curr_viz_obs_image, curr_viz_goal_image, curr_viz_pred_actions, curr_viz_actions, min_idx, loss):
    curr_viz_obs_image = curr_viz_obs_image.permute(1, 2, 0).cpu().numpy()
    curr_viz_goal_image = curr_viz_goal_image.permute(1, 2, 0).cpu().numpy()

    # scale back to metric space for plotting
    curr_viz_pred_actions = curr_viz_pred_actions * data_config[dataset_name]['metric_waypoint_spacing']
    curr_viz_actions = curr_viz_actions * data_config[dataset_name]['metric_waypoint_spacing']
    
    # Create the figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    # Plot condition image
    axs[0].imshow(curr_viz_obs_image)
    axs[0].set_title("Condition Image", fontsize=13)
    axs[0].axis("off")

    # Plot goal image
    axs[1].imshow(curr_viz_goal_image)
    axs[1].set_title("Goal Image", fontsize=13)
    axs[1].axis("off")

    colors = ['red', 'orange', 'cyan']
    for i in range(1, curr_viz_pred_actions.shape[0]):
        color = colors[(i - 1) % len(colors)]
        label = f"Sample {i} Min Loss" if i == min_idx.item() else f"{i}"

        if i != min_idx.item():
            axs[2].plot(-curr_viz_pred_actions[i, :, 1], curr_viz_pred_actions[i, :, 0], 
                        color=color, marker="o", markersize=5, label=label)
            axs[2].text(-curr_viz_pred_actions[i, -1, 1], 
                curr_viz_pred_actions[i, -1, 0], 
                round(loss[i].item(), 3), 
                color='black', 
                fontsize=10, 
                ha='left', va='bottom')  # Adjust position to avoid overlap

    # Highlight the minimum loss sample
    axs[2].plot(-curr_viz_pred_actions[min_idx.item(), :, 1], curr_viz_pred_actions[min_idx.item(), :, 0], 
                color='green', marker="o", markersize=5, label=f"{min_idx.item()}")
    axs[2].text(-curr_viz_pred_actions[min_idx.item(), -1, 1], 
        curr_viz_pred_actions[min_idx.item(), -1, 0], 
        round(loss[min_idx.item()].item(), 3), 
        color='black', 
        fontsize=10, 
        ha='left', va='bottom')  # Adjust position to avoid overlap

    # Plot ground truth actions
    axs[2].plot(-curr_viz_actions[:, 1], curr_viz_actions[:, 0], color='blue', marker="o", label="GT")

    # Set titles and labels with larger font size
    axs[2].set_title("   ", fontsize=13)
    axs[2].set_xlabel("X (m)", fontsize=11)
    axs[2].set_ylabel("Y (m)", fontsize=11)

    # Set equal aspect ratio and adjust axis limits
    axs[2].set_aspect('equal', adjustable='box')
    x_min, x_max = axs[2].get_xlim()
    y_min, y_max = axs[2].get_ylim()
    axis_range = max(x_max - x_min, y_max - y_min) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    axs[2].set_xlim(x_mid - axis_range, x_mid + axis_range)
    axs[2].set_ylim(y_mid - axis_range, y_mid + axis_range)

    axs[2].legend(loc='lower left', fontsize=10, frameon=True, bbox_to_anchor=(0, 0))
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_array = plot_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return plot_array


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'].to(ndata) - stats['min'].to(ndata)) + stats['min'].to(ndata)
    return data

def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1    
    delta_theta = delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))    
    return delta_theta

def get_delta_np(actions):
    # append zeros to first action (unbatched)
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1])), actions), axis=0)
    delta = ex_actions[1:] - ex_actions[:-1]
    
    return delta

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def calculate_delta_yaw(unnorm_actions):
    x = unnorm_actions[..., 0]
    y = unnorm_actions[..., 1]
    
    yaw = torch.atan2(y, x).unsqueeze(-1)
    delta_yaw = torch.cat((torch.zeros(yaw.shape[0], 1, yaw.shape[2]).to(yaw.device), yaw), dim=1)
    delta_yaw = delta_yaw[:, 1:, :] - delta_yaw[:, :-1, :]
    
    return delta_yaw

def save_planning_pred(dataset_save_output_dir, B, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions, plan_iter=0):
    for batch_idx, idx in enumerate(idxs.flatten()):
        sample_idx = int(idx)
        sample_folder = os.path.join(dataset_save_output_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        
        preds_save = {
            'obs_image': obs_image[batch_idx],
            'goal_image': goal_image[batch_idx],
            'nwm_preds': preds[batch_idx],
            'deltas': deltas[batch_idx],
            'loss': loss[batch_idx],
            'gt_actions': gt_actions[batch_idx],
        }
        preds_file = os.path.join(sample_folder, f"preds_{plan_iter}.pth")
        torch.save(preds_save, preds_file)
        
class CenterCropAR:
    def __init__(self, ar: float = IMAGE_ASPECT_RATIO):
        self.ar = ar

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w > h:
            img = TF.center_crop(img, (h, int(h * self.ar)))
        else:
            img = TF.center_crop(img, (int(w / self.ar), w))
        return img

transform = transforms.Compose([
    CenterCropAR(),
    transforms.Resize((320, 320)), # for dynamicrafter, the image size is 320x512, so we resize it to 320x320 first
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])

unnormalize = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)
