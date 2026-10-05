# Adapted from https://github.com/facebookresearch/nwm/tree/main

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import pickle
import tqdm
from torch.utils.data import Dataset
from worldfoundry.synthesis.visual_generation.vid2world.config_paths import (
    load_runtime_yaml,
)
from worldfoundry.synthesis.visual_generation.vid2world.nvm_utils.misc import (
    angle_difference,
    get_data_path,
    get_delta_np,
    normalize_data,
    to_local_coords,
    transform,
)
import random


# Help NumPy 1.x unpickle NumPy 2.x pickles, please use this sparingly and only temporarily
if np.__version__[:2] == "1.":
    import sys
    # More comprehensive compatibility mapping for NumPy 2.x pickles
    try:
        sys.modules["numpy._core.numeric"] = np.core.numeric
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        sys.modules["numpy._core.umath"] = np.core.umath
        sys.modules["numpy._core.arrayprint"] = np.core.arrayprint
        sys.modules["numpy._core.fromnumeric"] = np.core.fromnumeric
        sys.modules["numpy._core.defchararray"] = np.core.defchararray
        sys.modules["numpy._core.records"] = np.core.records
        sys.modules["numpy._core.function_base"] = np.core.function_base
        sys.modules["numpy._core.machar"] = np.core.machar
        sys.modules["numpy._core.getlimits"] = np.core.getlimits
        sys.modules["numpy._core.shape_base"] = np.core.shape_base
        sys.modules["numpy._core.stride_tricks"] = np.core.stride_tricks
        sys.modules["numpy._core.einsumfunc"] = np.core.einsumfunc
        sys.modules["numpy._core._asarray"] = np.core._asarray
        sys.modules["numpy._core._dtype_ctypes"] = np.core._dtype_ctypes
        sys.modules["numpy._core._internal"] = np.core._internal
        sys.modules["numpy._core._dtype"] = np.core._dtype
        sys.modules["numpy._core._exceptions"] = np.core._exceptions
        sys.modules["numpy._core._methods"] = np.core._methods
        sys.modules["numpy._core._type_aliases"] = np.core._type_aliases
        sys.modules["numpy._core._ufunc_config"] = np.core._ufunc_config
        sys.modules["numpy._core._add_newdocs"] = np.core._add_newdocs
        sys.modules["numpy._core._add_newdocs_scalars"] = np.core._add_newdocs_scalars
        sys.modules["numpy._core._multiarray_tests"] = np.core._multiarray_tests
        sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath
        sys.modules["numpy._core._operand_flag_tests"] = np.core._operand_flag_tests
        sys.modules["numpy._core._struct_ufunc_tests"] = np.core._struct_ufunc_tests
        sys.modules["numpy._core._umath_tests"] = np.core._umath_tests
    except AttributeError:
        # Some modules might not exist in older NumPy versions
        pass

class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = data_folder

        # Convert relative path to absolute path based on project root
        if data_split_folder and not os.path.isabs(data_split_folder):
            project_root = os.path.dirname(os.path.dirname(__file__))
            self.data_split_folder = os.path.join(project_root, data_split_folder)
        else:
            self.data_split_folder = data_split_folder

        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs


        traj_names_file = os.path.join(self.data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        all_data_config = load_runtime_yaml("data_config.yaml")

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        if predefined_index:
            print(f"****** Using a predefined evaluation index... {predefined_index}******")
            with open(predefined_index, "rb") as f:
                self.index_to_data = pickle.load(f)
                return
        else:
            print("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )

            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index

    def _get_trajectory(self, trajectory_name):
        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k,v in traj_data.items():
            traj_data[k] = v.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time, only_goal=False):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            # for the case that the trajectory is shorter than the video length, which can happen in our usage of this in training set
            if not only_goal:
                raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]

        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)

        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            # Ensure goal_pos is 2D for proper indexing
            if goal_pos.ndim == 1:
                goal_pos = goal_pos.reshape(1, -1)
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]

        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        # If goal_pos was originally 1D, squeeze it back to 1D
        if goal_pos.shape[0] == 1:
            goal_pos = goal_pos.squeeze(0)
        return actions, goal_pos

class RECONVIDDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int, # dummy
        transform: object,
        traj_names: str = 'traj_names.txt',
        normalize: bool = True,
        predefined_index: list = None,
        mode: str = 'training',
        video_length: int = 16,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat, len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs=video_length)
        self.mode = mode
        self.video_length = video_length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            # Our logic of sampling a continuous video & corresponding sequence is as follows:
            # 1. context_times is the first frame, goal_time is all the following frames
            # 2. actions are relative to the current frame: i.e. a_{t} = x_{t+1} - x_{t}
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            if max_goal_dist < self.video_length:
                raise ValueError(f"max_goal_dist < video_length, skipping {i}")
                # self.__getitem__(i+1)
            goal_offset = np.arange(self.video_length-1) + 1 # [1, 2, ..., video_length-1]
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor, currently a fixed const
            context_times = list(range(curr_time, curr_time + 1)) # [curr_time]
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)
            actions_seq = np.zeros((self.video_length, 3))
            # Compute actions
            for j,t in enumerate(goal_time):
                if j==0:
                    assert t-1 == curr_time # just making sure it is continuous
                _, t_goal_pos = self._compute_actions(curr_traj_data, t-1, t, only_goal=True)
                actions_seq[j] = t_goal_pos
            actions_seq[:, :2] = normalize_data(actions_seq[:, :2], self.ACTION_STATS)
            # one may realize that such instantiation method makes the last action in actions_seq all zeros before normalization
            # after normalization, it is [-0.33333333,  0., 0.]
            # this is very much correct since we never use this action anyway.


            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(actions_seq, dtype=torch.float32),
                f_curr,
                curr_time,
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)

class RECONEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int, # should be 16
        traj_stride: int,
        context_size: int, # should be 4
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        mode: str = 'training',
        video_length: int = 20,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat, len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs=video_length)
        self.mode = mode
        self.video_length = video_length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            # Our logic of sampling a continuous video & corresponding sequence is as follows:
            # 1. context_times is the first frame, goal_time is all the following frames
            # 2. actions are relative to the current frame: i.e. a_{t} = x_{t+1} - x_{t}
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            if max_goal_dist < self.video_length:
                self.__getitem__(i+1)
                print(f"max_goal_dist < video_length, skipping {i}")

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
            assert self.context_size + self.len_traj_pred == self.video_length
            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]
            all_frames = context + pred
            all_times = context_times + pred_times
            all_frames_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in all_frames])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)
            actions_seq = np.zeros((self.video_length, 3))
            # Compute actions
            for j,t in enumerate(all_times):
                if j==0:
                    continue
                if j==1:
                    assert t-1 == curr_time - self.context_size + 1 # just making sure it is continuous
                _, t_goal_pos = self._compute_actions(curr_traj_data, t-1, t, only_goal=True)
                actions_seq[j-1] = t_goal_pos
            actions_seq[:, :2] = normalize_data(actions_seq[:, :2], self.ACTION_STATS)
            # one may realize that such instantiation method makes the last action in actions_seq all zeros before normalization
            # after normalization, it is [-0.33333333,  0., 0.]
            # this is very much correct since we never use this action anyway.


            return (
                torch.as_tensor(all_frames_image, dtype=torch.float32),
                torch.as_tensor(actions_seq, dtype=torch.float32),
                f_curr,
                curr_time,
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
