# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import OpenEXR
import torch
import torch.utils.data
from decord import VideoReader
from lru import LRU

from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.eval_inputs.base import BaseDataset
from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.eval_inputs.datafield import DataField


def _safe_numpy_latent_globals() -> list[Any]:
    """Return inert NumPy/codecs globals needed to load legacy array latents with weights_only=True."""
    safe_globals: list[Any] = [
        np.ndarray,
        np.dtype,
        codecs.encode,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float16)),
        type(np.dtype(np.float64)),
    ]
    for module_name in ("core", "_core"):
        submod = getattr(np, module_name, None)
        multiarray = getattr(submod, "multiarray", None) if submod is not None else None
        reconstruct = getattr(multiarray, "_reconstruct", None) if multiarray is not None else None
        if reconstruct is not None:
            safe_globals.append(reconstruct)
    return safe_globals


def _load_rgb_latents(rgb_latents_path: Path) -> torch.Tensor:
    with torch.serialization.safe_globals(_safe_numpy_latent_globals()):
        rgb_latents_data = torch.load(
            rgb_latents_path,
            map_location="cpu",
            weights_only=True,
        )
    if isinstance(rgb_latents_data, np.ndarray):
        rgb_latents_data = torch.from_numpy(rgb_latents_data)
    return rgb_latents_data


class Radym(BaseDataset):
    MAX_ZIP_DESCRIPTORS = 10
    MAX_MP4_READERS = 2

    def __init__(
        self,
        root_path,
        filter_list_path: str | None = None,
        num_views: int = -1,
        depth_folder: str = "depth",
        custom_folders: list[str] | None = None,
        custom_fields: list[str] | None = None,
        start_view_idx: int = 0,
        depth_path: list[str] | None = None,
        camera_path: list[str] | None = None,
    ):
        # For multi-view datasets, root_path is the path to camera idx 0.
        self.root_path = root_path

        # filter_list_path is a text file containing the list of mp4 files to load.
        # Each line in the file should contain the name of the mp4 file with or without the extension.
        if filter_list_path is None:
            self.filter_set = None
        else:
            self.filter_list_path = (
                filter_list_path if os.path.isabs(filter_list_path) else os.path.join(root_path, filter_list_path)
            )
            with open(self.filter_list_path) as f:
                self.filter_set = [line.strip() for line in f.readlines()]
            self.filter_set = {x.split(".")[0] for x in self.filter_set}
        self.n_views = num_views

        # Recursively grab all mp4 files in subfolders with name 'rgb'.
        self.mp4_file_paths = []
        for rgb_root in Path(root_path).rglob("rgb"):
            if not rgb_root.is_dir():
                continue
            print(rgb_root)
            for mp4_file in rgb_root.glob("*.mp4"):
                if self.filter_set is None or mp4_file.stem in self.filter_set:
                    self.mp4_file_paths.append(mp4_file)
        self.mp4_file_paths = sorted(self.mp4_file_paths)
        # Process-dependent LRU cache for file handles of the tar files.
        self.worker_id = None
        self.zip_descriptors = LRU(self.MAX_ZIP_DESCRIPTORS, callback=self._evict_zip_handle)
        # self.mp4_readers = LRU(self.MAX_MP4_READERS, callback=self._evict_mp4_reader)
        self.depth_folder = depth_folder
        self.custom_folders = custom_folders
        self.custom_fields = custom_fields

        # If view index doesn't start at 0
        self.start_view_idx = start_view_idx

        # Store some annotations in different folders
        self.depth_path = Path(depth_path) if depth_path is not None else depth_path
        self.camera_path = Path(camera_path) if camera_path is not None else camera_path

    @staticmethod
    def _evict_zip_handle(_, zip_handle):
        zip_handle.close()

    @staticmethod
    def _evict_mp4_reader(_, mp4_reader: VideoReader):
        # This is no-op, just a placeholder.
        del mp4_reader

    def _check_worker_id(self):
        # Protect handle boundary:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if self.worker_id is not None:
                assert self.worker_id == worker_info.id, "Worker id mismatch"
            else:
                self.worker_id = worker_info.id

    def _get_zip_handle(self, idx, attr, view_idx, attr_path=None):
        self._check_worker_id()

        dict_key = f"{idx}_{view_idx}_{attr}" if self.n_views != -1 else f"{idx}_{attr}"
        if dict_key in self.zip_descriptors:
            return self.zip_descriptors[dict_key]

        rgb_path = self.mp4_file_paths[idx]
        root_path, zip_name = rgb_path.parent.parent, rgb_path.stem + ".zip"
        if self.n_views != -1:
            root_path = root_path.parent
            if attr_path is not None:
                root_path = attr_path
            root_path = root_path / str(view_idx)
        else:
            if attr_path is not None:
                root_path = attr_path
        file_path_in = root_path / attr / zip_name
        zip_handle = zipfile.ZipFile(file_path_in, "r")
        self.zip_descriptors[dict_key] = zip_handle
        return zip_handle

    def _get_mp4_reader(self, idx, attr, view_idx):
        # self._check_worker_id()

        # if self.n_views != -1:
        #     dict_key = f"{idx}_{view_idx}_{attr}"
        # else:
        #     dict_key = f"{idx}_{attr}"
        # if dict_key in self.mp4_readers:
        #     return self.mp4_readers[dict_key]

        rgb_path = self.mp4_file_paths[idx]
        if self.n_views != -1:
            root_path, mp4_name = rgb_path.parent.parent.parent, rgb_path.name
        else:
            root_path, mp4_name = rgb_path.parent.parent, rgb_path.name
        if self.n_views != -1:
            root_path = root_path / str(view_idx)

        mp4_reader = VideoReader(str(root_path / attr / mp4_name), num_threads=4)
        # self.mp4_readers[dict_key] = mp4_reader
        return mp4_reader

    def available_data_fields(self) -> list[DataField]:
        return [
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.METRIC_DEPTH,
            DataField.DYNAMIC_INSTANCE_MASK,
            DataField.BACKWARD_FLOW,
            DataField.OBJECT_BBOX,
            DataField.CAPTION,
            DataField.LATENT_RGB,
        ]

    def num_videos(self) -> int:
        return len(self.mp4_file_paths)

    def num_views(self, video_idx: int) -> int:
        return 1 if self.n_views == -1 else self.n_views

    def num_frames(self, video_idx: int, view_idx: int | None = None) -> int:
        if view_idx is None:
            view_idx = self.start_view_idx
        return len(self._get_mp4_reader(video_idx, "rgb", view_idx))

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: list[int],
        view_idxs: list[int],
        data_fields: list[DataField],
    ):
        frame_indices = np.asarray(frame_idxs).astype(np.int64)
        rgb_path = self.mp4_file_paths[video_idx]
        data_base_path, data_key = rgb_path.parent.parent, rgb_path.stem
        camera_path = self.camera_path if self.camera_path is not None else data_base_path
        if self.n_views != -1:
            # Currently support only load at most one camera.
            assert len(set(view_idxs)) == 1, "Currently support only one view"
            view_idx = view_idxs[0]
            data_base_path = data_base_path.parent / str(view_idx)
            camera_path = data_base_path if self.camera_path is None else camera_path / str(view_idx)
        else:
            view_idx = self.start_view_idx
        output_dict: dict[str | DataField, Any] = {"__key__": data_key}

        for data_field in data_fields:
            if data_field == DataField.IMAGE_RGB:
                rgb_reader = self._get_mp4_reader(video_idx, "rgb", view_idx)
                rgb_read = rgb_reader.get_batch(frame_indices)
                try:
                    rgb_np = rgb_read.asnumpy()
                except AttributeError:
                    rgb_np = rgb_read.numpy()
                rgb_np = rgb_np.astype(np.float32) / 255.0
                rgb_torch = torch.from_numpy(rgb_np).moveaxis(-1, 1).contiguous()
                output_dict[data_field] = rgb_torch

                rgb_reader.seek(0)  # set video reader point back to 0 to clean up cache
                del rgb_reader

            elif data_field == DataField.LATENT_RGB:
                # Load rgb latent
                rgb_latents_path = data_base_path / "latent" / f"{data_key}.pkl"
                rgb_latents_data = _load_rgb_latents(rgb_latents_path)
                # Remove batch index
                rgb_latents_data = rgb_latents_data[0]
                output_dict[data_field] = rgb_latents_data
                # NOTE: Assuming all latents are loaded (no subsampling with frame_indices)

            elif data_field == DataField.CAMERA_C2W_TRANSFORM:
                c2w_data = np.load(camera_path / "pose" / f"{data_key}.npz")
                f_idx = np.searchsorted(c2w_data["inds"], frame_indices)
                assert np.all(c2w_data["inds"][f_idx] == frame_indices), "Pose not found"
                c2w_np = c2w_data["data"][f_idx].astype(np.float32)
                c2w_torch = torch.from_numpy(c2w_np).contiguous()
                output_dict[data_field] = c2w_torch

            elif data_field == DataField.CAMERA_INTRINSICS:
                intrinsics_data = np.load(camera_path / "intrinsics" / f"{data_key}.npz")
                f_idx = np.searchsorted(intrinsics_data["inds"], frame_indices)
                assert np.all(intrinsics_data["inds"][f_idx] == frame_indices), "Intrinsics not found"
                intrinsics_np = intrinsics_data["data"][f_idx].astype(np.float32)
                intrinsics_torch = torch.from_numpy(intrinsics_np).contiguous()
                output_dict[data_field] = intrinsics_torch

            elif data_field == DataField.METRIC_DEPTH:
                depth_zip_handle = self._get_zip_handle(video_idx, self.depth_folder, view_idx, self.depth_path)
                depth_np = []
                for frame_idx in frame_indices:
                    frame_name = f"{frame_idx:05d}.exr"
                    with depth_zip_handle.open(frame_name, "r") as f:
                        exr_file = OpenEXR.InputFile(f)
                        exr_dw = exr_file.header()["dataWindow"]
                        depth_np.append(
                            np.frombuffer(exr_file.channel("Z"), np.float16).reshape(
                                exr_dw.max.y - exr_dw.min.y + 1,
                                exr_dw.max.x - exr_dw.min.x + 1,
                            )
                        )
                depth_np = np.stack(depth_np, axis=0).astype(np.float32)
                depth_torch = torch.from_numpy(depth_np).contiguous()
                output_dict[data_field] = depth_torch

            elif data_field == DataField.OBJECT_BBOX:
                bbox_zip_handle = self._get_zip_handle(video_idx, "object_info", view_idx)
                bbox_list = []
                for frame_idx in frame_indices:
                    frame_name = f"{frame_idx:05d}.json"
                    with bbox_zip_handle.open(frame_name, "r") as f:
                        bbox_data = json.load(f)
                        bbox_list.append(bbox_data)
                output_dict[data_field] = bbox_list

            elif data_field == DataField.DYNAMIC_INSTANCE_MASK:
                mask_zip_handle = self._get_zip_handle(video_idx, "mask", view_idx)
                mask_np = []
                for frame_idx in frame_indices:
                    frame_name = f"{frame_idx:05d}.png"
                    with mask_zip_handle.open(frame_name, "r") as f:
                        mask_np.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED))
                mask_np = np.stack(mask_np, axis=0).astype(np.uint8)
                mask_torch = torch.from_numpy(mask_np).contiguous()
                output_dict[data_field] = mask_torch

            elif data_field == DataField.BACKWARD_FLOW:
                flow_zip_handle = self._get_zip_handle(video_idx, "flow", view_idx)
                flow_np = []
                for frame_idx in frame_indices:
                    frame_name = f"{frame_idx:05d}.exr"
                    with flow_zip_handle.open(frame_name, "r") as f:
                        exr_file = OpenEXR.InputFile(f)
                        exr_dw = exr_file.header()["dataWindow"]
                        height, width = (
                            exr_dw.max.y - exr_dw.min.y + 1,
                            exr_dw.max.x - exr_dw.min.x + 1,
                        )
                        flow_np.append(
                            np.stack(
                                [np.frombuffer(exr_file.channel(f"{channel}"), np.float16) for channel in ["U", "V"]],
                                axis=-1,
                            ).reshape(height, width, 2)
                        )
                flow_np = np.stack(flow_np, axis=0).astype(np.float32)
                flow_torch = torch.from_numpy(flow_np).contiguous()
                output_dict[data_field] = flow_torch

            elif data_field == DataField.CAPTION:
                caption_path = data_base_path / "caption" / f"{data_key}.txt"
                with open(caption_path) as f:
                    caption = f.read()
                output_dict[data_field] = caption
            elif data_field == "custom":
                if self.custom_folders is not None:
                    output_dict[data_field] = {}
                    assert len(self.custom_folders) == len(self.custom_fields), (
                        "Custom folders and types must have the same length"
                    )
                    for custom_folder, custom_fields in zip(self.custom_folders, self.custom_fields, strict=False):
                        if custom_fields == "ftheta_intrinsic":
                            intrinsics_data = np.load(data_base_path / custom_folder / f"{data_key}.npz")
                            f_idx = np.searchsorted(intrinsics_data["inds"], frame_indices)
                            assert np.all(intrinsics_data["inds"][f_idx] == frame_indices), "Intrinsics not found"
                            intrinsics_np = intrinsics_data["data"][f_idx].astype(np.float32)
                            intrinsics_torch = torch.from_numpy(intrinsics_np).contiguous()
                            output_dict[data_field][custom_fields] = intrinsics_torch
                        elif custom_fields in ["hdmap"]:
                            mp4_reader = self._get_mp4_reader(video_idx, custom_folder, view_idx)
                            mp4_read = mp4_reader.get_batch(frame_indices)
                            try:
                                mp4_np = mp4_read.asnumpy()
                            except AttributeError:
                                mp4_np = mp4_read.numpy()
                            mp4_np = mp4_np.astype(np.float32) / 255.0
                            mp4_torch = torch.from_numpy(mp4_np).moveaxis(-1, 1).contiguous()
                            output_dict[data_field][custom_fields] = mp4_torch

                            mp4_reader.seek(0)  # set video reader point back to 0 to clean up cache
                            del mp4_reader

                        else:
                            raise NotImplementedError(f"Can't handle custom data field {data_field}")
            else:
                raise NotImplementedError(f"Can't handle data field {data_field}")

        return output_dict
