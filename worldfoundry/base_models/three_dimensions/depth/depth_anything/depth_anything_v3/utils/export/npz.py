# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> three_dimensions -> depth -> depth_anything -> depth_anything_v3 -> utils -> export -> npz.py functionality."""

import os
import numpy as np

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.specs import Prediction
from worldfoundry.core.execution.parallel_execution import async_call


@async_call
def export_to_npz(
    prediction: Prediction,
    export_dir: str,
):
    """Export to npz.

    Args:
        prediction: The prediction.
        export_dir: The export dir.
    """
    output_file = os.path.join(export_dir, "exports", "npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    image = prediction.processed_images  # (N,H,W,3) uint8

    # Build save dict with only non-None values
    save_dict = {
        "image": image,
        "depth": np.round(prediction.depth, 6),
    }

    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics

    # aux = {k: np.round(v, 4) for k, v in prediction.aux.items()}
    np.savez_compressed(output_file, **save_dict)


@async_call
def export_to_mini_npz(
    prediction: Prediction,
    export_dir: str,
):
    """Export to mini npz.

    Args:
        prediction: The prediction.
        export_dir: The export dir.
    """
    output_file = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Build save dict with only non-None values
    save_dict = {
        "depth": np.round(prediction.depth, 8),
    }

    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics

    np.savez_compressed(output_file, **save_dict)
