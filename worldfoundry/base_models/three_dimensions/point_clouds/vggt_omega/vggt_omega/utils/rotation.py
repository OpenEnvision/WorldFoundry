# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Compatibility names for WorldFoundry's shared XYZW rotation helpers."""

from worldfoundry.core.geometry.transforms import (
    quaternion_xyzw_to_rotation_matrix as quat_to_mat,
    rotation_matrix_to_quaternion_xyzw as mat_to_quat,
    standardize_quaternion_xyzw as standardize_quaternion,
)

__all__ = ["mat_to_quat", "quat_to_mat", "standardize_quaternion"]
