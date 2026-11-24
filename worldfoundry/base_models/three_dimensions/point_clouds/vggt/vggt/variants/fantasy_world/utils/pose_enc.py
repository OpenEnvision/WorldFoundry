"""Compatibility exports for FantasyWorld's legacy VGGT path."""

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils import pose_enc as _shared_pose_enc

extri_intri_to_pose_encoding = _shared_pose_enc.extri_intri_to_pose_encoding


def pose_encoding_to_extri_intri(*args, **kwargs):
    extrinsics, intrinsics = _shared_pose_enc.pose_encoding_to_extri_intri(*args, **kwargs)
    pose_encoding = args[0] if args else kwargs.get("pose_encoding")
    if intrinsics is not None and pose_encoding is not None:
        intrinsics = intrinsics.to(dtype=pose_encoding.dtype)
    return extrinsics, intrinsics


__all__ = ["extri_intri_to_pose_encoding", "pose_encoding_to_extri_intri"]
