from __future__ import annotations

import torch

from worldfoundry.operators.hunyuan_mirror_operator import HunyuanMirrorOperator


def test_reset_camera_uses_scalar_last_identity_quaternion() -> None:
    operator = HunyuanMirrorOperator()
    operator.get_interaction("reset_camera")

    result = operator.process_interaction(num_frames=2, image_hw=(240, 320))

    expected = torch.eye(3).expand(2, -1, -1)
    torch.testing.assert_close(result["camera_poses"][:, :3, :3], expected)
    torch.testing.assert_close(result["camera_poses"][:, :3, 3], torch.zeros(2, 3))


def test_camera_look_rotation_remains_proper() -> None:
    operator = HunyuanMirrorOperator()
    operator.get_interaction("camera_look_left")

    rotation = operator.process_interaction(num_frames=1)["camera_poses"][0, :3, :3]

    torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(rotation, torch.diag(torch.tensor([1.0, -1.0, -1.0])))
