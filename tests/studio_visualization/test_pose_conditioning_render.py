"""Wan Animate's actual metadata-to-conditioning rendering path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("matplotlib")

from worldfoundry.studio.visualization.plugins.perception.human_pose import draw_aapose_by_meta


@pytest.mark.parametrize("draw_hand", [False, True])
def test_pose_metadata_renders_body_and_preserves_keypoints(draw_hand):
    body = np.column_stack((np.linspace(12, 42, 20), np.linspace(15, 48, 20)))
    hand = np.column_stack((np.linspace(0.2, 0.6, 21), np.linspace(0.3, 0.7, 21)))
    meta = SimpleNamespace(kps_body=body.copy(), kps_body_p=np.ones(20),
                           kps_lhand=hand.copy(), kps_lhand_p=np.ones(21),
                           kps_rhand=hand.copy(), kps_rhand_p=np.ones(21))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    rendered = draw_aapose_by_meta(image, meta, draw_hand=draw_hand)
    assert rendered.shape == image.shape
    assert rendered.dtype == np.uint8
    assert np.count_nonzero(rendered) > 0
    np.testing.assert_array_equal(meta.kps_body, body)
    np.testing.assert_array_equal(meta.kps_lhand, hand)


def test_low_confidence_pose_does_not_generate_conditioning_pixels():
    body = np.zeros((20, 2))
    hand = np.zeros((21, 2))
    meta = SimpleNamespace(kps_body=body, kps_body_p=np.zeros(20),
                           kps_lhand=hand, kps_lhand_p=np.zeros(21),
                           kps_rhand=hand, kps_rhand_p=np.zeros(21))
    rendered = draw_aapose_by_meta(np.zeros((64, 64, 3), dtype=np.uint8), meta)
    assert np.count_nonzero(rendered) == 0
