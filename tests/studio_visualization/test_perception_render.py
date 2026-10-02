from __future__ import annotations

import numpy as np

from worldfoundry.studio.visualization.plugins.perception.render import (
    as_rgb_uint8,
    render_depth,
    render_detections,
    render_feature_pca,
    render_keypoints,
    render_masks,
    render_normals,
    render_optical_flow,
    render_text_overlay,
    render_tracks,
)


def test_detection_and_mask_renderers_are_nonblank():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    detected = render_detections(image, [[5, 6, 35, 30]], labels=["object"], scores=[0.9])
    masked = render_masks(image, np.pad(np.ones((20, 20), dtype=np.uint8), ((10, 10), (20, 20))))

    assert detected.shape == image.shape and detected.std() > 0
    assert masked.shape == image.shape and masked.std() > 0


def test_detection_renderer_clips_model_boxes_outside_the_image():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    rendered = render_detections(image, [[-8, -5, 70, 50]], labels=["object"])
    assert rendered.shape == image.shape
    assert rendered.std() > 0


def test_optical_flow_uses_official_color_wheel():
    flow = np.zeros((24, 32, 2), dtype=np.float32)
    flow[..., 0] = np.linspace(-2, 2, 32)
    rendered = render_optical_flow(flow)

    assert rendered.shape == (24, 32, 3)
    assert rendered.dtype == np.uint8
    assert np.unique(rendered.reshape(-1, 3), axis=0).shape[0] > 8


def test_track_renderer_preserves_frame_count_and_draws_motion():
    frames = np.zeros((3, 32, 48, 3), dtype=np.uint8)
    tracks = np.asarray([[[5, 10]], [[15, 12]], [[25, 14]]], dtype=np.float32)
    rendered = render_tracks(frames, tracks, trace_length=3)

    assert len(rendered) == 3
    assert all(frame.shape == (32, 48, 3) for frame in rendered)
    assert rendered[-1].std() > 0


def test_feature_pca_accepts_patch_grid_and_resizes():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(4, 6, 12)).astype(np.float32)
    rendered = render_feature_pca(features, output_size=(60, 40))

    assert rendered.shape == (40, 60, 3)
    assert rendered.std() > 0


def test_depth_and_normal_renderers_accept_official_array_layouts():
    depth = np.linspace(0.2, 4.0, 24 * 32, dtype=np.float32).reshape(24, 32)
    normals = np.zeros((3, 24, 32), dtype=np.float32)
    normals[2] = 1.0

    depth_image = render_depth(depth)
    normal_image = render_normals(normals)

    assert depth_image.shape == (24, 32, 3) and depth_image.std() > 0
    assert normal_image.shape == (24, 32, 3) and normal_image.std() > 0


def test_keypoint_and_text_renderers_draw_nonblank_overlays():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    keypoints = np.asarray([[5, 6, 0.9], [30, 20, 0.8], [50, 32, 0.7]])

    posed = render_keypoints(image, keypoints, edges=[[0, 1], [1, 2]], score_threshold=0.5)
    captioned = render_text_overlay(image, "running through the scene")

    assert posed.std() > 0
    assert captioned.std() > 0


def test_rgb_normalization_accepts_chw_float_image():
    image = np.ones((3, 8, 10), dtype=np.float32) * 0.5
    rendered = as_rgb_uint8(image)

    assert rendered.shape == (8, 10, 3)
    assert 126 <= int(rendered.mean()) <= 128
