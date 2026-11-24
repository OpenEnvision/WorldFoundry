import argparse
import os
import time

import decord
import numpy as np
import open3d as o3d
import torch
from diffusers.utils import export_to_video
from PIL import Image

from worldfoundry.synthesis.visual_generation.spatia.spatia_runtime.utils.camera_io import (
    read_intrinsics_from_txt,
    read_w2cs_from_txt,
)

from .render_point_torch3d import render_multi_view_pointcloud


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="outputs/table/point/iter0123_gen_video_cam.ply")
    parser.add_argument("--w2c_path", type=str, default="test_case/table/pose-table-interp_121.txt")
    parser.add_argument("--normed_intrinsics", type=str, default=None)
    parser.add_argument("--downsample", action="store_true", default=False)
    parser.add_argument("--point_coordinate", type=str, default="opencv", choices=["opencv", "opengl"])
    parser.add_argument("--img_vid_path", type=str, default="test_case/table/table.jpg")
    parser.add_argument("--output_path", type=str, default="test_case/table/iter4/")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--score_render", action="store_true", default=False)
    parser.add_argument("--dummy_score", action="store_true", default=False)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--render_batchsize", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def downsample_dense_point_cloud(pcd, voxel_size):
    """Reduce dense point cloud redundancy with voxel downsampling."""
    print("--- Starting voxel downsampling ---")
    points_before_downsample = len(pcd.points)
    print(f"Points before downsampling: {points_before_downsample}")

    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    points_after_downsample = len(pcd_downsampled.points)
    print(f"Points after downsampling: {points_after_downsample}")

    reduction_percent = (points_before_downsample - points_after_downsample) / points_before_downsample * 100
    print(f"Point count reduced by {reduction_percent:.2f}%")
    print("--- Voxel downsampling completed ---\n")

    return pcd_downsampled


def render_cams(args):
    if args.width is None or args.height is None:
        if args.img_vid_path.endswith(".jpg") or args.img_vid_path.endswith(".png"):
            image = Image.open(args.img_vid_path)
            args.width = image.width
            args.height = image.height
        else:
            video_reader = decord.VideoReader(args.img_vid_path)
            vid_height, vid_width = video_reader[0].shape[:2]
            args.width = vid_width
            args.height = vid_height
    width = args.width
    height = args.height
    matrix_extrinsic = read_w2cs_from_txt(args.w2c_path)
    normed_intrinsics = read_intrinsics_from_txt(args.normed_intrinsics)
    pcd = o3d.io.read_point_cloud(args.data_path)
    points_xyz = np.asarray(pcd.points)
    points_rgb = np.asarray(pcd.colors)
    points_scores = np.ones_like(points_rgb) * 1.0
    print("Warning: points_xyz_scores[...,3:]=1.")
    points_xyz_rgb = np.concatenate([points_xyz, points_rgb], axis=1)
    points_xyz_scores = np.concatenate([points_xyz, points_scores], axis=1)
    if args.dummy_score:
        print("Warning: points_xyz_scores[...,3:]=1.")
        points_xyz_scores[..., 3:] = 1.0

    if args.downsample:
        print(f"before voxel downsample points_xyz_rgb.shape: {points_xyz_rgb.shape}")
        voxel_size = args.voxel_size
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz_rgb[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(points_xyz_rgb[:, 3:].astype(np.float64))
        pcd = downsample_dense_point_cloud(pcd, voxel_size=voxel_size)
        pcd_scores = o3d.geometry.PointCloud()
        pcd_scores.points = o3d.utility.Vector3dVector(points_xyz_scores[:, :3].astype(np.float64))
        pcd_scores.colors = o3d.utility.Vector3dVector(points_xyz_scores[:, 3:].astype(np.float64))
        pcd_scores = downsample_dense_point_cloud(pcd_scores, voxel_size=voxel_size)
        points_xyz = np.asarray(pcd.points)
        points_rgb = np.asarray(pcd.colors)
        points_scores = np.asarray(pcd_scores.colors)
        points_xyz_rgb = np.concatenate([points_xyz, points_rgb], axis=1)
        points_xyz_scores = np.concatenate([points_xyz, points_scores], axis=1)
        print(f"after voxel downsample points_xyz_rgb.shape: {points_xyz_rgb.shape}")

    if args.point_coordinate == "opengl":
        print("INFO: convert point opengl coordinate to opencv coordinate")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz_rgb[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(points_xyz_rgb[:, 3:].astype(np.float64))
        pcd.transform(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        points_xyz = np.asarray(pcd.points)
        points_rgb = np.asarray(pcd.colors)

        pcd_scores = o3d.geometry.PointCloud()
        pcd_scores.points = o3d.utility.Vector3dVector(points_xyz_scores[:, :3].astype(np.float64))
        pcd_scores.colors = o3d.utility.Vector3dVector(points_xyz_scores[:, 3:].astype(np.float64))
        pcd_scores.transform(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        points_scores = np.asarray(pcd_scores.colors)
        points_xyz_rgb = np.concatenate([points_xyz, points_rgb], axis=1)
        points_xyz_scores = np.concatenate([points_xyz, points_scores], axis=1)

    render_intrinsics = [
        normed_intrinsics[0].item() * width,
        normed_intrinsics[1].item() * height,
        normed_intrinsics[2].item() * width,
        normed_intrinsics[3].item() * height,
    ]
    points_xyz_rgb = torch.from_numpy(points_xyz_rgb).to(torch.float32)
    points_xyz_scores = torch.from_numpy(points_xyz_scores).to(torch.float32)
    matrix_extrinsic = torch.from_numpy(matrix_extrinsic).to(torch.float32)

    torch.cuda.empty_cache()
    start_time = time.time()
    rendered_images = render_multi_view_pointcloud(
        points_xyz_rgb=points_xyz_rgb.to(args.device),
        w2c_matrices=matrix_extrinsic.to(args.device),
        normalized_intrinsics=render_intrinsics,
        image_size=(height, width),
        batch_size=args.render_batchsize,
    )
    rendered_images = (rendered_images.cpu().numpy() * 255).astype(np.uint8)
    rendered_images = [Image.fromarray(rendered_image) for rendered_image in rendered_images]

    if args.score_render:
        rendered_scores = render_multi_view_pointcloud(
            points_xyz_rgb=points_xyz_scores.to(args.device),
            w2c_matrices=matrix_extrinsic.to(args.device),
            normalized_intrinsics=render_intrinsics,
            image_size=(height, width),
            batch_size=args.render_batchsize,
        )
        rendered_scores = (rendered_scores.cpu().numpy() * 255).astype(np.uint8)
        rendered_scores = [Image.fromarray(rendered_score) for rendered_score in rendered_scores]
    end_time = time.time()
    print(f"[INFO] Render time: {end_time - start_time:.2f} seconds")
    os.makedirs(args.output_path, exist_ok=True)
    export_to_video(rendered_images, os.path.join(args.output_path, "render.mp4"), fps=24)
    if args.score_render:
        export_to_video(rendered_scores, os.path.join(args.output_path, "render-score.mp4"), fps=24)


if __name__ == "__main__":
    args = get_args()
    render_cams(args)
