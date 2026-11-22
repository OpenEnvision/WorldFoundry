import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time

import decord
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as TF
from PIL import Image

from worldfoundry.base_models.perception_core.general_perception.uniception.uniception.models.encoders.image_normalizations import (
    IMAGE_NORMALIZATION_DICT,
)
from worldfoundry.base_models.three_dimensions.general_3d.mapanything.mapanything.models import MapAnything
from worldfoundry.base_models.three_dimensions.general_3d.mapanything.mapanything.utils.geometry import (
    closed_form_pose_inverse,
)
from worldfoundry.synthesis.visual_generation.spatia.spatia_runtime.utils.camera_io import (
    read_intrinsics_from_txt,
    read_w2cs_from_txt,
)


def load_and_preprocess_videos(
    video_path,
    mask_path=[],  # noqa: B006 - Keeps the upstream default object and signature.
    read_fps=24,
    extrinsics_path=None,
    intrinsic_path=None,
    per_frame_mask=False,
):
    assert os.path.exists(video_path), f"Video file {video_path} does not exist"
    video = decord.VideoReader(video_path)
    fps = video.get_avg_fps()
    stride = int(round(fps / read_fps))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    selected_index = np.arange(0, len(video), stride)
    selected_frames = video.get_batch(selected_index).asnumpy()
    raw_images = [Image.fromarray(frame).convert("RGB") for frame in selected_frames]
    raw_masks = []
    if mask_path:
        if mask_path[0].endswith(".npy"):
            raw_masks = np.load(mask_path[0])
            if per_frame_mask:
                assert len(raw_masks) == len(video), (
                    f"The number of masks {len(raw_masks)} is not equal to the number of frames {len(video)}"
                )
                raw_masks = raw_masks[::stride]
        else:
            raw_masks = [Image.open(mask_path_i).convert("L") for mask_path_i in mask_path]
            if per_frame_mask:
                assert len(raw_masks) == len(video), (
                    f"The number of masks {len(raw_masks)} is not equal to the number of frames {len(video)}"
                )
                raw_masks = raw_masks[::stride]
        print(f"The number of masks {len(raw_masks)}")

    images = []
    masks = []
    shapes = set()
    to_tensor = TF.ToTensor()
    img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    norm_tensor = TF.Normalize(mean=img_norm.mean, std=img_norm.std)
    new_width = 518

    for f_idx, image in enumerate(raw_images):
        img = image
        width, height = img.size

        mask = np.ones((height, width), dtype=bool)
        if len(raw_masks) > 0:
            if per_frame_mask:
                mask_image = np.asarray(raw_masks[f_idx]).astype(bool)
            else:
                mask_image = np.asarray(raw_masks).astype(bool)
                mask_image = np.any(mask_image, axis=0)
            mask = mask & (~mask_image)
        mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        mask = mask.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = norm_tensor(to_tensor(img))
        mask = to_tensor(mask).to(torch.bool)
        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y : start_y + 518, :]
            mask = mask[:, start_y : start_y + 518, :]
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        masks.append(mask)

    if len(shapes) > 1:
        raise ValueError(f"Found images with different shapes: {shapes}")

    images = torch.stack(images)
    masks = torch.stack(masks)
    if len(video) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
    if extrinsics_path is not None:
        w2cs = read_w2cs_from_txt(extrinsics_path, homogeneous=True)[::stride]
    if intrinsic_path is not None:
        norm_intrinsics = read_intrinsics_from_txt(intrinsic_path)
        fx, fy, cx, cy = norm_intrinsics
        fx = fx * new_width
        fy = fy * new_height
        cx = cx * new_width
        cy = cy * new_height
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).to(images.device)
    imgs_list = []
    for img_i in images:
        data = {
            "img": img_i[None],
            "true_shape": np.int32([img_i.shape[1:]]),
            "idx": len(imgs_list),
            "instance": str(len(imgs_list)),
            "data_norm_type": ["dinov2"],
        }
        if extrinsics_path is not None:
            w2c = torch.from_numpy(w2cs[len(imgs_list)]).to(images.device)
            c2w = torch.linalg.inv(w2c)
            data["camera_poses"] = c2w[None].to(torch.float32)
            data["is_metric_scale"] = torch.tensor([True])
        if intrinsic_path is not None:
            data["intrinsics"] = intrinsics[None].to(torch.float32)
        imgs_list.append(data)
    return imgs_list, masks, (new_width, new_height), True


def denormalize_image(image):
    img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    mean = torch.as_tensor(img_norm.mean, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(img_norm.std, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
    image = image * std + mean
    return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=[], nargs="+")
    parser.add_argument("--per_frame_mask", action="store_true", default=False)
    parser.add_argument("--conf_percentile", type=float, default=0.0)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--extrinsics_path", type=str, default=None)
    parser.add_argument("--intrinsic_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = args.device
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"MapAnything checkpoint is not staged locally: {model_path}. Runtime downloads are disabled."
        )
    model = MapAnything.from_pretrained(model_path).to(device)
    vid_path = args.vid_path
    mask_path = args.mask_path
    views, masks, (new_width, new_height), _ = load_and_preprocess_videos(
        vid_path,
        mask_path,
        args.fps,
        extrinsics_path=args.extrinsics_path,
        intrinsic_path=args.intrinsic_path,
        per_frame_mask=args.per_frame_mask,
    )

    start_time = time.time()
    predictions = model.infer(
        views,
        memory_efficient_inference=len(views) > 200,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=10,
    )
    end_time = time.time()
    print(f"[INFO] MapAnything Inference time: {end_time - start_time:.2f} seconds")
    extrinsics = []
    intrinsics = []
    points_xyz = []
    points_rgb = []
    points_conf = []
    frames = torch.cat([view_i["img"] for view_i in views])
    frames = denormalize_image(frames)
    frames = frames.permute(0, 2, 3, 1).contiguous()
    for prediction_i, frame_i, mask_i in zip(predictions, frames, masks, strict=False):
        w2c = closed_form_pose_inverse(prediction_i["camera_poses"])[0, :3, :4]
        intrinsics_i = prediction_i["intrinsics"][0]
        extrinsics.append(w2c)
        intrinsics.append(intrinsics_i)

        points_xyz_i = prediction_i["pts3d"].view(-1, 3)
        points_rgb_i = frame_i.view(-1, 3)
        confidence_i = prediction_i["conf"].view(-1)
        mask_i = mask_i.view(-1)
        points_conf.append(confidence_i[mask_i])
        points_xyz.append(points_xyz_i[mask_i])
        points_rgb.append(points_rgb_i[mask_i])
    extrinsics = torch.stack(extrinsics).cpu().numpy()
    intrinsics = torch.stack(intrinsics).mean(dim=0)
    points_xyz = torch.cat(points_xyz)
    points_rgb = torch.cat(points_rgb)
    points_conf = torch.cat(points_conf)
    if args.conf_percentile > 0:
        conf_th = np.percentile(points_conf.cpu().numpy(), args.conf_percentile).item()
    else:
        conf_th = -1e9
    points_xyz = points_xyz[points_conf > conf_th]
    points_rgb = points_rgb[points_conf > conf_th]
    points_conf = points_conf[points_conf > conf_th]
    points_xyz = points_xyz.cpu().numpy()
    points_rgb = points_rgb.cpu().numpy()

    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx = intrinsics[0, 2].item()
    cy = intrinsics[1, 2].item()
    norm_fx = fx / new_width
    norm_fy = fy / new_height
    norm_cx = cx / new_width
    norm_cy = cy / new_height

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "w2c.txt"), "w") as f:
        for i in range(extrinsics.shape[0]):
            w2c_row = extrinsics[i]
            f.write(f"{w2c_row.tolist()}\n")
    with open(os.path.join(args.output_path, "intrinsics.txt"), "w") as f:
        f.write(f"[{norm_fx} {norm_fy} {norm_cx} {norm_cy}]")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(points_rgb)
    pcd = pcd.voxel_down_sample(args.voxel_size) if args.voxel_size > 0 else pcd
    o3d.io.write_point_cloud(os.path.join(args.output_path, "points.ply"), pcd)
