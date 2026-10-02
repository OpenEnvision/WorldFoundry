import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import itertools
import warnings
import logging
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.three_dimensions.depth.moge.model import (
    import_model_class_by_version,
)
from worldfoundry.base_models.three_dimensions.depth.moge.utils.io import (
    save_glb,
    save_ply,
)
from worldfoundry.base_models.three_dimensions.general_3d.eastern_journalist.utils3d import (
    numpy as utils3d_np,
)
from worldfoundry.core.media.artifacts import colorize_depth_map

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(
    input_path: str,
    fov_x_: float,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    resolution_level: int,
    num_tokens: int,
    threshold: float,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):

    device = torch.device(device_name)
    logger.debug('Torch device: %s', device)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]

    if len(image_paths) == 0:
        logger.error('No image files found in %s', input_path)
        raise FileNotFoundError(f'No image files found in {input_path}')

    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    logger.info('Loaded model: %s on %s', pretrained_model_name_or_path, device)
    if use_fp16:
        model.half()

    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", or "--ply" to specify the output.')
        save_maps_ = save_glb_ = save_ply_ = True

    # Helper: replace infs/non-positive with max(valid)+10
    def replace_infs_with_max(depth: np.ndarray) -> np.ndarray:
        d = depth.copy()
        valid_mask = np.isfinite(d) & (d > 0)
        if not np.any(valid_mask):
            logger.warning('No valid depth values found; cannot compute replacement for infinities.')
            raise ValueError('No valid depth values found to compute replacement for infinities.')
        max_valid = float(np.max(d[valid_mask]))
        replacement = max_valid + 10.0
        inf_mask = ~valid_mask
        d[inf_mask] = replacement
        return d

    def grayscale_depth(depth: np.ndarray) -> np.ndarray:
        """Convert a depth map to 3-channel grayscale (closer = whiter)."""
        depth_f = depth.astype(np.float64)
        valid = (depth_f > 0) & np.isfinite(depth_f)
        disp = np.zeros_like(depth_f, dtype=np.float64)
        disp[valid] = 1.0 / depth_f[valid]

        if np.any(valid):
            sampled = disp[valid]
            try:
                min_disp = float(np.quantile(sampled, 0.001))
                max_disp = float(np.quantile(sampled, 0.99))
            except Exception:
                min_disp = float(np.min(sampled))
                max_disp = float(np.max(sampled))
            if max_disp - min_disp <= 1e-8:
                disp_norm = (disp - min_disp)
            else:
                disp_norm = (disp - min_disp) / (max_disp - min_disp)
        else:
            disp_norm = disp

        disp_norm = np.clip(disp_norm, 0.0, 1.0)
        gray = (disp_norm * 255.0).astype(np.uint8)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        return gray_rgb

    for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
        if not image_path.exists():
            logger.error('File %s does not exist', image_path)
            raise FileNotFoundError(f'File {image_path} does not exist.')
        logger.info('Processing image: %s', image_path)
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        logger.debug('Image tensor shape (C,H,W): %s', tuple(image_tensor.shape))

        # Inference
        output = model.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
        logger.info('Inference finished for %s', image_path)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        normal = output['normal'].cpu().numpy() if 'normal' in output else None

        # save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        save_path = Path(output_path)
        save_path.mkdir(exist_ok=True, parents=True)

        # Save images / maps
        if save_maps_:
            # cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Replace infinities in depth before visualization / saving
            try:
                depth_fixed = replace_infs_with_max(depth)
            except ValueError:
                # fallback: use original depth if replacement not possible
                depth_fixed = depth.copy()
                logger.warning('Falling back to original depth values for %s', image_path)

            # Color visualization (existing style)
            depth_vis_path = save_path / 'depth_vis.png'
            cv2.imwrite(
                str(depth_vis_path),
                cv2.cvtColor(colorize_depth_map(depth_fixed), cv2.COLOR_RGB2BGR),
            )
            logger.info('Saved depth visualization to %s', depth_vis_path)

            # Grayscale visualization (closer = whiter)
            gray_rgb = grayscale_depth(depth_fixed)
            depth_gray_path = save_path / 'depth_gray.png'
            cv2.imwrite(str(depth_gray_path), cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR))
            logger.info('Saved grayscale depth to %s', depth_gray_path)

            # Save depth as EXR (float) and also save a compact fp16 npz with intrinsics
            # cv2.imwrite(str(save_path / 'depth.exr'), depth_fixed, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            npz_path = save_path / 'depth_intrinsics.npz'
            np.savez_compressed(str(npz_path), depth=depth_fixed.astype(np.float16), intrinsic=intrinsics.astype(np.float16))
            logger.info('Saved depth (fp16) and intrinsics to %s', npz_path)

            # cv2.imwrite(str(save_path / 'mask.png'), (mask * 255).astype(np.uint8))
            # cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            # if normal is not None:
            # fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
            # with open(save_path / 'fov.json', 'w') as f:
            #     json.dump({
            #         'fov_x': round(float(np.rad2deg(fov_x)), 2),
            #         'fov_y': round(float(np.rad2deg(fov_y)), 2),
            #     }, f)

        # Export mesh & visulization
        if save_glb_ or save_ply_ or show:
            mask_cleaned = mask & ~utils3d_np.depth_map_edge(depth, rtol=threshold)
            if normal is None:
                faces, vertices, vertex_colors, vertex_uvs = utils3d_np.build_mesh_from_map(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d_np.uv_map(height, width),
                    mask=mask_cleaned,
                    tri=True
                )
                vertex_normals = None
            else:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d_np.build_mesh_from_map(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d_np.uv_map(height, width),
                    normal,
                    mask=mask_cleaned,
                    tri=True
                )
            # When exporting the model, follow the OpenGL coordinate conventions:
            # - world coordinate system: x right, y up, z backward.
            # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
            if normal is not None:
                vertex_normals = vertex_normals * [1, -1, -1]

            if save_glb_:
                save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)
                logger.info('Saved GLB to %s', save_path / 'mesh.glb')

        if save_ply_:
            save_ply(save_path / 'pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)
            logger.info('Saved PLY to %s', save_path / 'pointcloud.ply')

        if show:
            import trimesh
            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                vertex_normals=vertex_normals,
                faces=faces,
                process=False
            ).show()
            logger.info('Displayed mesh for %s', image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--input', '-i', dest='input_path', required=True, help='Input image or folder path. "jpg" and "png" are supported.')
    parser.add_argument('--fov_x', dest='fov_x_', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
    parser.add_argument('--output', '-o', dest='output_path', default='./output', help='Output folder path')
    parser.add_argument('--pretrained', dest='pretrained_model_name_or_path', default=None, help='Pretrained model name or path. If not provided, the corresponding default model will be chosen.')
    parser.add_argument('--version', dest='model_version', choices=['v1', 'v2'], default='v2', help='Model version. Defaults to "v2"')
    parser.add_argument('--device', dest='device_name', default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
    parser.add_argument('--fp16', dest='use_fp16', action='store_true', help='Use fp16 precision for much faster inference.')
    parser.add_argument('--resize', dest='resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
    parser.add_argument('--resolution_level', dest='resolution_level', type=int, default=9, help='Resolution level for inference [0-9]')
    parser.add_argument('--num_tokens', dest='num_tokens', type=int, default=None, help='Number of tokens used for inference. Overrides resolution_level if provided.')
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.04, help='Threshold for removing edges.')
    parser.add_argument('--maps', dest='save_maps_', action='store_true', help='Whether to save the output maps (image, point map, depth map, normal map, mask) and fov.')
    parser.add_argument('--glb', dest='save_glb_', action='store_true', help='Whether to save the output as a .glb file.')
    parser.add_argument('--ply', dest='save_ply_', action='store_true', help='Whether to save the output as a .ply file.')
    parser.add_argument('--show', dest='show', action='store_true', help='Whether to show the output in a window.')

    args = parser.parse_args()
    main(
        input_path=args.input_path,
        fov_x_=args.fov_x_,
        output_path=args.output_path,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        model_version=args.model_version,
        device_name=args.device_name,
        use_fp16=args.use_fp16,
        resize_to=args.resize_to,
        resolution_level=args.resolution_level,
        num_tokens=args.num_tokens,
        threshold=args.threshold,
        save_maps_=args.save_maps_,
        save_glb_=args.save_glb_,
        save_ply_=args.save_ply_,
        show=args.show,
    )

    logger.info("=" * 80)
    logger.info("Depth Estimation complete!")
    logger.info("=" * 80)
