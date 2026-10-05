from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

from worldgen import WorldGen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldGen offline inference")
    parser.add_argument("--prompt", "-p", type=str, default="a 3D scene", help="Prompt for world generation")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to an input image")
    parser.add_argument("--pano_image", type=str, default=None, help="Path to an input panorama image")
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Directory for generated artifacts")
    parser.add_argument("--resolution", "-r", type=int, default=1600, help="Generated world resolution")
    parser.add_argument("--use_sharp", action="store_true", help="Enable ml-sharp generation")
    parser.add_argument("--inpaint_bg", action="store_true", help="Inpaint panorama background")
    parser.add_argument("--return_mesh", action="store_true", help="Save a mesh artifact instead of gaussian splats")
    parser.add_argument("--low_vram", action="store_true", help="Enable low VRAM mode")
    parser.add_argument("--save_scene", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_viewer", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _maybe_enable_low_vram(args: argparse.Namespace) -> None:
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / (1024**3) < 24:
        args.low_vram = True


def generate_scene(args: argparse.Namespace):
    if args.return_mesh and args.inpaint_bg:
        raise ValueError("inpaint_bg is not supported when return_mesh is True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "t2s" if args.image is None else "i2s"
    worldgen = WorldGen(
        mode=mode,
        use_sharp=args.use_sharp,
        inpaint_bg=args.inpaint_bg,
        resolution=args.resolution,
        device=device,
        low_vram=args.low_vram,
        initialize_pano_generator=args.pano_image is None,
    )

    if args.pano_image is not None:
        pano_image = Image.open(args.pano_image).convert("RGB").resize((2048, 1024))
        return worldgen._generate_world(pano_image, return_mesh=args.return_mesh)
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        return worldgen.generate_world(args.prompt, image, return_mesh=args.return_mesh)
    return worldgen.generate_world(args.prompt, return_mesh=args.return_mesh)


def save_scene(scene, output_dir: str | os.PathLike[str], *, return_mesh: bool) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if return_mesh:
        import numpy as np
        import trimesh

        output_path = output_root / "worldgen.glb"
        vertices = np.asarray(scene.vertices)
        triangles = np.asarray(scene.triangles)
        vertex_colors = np.asarray(scene.vertex_colors)
        if len(vertex_colors) != len(vertices):
            vertex_colors = None
        else:
            vertex_colors = np.rint(np.clip(vertex_colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_colors=vertex_colors,
            process=False,
        ).export(output_path, file_type="glb")
    else:
        output_path = output_root / "worldgen.ply"
        scene.save(str(output_path))
    return output_path


def main() -> None:
    args = parse_args()
    _maybe_enable_low_vram(args)
    scene = generate_scene(args)
    output_path = save_scene(scene, args.output_dir, return_mesh=args.return_mesh)
    print(f"Saved WorldGen artifact to {output_path}")


if __name__ == "__main__":
    main()
