#!/usr/bin/env python3
"""Compare real v1.5 weights with the pinned official runtime on local GPUs.

Run prepare once, then prepare-depth for the official video example (which
ships without depth), then native/upstream in separate processes and GPUs.
Compare includes latent tensors, decoded RGB frames, masks and view selection.
Use --direct to exercise raw RGB -> native DA3 -> video, then replay the
generated geometry through upstream; this does not compare DA3 implementations.
No dependencies or weights are downloaded or modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
UPSTREAM_REVISION = "dd3561f544053fe22d739b6b2f8461c9c97bf8cb"
CASES = ("image_example_00", "multiview_example_00", "video_example_00")


def check_upstream(upstream):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected pinned upstream {UPSTREAM_REVISION}, got {revision}")


def prepare(upstream, directory, cases):
    import numpy as np

    check_upstream(upstream)
    for case in cases:
        source, target = upstream / "examples" / case, directory / "fixtures" / case
        target.mkdir(parents=True, exist_ok=False)
        meta = json.loads((source / "scene.json").read_text())
        meta.update(frames=21, valid_frames=20, output_id=case)
        if meta["kind"] == "video":
            meta.update(views=21, valid_frames=21)
        for subdir in ("input", "depth"):
            (target / subdir).mkdir()
            if not (source / subdir).is_dir():
                continue
            for path in (source / subdir).iterdir():
                if path.suffix == ".mp4":
                    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
                                    "-frames:v", "21", "-c:v", "libx264rgb", "-crf", "0",
                                    str(target / subdir / path.name)], check=True)
                elif path.is_file():
                    shutil.copyfile(path, target / subdir / path.name)
        for relative, size, count in (("input/target_tcw.txt", 4, 21),
                                      ("depth/source_intrinsics.txt", 3, meta["views"]),
                                      ("depth/source_tcw.txt", 4, meta["views"])):
            if not (source / relative).is_file():
                continue
            values = np.loadtxt(source / relative, ndmin=2).reshape(-1, size * size)
            np.savetxt(target / relative, values[:count])
        (target / "scene.json").write_text(json.dumps(meta, indent=2))


def prepare_depth(args):
    from worldfoundry.synthesis.visual_generation.inspatio_world.v15_runtime import InspatioWorldV15Runtime
    from worldfoundry.synthesis.visual_generation.inspatio_world.v15_scene import validate_scene

    runtime = InspatioWorldV15Runtime(da3_model_path=str(args.checkpoints / "depth-anything--DA3NESTED-GIANT-LARGE-1.1"),
                                    device="cuda:0")
    for case in args.cases:
        scene = args.output / "fixtures" / case
        record = {**json.loads((scene / "scene.json").read_text()), "id": "scene", "path": str(scene)}
        if not (scene / "depth/metadata.txt").is_file():
            runtime._estimate_depth(record)
        validate_scene(record)
        print(f"depth ready {case}", flush=True)


def backend_root(args, name):
    return args.output / (name + ("-direct" if args.direct else ""))


def scene_path(args, case):
    return args.output / ("fixtures-direct" if args.direct else "fixtures") / case


def run_native(args):
    import torch

    from worldfoundry.pipelines.inspatio_world.pipeline_inspatio_world_v15 import InspatioWorldV15Pipeline

    pipeline = InspatioWorldV15Pipeline.from_pretrained(
        str(args.checkpoints / "inspatio--world-1.5"),
        wan_model_path=str(args.checkpoints / "Wan-AI--Wan2.1-T2V-1.3B"), device="cuda:0",
        da3_model_path=str(args.checkpoints / "depth-anything--DA3NESTED-GIANT-LARGE-1.1"),
    )
    for case in args.cases:
        output = backend_root(args, "native") / case
        fixture = args.output / "fixtures" / case
        if args.direct:
            if scene_path(args, case).exists():
                raise FileExistsError(f"Refusing to overwrite direct replay fixture: {scene_path(args, case)}")
            meta = json.loads((fixture / "scene.json").read_text())
            direct_inputs = ({"videos": str(fixture / "input/video.mp4")} if meta["kind"] == "video" else
                             {"images": [str(path) for path in sorted((fixture / "input").glob("view_*.png"))]})
            request = {**direct_inputs, "prompt": (fixture / "input/prompt.txt").read_text().strip() or "A moving camera.",
                       "traj_txt_path": str(fixture / "input/target_tcw.txt")}
        else:
            request = {"scene_dir": str(fixture)}
        result = pipeline(**request,
                          output_path=str(output / "pred.mp4"), seed=0, return_latents=True)
        torch.save(result.pop("latents"), output / "latents.pt")
        if args.direct:
            shutil.copytree(Path(result["work_dir"]) / ".prepared", scene_path(args, case))
        source = Path(result["work_dir"]) / "scene"
        for name in ("source", "render", "mask"):
            shutil.copyfile(source / f"{name}.mp4", output / f"{name}.mp4")
        (output / "report.json").write_text(json.dumps({"case": case, **result}, indent=2))
        print(f"native completed {case}", flush=True)


def run_upstream(args):
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    check_upstream(args.upstream)
    sys.path.insert(0, str(args.upstream))
    from datasets.scene_dataset import SceneDataset
    from datasets.utils import iter_mask_latents
    from inference import decode_stream, encode_stream, set_seed
    from pipeline.causal_inference import CausalInferencePipeline
    from pipeline.render_scene import render_one
    from pipeline.video_writer import video_writer

    wan = args.checkpoints / "Wan-AI--Wan2.1-T2V-1.3B"
    config = OmegaConf.merge(OmegaConf.load(args.upstream / "configs/default_config.yaml"),
                             OmegaConf.load(args.upstream / "configs/inference_1.3b.yaml"))
    config.wan_model_folder = str(wan)
    config.generator.model_path = str(wan)
    pipeline = CausalInferencePipeline(config)
    weights = args.checkpoints / "inspatio--world-1.5/InSpatio-World-1.5-1.3B.safetensors"
    pipeline.generator.load_state_dict(load_file(str(weights)), strict=True)
    pipeline = pipeline.to(dtype=torch.bfloat16).eval()
    pipeline.text_encoder.to("cuda:0")
    pipeline.generator.to("cuda:0")
    pipeline.vae.to("cuda:0")
    with torch.no_grad():
        for case in args.cases:
            scene = scene_path(args, case)
            record = {**json.loads((scene / "scene.json").read_text()), "id": case,
                      "path": str(scene), "scene_id": case}
            output = backend_root(args, "upstream") / case
            render_one(record, torch.device("cuda:0"), backend_root(args, "upstream"))
            source_type = "video" if record["kind"] == "video" else ("image" if record["views"] == 1 else "multi-image")
            item = {"valid_frames": record["valid_frames"], "source_type": source_type,
                    **{f"{name}_video": str(output / f"{name}.mp4") for name in ("source", "render", "mask")}}
            manifest = output / "manifest.json"
            manifest.write_text(json.dumps([item]))
            dataset = SceneDataset(manifest)
            set_seed(0)
            render = encode_stream(dataset.frames(item, "render"), pipeline, "cuda:0")
            source = encode_stream(dataset.frames(item, "source"), pipeline, "cuda:0")
            masks = torch.cat(list(iter_mask_latents(dataset.frames(item, "mask"), "cuda:0", torch.bfloat16)), dim=1)
            noise = torch.randn_like(source)
            latents = pipeline.inference(noise, [(scene / "input/prompt.txt").read_text().strip()], source, render, masks)
            torch.save(latents.cpu(), output / "latents.pt")
            written = 0
            chunks = decode_stream(latents, pipeline, "cuda:0")
            try:
                with video_writer(output / "pred.mp4", record["fps"]) as writer:
                    for frames in chunks:
                        count = min(frames.shape[1], record["valid_frames"] - written)
                        if count:
                            pixels = (frames[0, :count].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                            writer.write(pixels.tobytes())
                            written += count
                        if written == record["valid_frames"]:
                            break
                assert written == record["valid_frames"]
            finally:
                chunks.close()
            (output / "report.json").write_text(json.dumps({"case": case, "frames": written}, indent=2))
            print(f"upstream completed {case}", flush=True)


def compare(args):
    import numpy as np
    import torch
    from geometry_regression import sha256

    from worldfoundry.core.media.codecs.video import load_video_frames
    from worldfoundry.synthesis.visual_generation.inspatio_world.v15_runtime import (
        CHECKPOINT_FILENAME,
        CHECKPOINT_SHA256,
    )

    check_upstream(args.upstream)
    weight_hash = sha256(args.checkpoints / "inspatio--world-1.5" / CHECKPOINT_FILENAME)
    if weight_hash != CHECKPOINT_SHA256:
        raise ValueError("Regression requires the pinned released v1.5 checkpoint")
    records = []
    latent_records = []
    for case in args.cases:
        native, upstream = (backend_root(args, backend) / case for backend in ("native", "upstream"))
        actual = torch.load(native / "latents.pt", map_location="cpu", weights_only=True)
        expected = torch.load(upstream / "latents.pt", map_location="cpu", weights_only=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        latent_records.append({"case": case, "shape": list(actual.shape), "dtype": str(actual.dtype),
                               "sha256": hashlib.sha256(actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
        for name in ("source", "render", "mask", "pred"):
            a = np.stack([np.asarray(frame) for frame in load_video_frames(native / f"{name}.mp4")])
            b = np.stack([np.asarray(frame) for frame in load_video_frames(upstream / f"{name}.mp4")])
            np.testing.assert_array_equal(a, b)
            records.append({"case": case, "field": name, "frames": len(a), "exact": True,
                            "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
    comparison = args.output / ("comparison-direct.json" if args.direct else "comparison.json")
    comparison.write_text(json.dumps({"upstream_revision": UPSTREAM_REVISION, "direct": args.direct,
                                      "checkpoint_sha256": weight_hash, "results": records,
                                      "latent_outputs": latent_records, "latent_exact": True}, indent=2))
    print(f"Exact upstream parity: {len(args.cases)} latent outputs and {len(records)} decoded videos", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "prepare-depth", "native", "upstream", "compare"))
    parser.add_argument("--direct", action="store_true", help="Validate direct RGB inputs and replay their generated geometry")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": lambda: prepare(args.upstream, args.output, args.cases), "prepare-depth": lambda: prepare_depth(args),
     "native": lambda: run_native(args),
     "upstream": lambda: run_upstream(args), "compare": lambda: compare(args)}[args.mode]()


if __name__ == "__main__":
    main()
