"""Small launcher for the official WildDet3D inference API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "checkpoint-path", "lingbot-config-path", "image-path", "output-path"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--intrinsics-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classes", default="person, car")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--score-3d-threshold", type=float, default=0.1)
    return parser


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    from PIL import Image
    import huggingface_hub

    classes = [label.strip() for label in args.classes.split(",") if label.strip()]
    if not classes:
        raise ValueError("WildDet3D requires at least one comma-separated detection class")
    with Image.open(args.image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    intrinsics = np.load(args.intrinsics_path) if args.intrinsics_path else None
    if intrinsics is not None and (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()):
        raise ValueError("intrinsics must contain one finite 3x3 matrix")

    source = str(args.source_dir.resolve())
    sys.path.insert(0, source)
    from wilddet3d import build_model, preprocess
    from wilddet3d.model import WildDet3D

    # Upstream's skip_pretrained path still requests model.pt to read only its
    # architecture config. Redirect just that exact request to the staged file.
    original_download = huggingface_hub.hf_hub_download
    original_load = WildDet3D.load_state_dict

    def local_lingbot_config(*, repo_id, filename, **kwargs):
        if repo_id == "robbyant/lingbot-depth-postrain-dc-vitl14" and filename == "model.pt":
            return str(args.lingbot_config_path.resolve())
        return original_download(repo_id=repo_id, filename=filename, **kwargs)

    def checked_load(self, state_dict, *load_args, **load_kwargs):
        report = original_load(self, state_dict, *load_args, **load_kwargs)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "WildDet3D checkpoint does not exactly cover the model: "
                f"missing={report.missing_keys[:12]} ({len(report.missing_keys)} total), "
                f"unexpected={report.unexpected_keys[:12]} ({len(report.unexpected_keys)} total)"
            )
        print(f"WildDet3D checkpoint matched all {len(state_dict)} parameter and buffer keys")
        return report

    huggingface_hub.hf_hub_download = local_lingbot_config
    WildDet3D.load_state_dict = checked_load
    try:
        model = build_model(
            checkpoint=str(args.checkpoint_path.resolve()),
            score_threshold=args.score_threshold,
            score_3d_threshold=args.score_3d_threshold,
            skip_pretrained=True,
            device=args.device,
        )
    finally:
        huggingface_hub.hf_hub_download = original_download
        WildDet3D.load_state_dict = original_load

    data = preprocess(pixels, intrinsics)
    device = torch.device(args.device)
    with torch.inference_mode():
        result = model(
            images=data["images"].to(device),
            intrinsics=data["intrinsics"].to(device)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_texts=classes,
        )
    names = ("boxes", "boxes3d", "scores", "scores_2d", "scores_3d", "class_ids", "depth_maps")
    output = {}
    for name, values in zip(names, result, strict=True):
        if values is None:
            continue
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            raise ValueError(f"unexpected WildDet3D {name} batch shape")
        value = values[0]
        if value is None:
            continue
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"WildDet3D {name} contains nonfinite values")
        output[name] = array
    count = len(output["scores"])
    if output["boxes"].shape != (count, 4) or output["boxes3d"].shape != (count, 10):
        raise ValueError("WildDet3D returned inconsistent 2D/3D box shapes")
    if any(output[name].shape != (count,) for name in ("scores_2d", "scores_3d", "class_ids")):
        raise ValueError("WildDet3D returned inconsistent detection score or class shapes")
    if count and (output["class_ids"].min() < 0 or output["class_ids"].max() >= len(classes)):
        raise ValueError("WildDet3D class ID is outside the requested class vocabulary")
    output["class_names"] = np.asarray(classes)
    output["image_hw"] = np.asarray(pixels.shape[:2], dtype=np.int32)
    path = args.output_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    print(json.dumps({"output_path": str(path), "detections": int(len(output["scores"])), "keys": list(output)}))
    return path


if __name__ == "__main__":
    run(_parser().parse_args())
