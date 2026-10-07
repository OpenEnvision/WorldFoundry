"""Generate and decode a checkpoint-backed Kandinsky-6 audio-video clip."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def main() -> None:
    import av
    import numpy as np
    import torch

    from worldfoundry.pipelines.kandinsky6 import Kandinsky6Pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="kandinsky6-pro-distill")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--offload-mode", default="component")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default=(
            "A small orange cat walks slowly along a sunlit garden path, "
            "turning its head toward the camera. Green leaves sway gently "
            "in the breeze. Birds chirp clearly in the background. "
            "Realistic footage, one continuous shot."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    print(f"Loading {args.model_id} on {torch.cuda.get_device_name(0)}", flush=True)
    pipeline = Kandinsky6Pipeline.from_pretrained(
        args.checkpoint_dir,
        model_id=args.model_id,
        device="cuda:0",
        offload_mode=args.offload_mode,
        attention_backend=args.attention_backend,
    )
    load_seconds = time.monotonic() - started
    print(f"Weights loaded in {load_seconds:.1f}s; generating clip", flush=True)
    result = pipeline(
        prompt=args.prompt,
        images=args.image,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed,
        output_path=args.output_dir / "output.mp4",
        return_dict=True,
    )
    artifact = Path(result["artifact_path"])
    with av.open(str(artifact)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        video_stream = container.streams.video[0]
        assert (video_stream.width, video_stream.height) == (args.width, args.height)
        assert video_stream.average_rate == 24
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(frames) == args.num_frames
    with av.open(str(artifact)) as container:
        audio_stream = container.streams.audio[0]
        assert audio_stream.sample_rate == 44100
        audio = np.concatenate([frame.to_ndarray().reshape(-1) for frame in container.decode(audio=0)])
    assert audio.size > 0 and np.isfinite(audio).all()
    report = {
        "status": "success",
        "model_id": pipeline.model_id,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "artifact_path": str(artifact.resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "environment": {
            "python": sys.version.split()[0],
            "cuda": torch.version.cuda,
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "diffusers",
                    "transformers",
                    "safetensors",
                    "accelerate",
                    "av",
                    "librosa",
                    "pydantic",
                    "numpy",
                )
            },
        },
        "offload_mode": args.offload_mode,
        "attention_backend": args.attention_backend,
        "prompt": args.prompt,
        "image": str(args.image.resolve()) if args.image else None,
        "seed": args.seed,
        "load_seconds": load_seconds,
        "generation_seconds": result["elapsed_seconds"],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "artifact_bytes": artifact.stat().st_size,
        "video": {
            "width": args.width,
            "height": args.height,
            "frames": len(frames),
            "fps": 24,
            "duration_seconds": len(frames) / 24,
            "pixel_std": float(np.std(frames[0])),
            "first_last_frame_mae": float(np.mean(np.abs(frames[0].astype(float) - frames[-1]))),
        },
        "audio": {
            "sample_rate": 44100,
            "decoded_samples": int(audio.size),
            "rms": float(np.sqrt(np.mean(audio.astype(float) ** 2))),
            "peak": float(np.max(np.abs(audio))),
        },
    }
    report_path = args.output_dir / "cuda-validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    from PIL import Image

    Image.fromarray(frames[0]).save(args.output_dir / "first-frame.png")
    Image.fromarray(frames[len(frames) // 2]).save(args.output_dir / "middle-frame.png")
    Image.fromarray(frames[-1]).save(args.output_dir / "last-frame.png")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
