"""Run one offline MIRA rollout.

This is an inference-only adapter around the upstream MIRA checkpoint loader,
Rocket Science data loader, and autoregressive rollout. The original model
implementation is preserved under ``src/mira``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

RUNTIME_DIR = Path(__file__).resolve().parent
RUNTIME_SRC = RUNTIME_DIR / "src"


def _noise_level(value: str) -> float | None:
    return None if value.strip().lower() == "none" else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="MIRA checkpoint file or run directory.")
    parser.add_argument("--dataset", required=True, help="Rocket Science split directory or index.json.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument(
        "--actions-file",
        type=Path,
        default=None,
        help="Optional JSON list of {player, keys, start, frames} action segments.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (cuda, cuda:N, cpu, or auto).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--n-context-frames", type=int, default=38)
    parser.add_argument("--num-unrolled-frames", type=int, default=20)
    parser.add_argument("--n-diffusion-steps", type=int, default=10)
    parser.add_argument("--schedule-type", choices=("linear", "linear_quadratic"), default="linear")
    parser.add_argument("--noise-level", type=_noise_level, default=0.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-overlay-actions", action="store_true")
    parser.add_argument("--generated-only", action="store_true")
    return parser.parse_args()


def _load_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    direct = path / "checkpoint.pth"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(
        path.glob("checkpoint-*/checkpoint.pth"),
        key=lambda item: int(item.parent.name.rsplit("-", 1)[-1])
        if item.parent.name.rsplit("-", 1)[-1].isdigit()
        else -1,
    )
    if candidates:
        return candidates[-1].resolve()
    raise FileNotFoundError(f"No MIRA checkpoint.pth found under {path}")


def _normalize_actions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    rows = [value] if isinstance(value, Mapping) else value
    if not isinstance(rows, list):
        raise TypeError("MIRA actions JSON must be a list of objects")

    valid_keys = {"W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"}
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Each MIRA action segment must be a JSON object")

        player: int | str = row.get("player", "all")
        if isinstance(player, str):
            player_text = player.strip().lower()
            if player_text in {"all", "*"}:
                player = "all"
            else:
                if player_text.startswith("player"):
                    player_text = player_text.removeprefix("player")
                elif player_text.startswith("p"):
                    player_text = player_text[1:]
                if not player_text.isdigit():
                    raise ValueError(f"Invalid MIRA player {player!r}")
                player = int(player_text)
        if player != "all" and (isinstance(player, bool) or not isinstance(player, int) or not 0 <= player < 4):
            raise ValueError(f"MIRA player must be 'all' or an integer in [0, 3], got {player!r}")

        keys_value = row.get("keys", [])
        keys = [item.strip() for item in keys_value.replace(",", "+").split("+")] if isinstance(
            keys_value, str
        ) else list(keys_value)
        keys = list(dict.fromkeys(str(key).strip() for key in keys if str(key).strip()))
        unknown = [key for key in keys if key not in valid_keys]
        if unknown:
            raise ValueError(f"Unsupported MIRA action keys: {unknown}")

        start = row.get("start", cursor)
        frames = row.get("frames", 1)
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError(f"MIRA action start must be a non-negative integer, got {start!r}")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            raise ValueError(f"MIRA action frames must be a positive integer, got {frames!r}")
        segment = {"player": player, "keys": keys, "start": start, "frames": frames}
        normalized.append(segment)
        cursor = max(cursor, start + frames)
    return normalized


def _apply_action_timeline(
    batch: Any,
    timeline: list[dict[str, Any]],
    *,
    context_frames: int,
    video_fps: int,
) -> None:
    if not timeline:
        return
    actions = batch.actions
    valid_keys = list(actions.config.valid_keys)
    key_indices = {key: index for index, key in enumerate(valid_keys)}
    if video_fps < 1 or actions.config.target_fps % video_fps != 0:
        raise ValueError(
            "MIRA action target_fps must be a positive integer multiple of the video fps "
            f"(got action_fps={actions.config.target_fps}, video_fps={video_fps})"
        )
    actions_per_frame = actions.config.target_fps // video_fps
    first_step = max(0, context_frames - 1) * actions_per_frame

    for segment in timeline:
        player = segment["player"]
        rows = range(actions.batch_size) if player == "all" else (int(player),)
        start = first_step + int(segment["start"]) * actions_per_frame
        end = min(actions.n_steps, start + int(segment["frames"]) * actions_per_frame)
        if start >= actions.n_steps:
            continue
        keys = list(segment["keys"])
        unknown = [key for key in keys if key not in key_indices]
        if unknown:
            raise ValueError(f"Action keys do not match checkpoint vocabulary: {unknown}")
        for row in rows:
            if not 0 <= row < actions.batch_size:
                raise ValueError(f"Action timeline targets player {row}, but batch has {actions.batch_size} rows")
            actions.key_presses[row, start:end].zero_()
            for key in keys:
                actions.key_presses[row, start:end, key_indices[key]] = 1


def _next_clip(loader: Any, clip_index: int) -> tuple[Any, Any]:
    if clip_index < 0:
        raise ValueError("clip_index must be non-negative")
    iterator = iter(loader)
    for index in range(clip_index + 1):
        try:
            value = next(iterator)
        except StopIteration as exc:
            raise IndexError(f"Rocket Science split contains fewer than {clip_index + 1} clips") from exc
        if index == clip_index:
            return value
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    if str(RUNTIME_SRC) not in sys.path:
        sys.path.insert(0, str(RUNTIME_SRC))

    import numpy as np
    import torch
    from mira.data.inference_loader import create_loader
    from mira.inference.loading import load_world_model
    from mira.inference_support.visualization import videos_to_grid, write_video_ffmpeg
    from mira.world_model.config import WorldModelInferenceConfig

    device_text = args.device
    if device_text == "auto":
        device_text = "cuda" if torch.cuda.is_available() else "cpu"
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MIRA requires an NVIDIA GPU, but CUDA is unavailable")
    device = torch.device(device_text)

    timeline = _normalize_actions(_load_json(args.actions_file))
    n_context_frames = args.n_context_frames
    num_unrolled_frames = args.num_unrolled_frames

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    model, _run_config = load_world_model(checkpoint, device=device)
    model = model.eval()
    model.set_inference_context(n_context_frames)
    if args.compile:
        model.world_model.compile()

    n_players = int(getattr(model, "n_players", 1))
    clip_len = n_context_frames + num_unrolled_frames * int(model.temporal_downsampling)
    loader = create_loader(
        index_path=Path(args.dataset).expanduser(),
        clip_len=clip_len,
        target_fps=int(model.config.video.fps),
        action_fps=int(model.config.actions.target_fps),
        n_players=n_players,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=False,
        exclude_replays=True,
        valid_keys=list(model.config.actions.valid_keys),
        pin_memory=False,
    )
    batch, metadata = _next_clip(loader, args.clip_index)
    _apply_action_timeline(
        batch,
        timeline,
        context_frames=n_context_frames,
        video_fps=int(model.config.video.fps),
    )

    inference_config = WorldModelInferenceConfig(
        n_diffusion_steps=args.n_diffusion_steps,
        schedule_type=args.schedule_type,
        noise_level=args.noise_level,
    )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        outputs = model.inference(batch, config=inference_config, progress_bar=True)

    if args.no_overlay_actions:
        video = outputs.output_video
    else:
        video = model.visualize(outputs)["pred_video"]
    if args.generated_only:
        video = video[:, n_context_frames:]
    grid = videos_to_grid(video)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video_ffmpeg(args.output, grid.cpu(), fps=float(model.config.video.fps))
    result_path = args.output.with_suffix(args.output.suffix + ".result.json")
    result = {
        "status": "succeeded",
        "model_id": "mira",
        "checkpoint": str(checkpoint),
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "clip_index": args.clip_index,
        "n_players": n_players,
        "n_context_frames": n_context_frames,
        "num_unrolled_frames": num_unrolled_frames,
        "fps": float(model.config.video.fps),
        "output_frames": int(grid.shape[0]),
        "action_timeline": timeline,
        "source_clips": [
            {
                "match_id": item.match_id,
                "perspective": item.perspective,
                "clip_id": item.clip_id,
                "chunk_idx": item.chunk_idx,
            }
            for item in metadata
        ],
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
