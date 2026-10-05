"""Finite CS:GO rollouts from a released DIAMOND spawn, without a game client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from worldfoundry.core.media.codecs.video import save_video_h264
from worldfoundry.synthesis.visual_generation.diamond.models.diffusion.csgo import (
    CsgoDenoiser, CsgoDenoiserConfig, CsgoSampler, CsgoSamplerConfig,
)
from worldfoundry.synthesis.visual_generation.diamond.models.diffusion.inner_model import InnerModelConfig


def action_sequence(name: str, path: str | None, steps: int) -> torch.Tensor:
    if path:
        source = Path(path)
        value = np.load(source, allow_pickle=False) if source.suffix == ".npy" else json.loads(source.read_text())
        actions = torch.as_tensor(value)
        if actions.shape != (steps, 51):
            raise ValueError(f"CS:GO actions must have shape ({steps}, 51), got {tuple(actions.shape)}")
    else:
        actions = torch.zeros(steps, 51, dtype=torch.long)
        # Official action order: 11 keys, two clicks, 23 mouse-X, 15 mouse-Y.
        actions[:, 13 + 11] = 1
        actions[:, 36 + 7] = 1
        keys = {"forward": 0, "left": 1, "backward": 2, "right": 3, "jump": 4}
        if name in keys:
            actions[:, keys[name]] = 1
        elif name in {"turn-left", "turn-right"}:
            actions[:, 13 + 11] = 0
            actions[:, 13 + (5 if name == "turn-left" else 17)] = 1
        elif name != "idle":
            raise ValueError(f"Unknown CS:GO action: {name}")
    if not bool(((actions == 0) | (actions == 1)).all()):
        raise ValueError("CS:GO actions must contain binary values")
    if not bool((actions[:, 13:36].sum(1) == 1).all() and (actions[:, 36:51].sum(1) == 1).all()):
        raise ValueError("CS:GO actions require one mouse-X and one mouse-Y bin per step")
    return actions.long()


def load_models(root: Path, checkpoint: Path, device: torch.device):
    config = yaml.safe_load((root / "csgo/config/agent/csgo.yaml").read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    models, covered = {}, set()
    for name in ("denoiser", "upsampler"):
        values = {k: v for k, v in config[name].items() if k != "_target_"}
        inner = {k: v for k, v in values.pop("inner_model").items() if k != "_target_"}
        inner["num_actions"] = 51
        model = CsgoDenoiser(CsgoDenoiserConfig(inner_model=InnerModelConfig(**inner), **values))
        prefix = name + "."
        subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        covered.update(key for key in state if key.startswith(prefix))
        model.load_state_dict(subset, strict=True)
        models[name] = model.eval().to(device)
    if covered != set(state):
        raise ValueError(f"Unconsumed CS:GO checkpoint keys: {sorted(set(state) - covered)[:10]}")
    return models


@torch.inference_mode()
def run(args):
    if args.headless_steps <= 0 or args.fps <= 0:
        raise ValueError("CS:GO headless_steps and fps must be positive")
    root = Path(args.pretrained_dir).expanduser().resolve()
    spawn = Path(args.spawn_dir).expanduser().resolve() if args.spawn_dir else root / "csgo/spawn" / str(args.spawn_id)
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else root / "csgo/model/csgo.pt"
    device = torch.device(args.device)
    actions = action_sequence(args.action, args.actions_path, args.headless_steps).to(device)
    models = load_models(root, checkpoint, device)
    fast = args.sampler == "fast"
    samplers = (
        CsgoSampler(models["denoiser"], CsgoSamplerConfig(num_steps_denoising=1 if fast else 3, s_cond=0.005)),
        CsgoSampler(models["upsampler"], CsgoSamplerConfig(
            num_steps_denoising=1 if fast else 10, sigma_min=1, order=2 if fast else 1,
            s_churn=10, s_tmin=1, s_tmax=5, s_noise=0.9,
        )),
    )

    def pixels(name, shape):
        value = np.load(spawn / name, allow_pickle=False)
        if value.ndim != 4 or tuple(value.shape[1:]) != shape or value.shape[0] < 4 or value.dtype != np.uint8:
            raise ValueError(f"Invalid released CS:GO spawn {name}: {value.shape}/{value.dtype}")
        return torch.tensor(value, device=device).float().div(255).mul(2).sub(1).unsqueeze(0)

    obs = pixels("low_res.npy", (3, 30, 56))
    full = pixels("full_res.npy", (3, 150, 280))
    history = torch.as_tensor(np.load(spawn / "act.npy", allow_pickle=False), device=device).unsqueeze(0)
    if history.shape != (1, obs.shape[1], 51) or full.shape[1] != obs.shape[1] or not bool(((history == 0) | (history == 1)).all()):
        raise ValueError("CS:GO spawn history must contain aligned frames and 51D binary actions")
    history = history.long()
    torch.manual_seed(args.seed)
    frames = [full[0, -1].cpu()]
    for action in actions:
        history[:, -1] = action
        low, _ = samplers[0].sample(obs[:, -4:], history[:, -4:])
        up = F.interpolate(low, scale_factor=5, mode="bicubic").unsqueeze(1)
        high, _ = samplers[1].sample(torch.cat((full[:, -1:], up), dim=1), None)
        if not bool(torch.isfinite(low).all() and torch.isfinite(high).all()):
            raise FloatingPointError("Nonfinite DIAMOND CS:GO frame")
        obs = obs.roll(-1, dims=1)
        obs[:, -1] = low
        full = full.roll(-1, dims=1)
        full[:, -1] = high
        history = history.roll(-1, dims=1)
        frames.append(high[0].cpu())
    video = torch.stack(frames)
    output = Path(args.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_video_h264(((video.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy(), output, fps=args.fps)
    output.with_suffix(".inference.json").write_text(json.dumps({
        "variant": "csgo", "checkpoint": str(checkpoint), "spawn": str(spawn),
        "sampler": args.sampler, "seed": args.seed, "frames": len(frames),
        "shape": list(video.shape), "actions": actions.cpu().tolist(),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--spawn-dir")
    parser.add_argument("--spawn-id", type=int, default=0)
    parser.add_argument("--action", default="forward")
    parser.add_argument("--actions-path")
    parser.add_argument("--sampler", choices=("fast", "higher_quality"), default="higher_quality")
    parser.add_argument("--headless-steps", type=int, default=24)
    parser.add_argument("--fps", type=float, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-path", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
