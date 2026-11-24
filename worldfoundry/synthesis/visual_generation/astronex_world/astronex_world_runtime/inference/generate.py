# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1, inference/generate.py.
"""Generate video with Astronex-World, causal or bidirectional.

Both modes run the same 5.35B backbone from one released weights directory:

* ``causal`` attends backwards over a sliding 20-frame window with a 4-frame sink
  and samples in 8 steps, block by block; this is the distilled student.
* ``bidirectional`` attends over the whole clip and samples in 50 steps; this is
  the teacher the student was distilled from.

This front end resolves the mode's sampling config against the weights directory
and starts :mod:`.sample` on the selected GPU.
"""

import argparse
import os
import subprocess
import sys

SAMPLER_MODULE = "worldfoundry.synthesis.visual_generation.astronex_world.astronex_world_runtime.inference.sample"


def resolve_config(src: str, dst: str, weights_dir: str) -> str:
    """Write ``src`` to ``dst`` with ``model_kwargs.model_root`` set to the weights directory."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(src)
    cfg.model_kwargs.model_root = weights_dir
    cfg.model_kwargs.model_name = "transformer"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    OmegaConf.save(cfg, dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description="Astronex-World generation (causal or bidirectional)")
    ap.add_argument("--mode", choices=("causal", "bidirectional"), default="causal")
    ap.add_argument("--prompt", required=True, help="prompt text, or a path to a .txt of prompts")
    ap.add_argument("--image", default=None, help="reference image; omit for text-to-video")
    ap.add_argument(
        "--trajectory",
        default=None,
        help="camera stream, e.g. 'w*23' forward, 'h*23' hold, 'a*12,d*11' left then right; "
        "defaults to holding still. Its length has to equal the frame count",
    )
    ap.add_argument(
        "--frames", type=int, default=None, help="latent frames to generate (default: 23 causal, 17 bidirectional)"
    )
    ap.add_argument("--out", default="outputs/astronex")
    ap.add_argument("--steps", type=int, default=None, help="sampler steps; 8 causal / 50 bidirectional by default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--config", required=True, help="sampling config of the mode")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--vram-limit-gb", type=float, default=None, help="cap this process's VRAM, e.g. 32")
    ap.add_argument("--weights", required=True, help="released weights directory, shared by both modes")
    ap.add_argument("--event-prompt", default=None, help="second caption, swapped in mid-rollout")
    ap.add_argument("--event-start-frame", type=int, default=None)
    args, passthrough = ap.parse_known_args()
    out_dir = os.path.abspath(args.out)
    config = os.path.abspath(args.config)
    image = os.path.abspath(args.image) if args.image else None
    prompt_path = os.path.abspath(args.prompt) if os.path.exists(args.prompt) else None
    weights = os.path.abspath(args.weights)

    config = resolve_config(config, os.path.join(out_dir, f"_astronex_{args.mode}.yaml"), weights)
    frames = args.frames if args.frames is not None else (23 if args.mode == "causal" else 17)

    # The causal pipeline generates whole blocks of 8, and i2v spends one frame on
    # the reference, so ``1 + frames`` has to divide by the block size.
    if args.mode == "causal":
        block = 8
        need = (block - (1 + frames) % block) % block if image else (block - frames % block) % block
        if need:
            print(
                f"[astronex] frames {frames} -> {frames + need}: the causal pipeline needs complete blocks of {block}"
                + (" including the i2v reference frame" if image else "")
            )
            frames += need

    trajectory = args.trajectory or f"h*{frames}"
    if args.event_start_frame and args.mode != "causal":
        raise SystemExit(
            "--event-start-frame is only supported by the causal pipeline, which conditions the rollout "
            "block by block; the bidirectional pipeline conditions the whole clip at once."
        )

    # The sampler reads prompts from a file; write a command-line prompt into this run's output directory.
    if prompt_path is None:
        os.makedirs(out_dir, exist_ok=True)
        prompt_path = os.path.join(out_dir, "_astronex_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(args.prompt.rstrip("\n") + "\n")

    cmd = [
        sys.executable,
        "-m",
        SAMPLER_MODULE,
        "--config_path",
        config,
        "--checkpoint_path",
        weights,
        "--data_path",
        prompt_path,
        "--output_folder",
        out_dir,
        "--num_output_frames",
        str(frames),
        "--trajectory",
        trajectory,
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
    ]
    if args.steps is not None:
        cmd += ["--sampling_steps", str(args.steps)]
    if image:
        cmd += ["--i2v", "--image_path", image]
    if args.event_prompt:
        cmd += ["--event_prompt", args.event_prompt]
    if args.event_start_frame is not None:
        cmd += ["--event_start_frame", str(args.event_start_frame)]
    cmd += passthrough

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.vram_limit_gb:
        env["ASTRONEX_VRAM_LIMIT_GB"] = str(args.vram_limit_gb)

    print(f"[astronex] {args.mode} | weights {weights}")
    print("[astronex] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=out_dir, env=env))


if __name__ == "__main__":
    main()
