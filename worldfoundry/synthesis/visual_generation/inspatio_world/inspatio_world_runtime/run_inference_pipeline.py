"""Python orchestration for the caption, geometry, and video inference stages."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.reproducibility import (
    SEED_ENV,
    reproducibility_env,
)


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    config_root = Path(os.environ.get(
        "WORLDFOUNDRY_INSPATIO_WORLD_CONFIG_ROOT",
        str(ROOT.parents[3] / "data/models/runtime/configs/inspatio_world"),
    ))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--traj_txt_path", default=str(config_root / "traj/x_y_circle_cycle.txt"))
    p.add_argument("--config_path", default=str(config_root / "inference_1.3b.yaml"))
    for name, default in {
        "checkpoint_path": "InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors",
        "florence_model_path": "Florence-2-large", "da3_model_path": "DA3",
        "wan_model_path": "Wan2.1-T2V-1.3B", "tae_checkpoint_path": "taehv/taew2_1.pth",
    }.items():
        p.add_argument(f"--{name}", default=str(ROOT / "checkpoints" / default))
    p.add_argument("--output_folder")
    for stage in (1, 2, 3):
        p.add_argument(f"--step{stage}_gpus", default="0")
        p.add_argument(f"--skip_step{stage}", action="store_true")
    p.add_argument("--step3_nproc", type=int, default=1)
    p.add_argument("--master_port", type=int, default=29513)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--freeze_repeat", type=int, default=0)
    p.add_argument("--freeze_frame", type=int)
    for flag in ("relative_to_source", "rotation_only", "disable_adaptive_frame", "use_tae", "compile_dit"):
        p.add_argument(f"--{flag}", action="store_true")
    return p


def _override_config(value, overrides):
    if isinstance(value, dict):
        return {key: overrides[key] if key in overrides else _override_config(item, overrides)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_override_config(item, overrides) for item in value]
    return value


def run_pipeline(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    trajectory = Path(args.traj_txt_path).expanduser().resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory}")
    if args.step3_nproc < 1:
        raise ValueError("step3_nproc must be positive")
    json_path = input_dir / "new.json"
    output = Path(args.output_folder or ROOT / "output" / input_dir.name / trajectory.stem).resolve()
    env = reproducibility_env(os.environ, args.seed, args.deterministic)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
    cuda_compat = Path(env.get("CUDA_COMPAT_DIR", "/usr/local/cuda-12.1/compat"))
    if cuda_compat.is_dir() and str(cuda_compat) not in env.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(cuda_compat), env.get("LD_LIBRARY_PATH"))))
    if "CUDA_HOME" not in env and Path("/usr/local/cuda-12.1").is_dir():
        env["CUDA_HOME"] = "/usr/local/cuda-12.1"

    def run(command, gpu=None):
        child_env = dict(env)
        if gpu is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        subprocess.run([sys.executable, *map(str, command)], check=True, cwd=ROOT, env=child_env)

    if not args.skip_step1:
        gpus = [gpu.strip() for gpu in args.step1_gpus.split(",") if gpu.strip()]
        if not gpus:
            raise ValueError("step1_gpus must contain at least one GPU")
        caption = [ROOT / "scripts/gen_json.py", "--root_dir", input_dir,
                   "--model_path", args.florence_model_path]
        if len(gpus) == 1:
            run(caption, gpus[0])
        else:
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = [pool.submit(run, [*caption, "--worker_id", i, "--num_workers", len(gpus),
                           "--output_json", input_dir / f"new_partial_{i}.json"], gpu)
                           for i, gpu in enumerate(gpus)]
                for future in futures:
                    future.result()
            run([ROOT / "scripts/merge_partial_jsons.py", "--input_dir", input_dir, "--output_json", json_path])

    if not args.skip_step2:
        da3_config = json.dumps({"model_path": args.da3_model_path, "fix_resize": True,
                                "fix_resize_height": 480, "fix_resize_width": 832,
                                "num_frames": 1000, "save_point_cloud": True})
        run([ROOT / "scripts/run_da3_parallel.py", "--json_path", json_path,
             "--gpu_list", args.step2_gpus, "--da3_cli", ROOT / "depth/depth_predict_da3_cli.py",
             "--da3_config", da3_config, "--convert_script", ROOT / "scripts/convert_da3_to_pi3.py"])

    # A new trajectory requires a new rendering, including when depth is cached.
    render = [ROOT / "scripts/run_render_parallel.py", "--json_path", json_path,
              "--gpu_list", args.step2_gpus, "--render_script", ROOT / "scripts/render_point_cloud.py",
              "--traj_txt_path", trajectory, "--width", 832, "--height", 480]
    for flag in ("relative_to_source", "rotation_only"):
        if getattr(args, flag):
            render.append(f"--{flag}")
    if args.freeze_repeat > 0:
        render.extend(["--freeze_repeat", args.freeze_repeat])
    if args.freeze_frame is not None:
        render.extend(["--freeze_frame", args.freeze_frame])
    run(render)

    if args.skip_step3:
        return
    wan = Path(args.wan_model_path).expanduser().resolve()
    t5_st = wan / "models_t5_umt5-xxl-enc-bf16.safetensors"
    if not t5_st.is_file():
        t5_pth = wan / "models_t5_umt5-xxl-enc-bf16.pth"
        if not t5_pth.is_file():
            raise FileNotFoundError(f"Wan T5 encoder checkpoint not found: {t5_pth}")
        run([ROOT / "utils/convert_pth_to_safetensors.py", "--input", t5_pth, "--output", t5_st])

    import yaml

    overrides = {"traj_txt_path": str(trajectory), "relative_to_source": args.relative_to_source,
                 "rotation_only": args.rotation_only, "adaptive_frame": not args.disable_adaptive_frame,
                 "freeze_repeat": args.freeze_repeat}
    if args.freeze_frame is not None:
        overrides["freeze_frame"] = args.freeze_frame
    config = _override_config(yaml.safe_load(Path(args.config_path).read_text()), overrides)
    with tempfile.TemporaryDirectory(prefix="inspatio-config-") as temp:
        config_path = Path(temp) / "inference.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        infer = ["-m", "torch.distributed.run", f"--nproc_per_node={args.step3_nproc}",
                 "--master_port", args.master_port, ROOT / "inference_causal.py",
                 "--config_path", config_path, "--json_path", json_path,
                 "--checkpoint_path", args.checkpoint_path, "--output_folder", output]
        if SEED_ENV in env:
            infer.extend(["--seed", env[SEED_ENV]])
        if args.deterministic:
            infer.append("--deterministic")
        if args.use_tae:
            infer.append("--use_tae")
            infer.extend(["--tae_checkpoint_path", args.tae_checkpoint_path])
        if args.compile_dit:
            infer.append("--compile_dit")
        run(infer, args.step3_gpus)


if __name__ == "__main__":
    run_pipeline(parser().parse_args())
