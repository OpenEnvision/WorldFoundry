from __future__ import annotations

import importlib.util
from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path, resolve_data_path

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "play.py"
CONFIG_DIR = resolve_data_path("models", "runtime", "configs", "diamond", "config")
DEFAULT_PRETRAINED_DIR = next(
    (
        path
        for path in (checkpoint_root_path("eloialonso--diamond"), checkpoint_root_path("diamond"))
        if path.is_dir()
    ),
    checkpoint_root_path("eloialonso--diamond"),
)
# The model-specific requirements hook below is authoritative. Keeping a static
# blocker here would reject execution even after every local asset is present.
BLOCKED_REASON = ""


def _is_csgo(options):
    return str(options.get("variant") or options.get("pretrained_game") or options.get("game") or "").lower() in {"csgo", "diamond-csgo"}


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    options = dict(options or {})
    missing = []
    if _is_csgo(options):
        root = Path(str(options.get("pretrained_dir") or DEFAULT_PRETRAINED_DIR)).expanduser()
        spawn = Path(str(options["spawn_dir"])).expanduser() if options.get("spawn_dir") else root / "csgo/spawn" / str(options.get("spawn_id", 0))
        checkpoint = options.get("checkpoint") or options.get("checkpoint_path") or options.get("model_path")
        required = [
            RUNTIME_DIR / "csgo_inference.py", root / "csgo/config/agent/csgo.yaml",
            Path(str(checkpoint)).expanduser() if checkpoint else root / "csgo/model/csgo.pt",
            *(spawn / name for name in ("low_res.npy", "full_res.npy", "act.npy")),
        ]
        if options.get("actions_path"):
            required.append(Path(str(options["actions_path"])).expanduser())
        return [{"kind": "asset", "path": str(p), "reason": "required local DIAMOND CS:GO asset is missing"} for p in required if not p.is_file()]
    config_dir = Path(str(options.get("config_dir") or CONFIG_DIR)).expanduser()
    checkpoint = options.get("checkpoint") or options.get("checkpoint_path") or options.get("model_path")
    pretrained = bool(options.get("pretrained", False))
    pretrained_game = str(options.get("pretrained_game") or options.get("game") or "Pong")
    pretrained_dir = Path(str(options.get("pretrained_dir") or DEFAULT_PRETRAINED_DIR)).expanduser()
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "DIAMOND play.py is missing"})
    if not config_dir.is_dir() or not (config_dir / "trainer.yaml").is_file():
        missing.append({"kind": "asset", "path": str(config_dir), "reason": "DIAMOND runtime config directory is missing trainer.yaml"})
    for module_name in (
        "hydra",
        "omegaconf",
        "pygame",
        "gymnasium",
        "ale_py",
        "cv2",
        "PIL",
        "torcheval",
        "tqdm",
    ):
        if importlib.util.find_spec(module_name) is None:
            missing.append({"kind": "python_module", "path": module_name, "reason": "required DIAMOND runtime package is not importable"})
    if not pretrained:
        if not checkpoint:
            missing.append(
                {
                    "kind": "checkpoint",
                    "path": "checkpoint",
                    "reason": "DIAMOND requires checkpoint/checkpoint_path/model_path unless pretrained=true",
                }
            )
        elif not Path(str(checkpoint)).expanduser().is_file():
            missing.append({"kind": "checkpoint", "path": str(checkpoint), "reason": "DIAMOND checkpoint file does not exist"})
    else:
        for relative in (
            f"atari_100k/models/{pretrained_game}.pt",
            "atari_100k/config/agent/default.yaml",
            "atari_100k/config/env/atari.yaml",
        ):
            path = pretrained_dir / relative
            if not path.is_file():
                missing.append({"kind": "checkpoint", "path": str(path), "reason": "DIAMOND local pretrained asset is missing"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    if _is_csgo(options):
        command = [
            context["python"], str(RUNTIME_DIR / "csgo_inference.py"),
            "--pretrained-dir", str(options.get("pretrained_dir") or DEFAULT_PRETRAINED_DIR),
            "--spawn-id", str(options.get("spawn_id", 0)),
            "--action", str(options.get("action", "forward")),
            "--sampler", str(options.get("sampler", "higher_quality")),
            "--headless-steps", str(options.get("headless_steps", options.get("frames", 24))),
            "--fps", str(options.get("fps", 16)), "--seed", str(options.get("seed", 42)),
            "--device", str(options.get("device", "cuda")),
            "--output-path", context["output_path"],
        ]
        checkpoint = options.get("checkpoint") or options.get("checkpoint_path") or options.get("model_path")
        if checkpoint:
            command.extend(("--checkpoint", str(checkpoint)))
        for name in ("spawn_dir", "actions_path"):
            if options.get(name):
                command.extend(("--" + name.replace("_", "-"), str(options[name])))
        return command
    command = [
        context["python"],
        context["entrypoint"],
        "--config-dir",
        str(options.get("config_dir") or CONFIG_DIR),
        "--fps",
        str(options.get("fps", 15)),
        "--size",
        str(options.get("size", 640)),
        "--num-steps-initial-collect",
        str(options.get("num_steps_initial_collect", 1000)),
        "--no-header",
    ]
    checkpoint = options.get("checkpoint") or options.get("checkpoint_path") or options.get("model_path")
    if bool(options.get("pretrained", False)):
        command.append("--pretrained")
        game = options.get("pretrained_game") or options.get("game") or "Pong"
        command.extend(["--pretrained-game", str(game)])
        pretrained_dir = Path(str(options.get("pretrained_dir") or DEFAULT_PRETRAINED_DIR)).expanduser()
        if pretrained_dir.is_dir() or options.get("pretrained_dir"):
            command.extend(["--pretrained-dir", str(pretrained_dir)])
    elif checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    if bool(options.get("record", False)):
        command.append("--record")
    if bool(options.get("store_denoising_trajectory", False)):
        command.append("--store-denoising-trajectory")
    if bool(options.get("store_original_obs", False)):
        command.append("--store-original-obs")
    headless_steps = int(options.get("headless_steps", options.get("frames", 0)) or 0)
    if headless_steps:
        dataset_dir = options.get("dataset_dir") or str(Path(context["output_dir"]) / "initialization_dataset")
        command.extend(
            [
                "--headless-steps",
                str(headless_steps),
                "--output-path",
                context["output_path"],
                "--dataset-dir",
                str(dataset_dir),
            ]
        )
    return command


__all__ = [
    "BLOCKED_REASON",
    "CONFIG_DIR",
    "DEFAULT_PRETRAINED_DIR",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
]
