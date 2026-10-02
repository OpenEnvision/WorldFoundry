from __future__ import annotations

from pathlib import Path

from worldfoundry.synthesis.visual_generation.vid2world.config_paths import (
    runtime_config_root,
)


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "main" / "inference.py"
CONFIG_ROOT = runtime_config_root()
DEFAULT_CONFIG = "game/config_csgo_test.yaml"
BLOCKED_REASON = (
    "Vid2World source is vendored in-tree; inference requires a test config with "
    "checkpoint/data paths, official config assets, and conditioning/action inputs."
)


def runtime_root() -> Path:
    return RUNTIME_DIR


def pythonpath_entries(*, runtime_root, options, profile) -> list[str]:
    del runtime_root, options, profile
    return []


def _option(options: dict, *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _config_path(options: dict) -> Path:
    value = _option(options, "config", "config_path", "base_config", "preset", default=DEFAULT_CONFIG) or DEFAULT_CONFIG
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    return (CONFIG_ROOT / candidate).expanduser().resolve()


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    config_path = _config_path(dict(options or {}))
    missing = []
    if not config_path.exists():
        missing.append({"kind": "asset", "path": str(config_path), "reason": "Vid2World test config does not exist"})
    else:
        config_text = config_path.read_text(encoding="utf-8")
        if "|<your_pretrained_checkpoint>|" in config_text:
            missing.append(
                {
                    "kind": "checkpoint",
                    "path": str(config_path),
                    "reason": "Vid2World config still contains the official checkpoint placeholder",
                }
            )
        if "|<your_data_dir>|" in config_text:
            missing.append(
                {
                    "kind": "asset",
                    "path": str(config_path),
                    "reason": "Vid2World config still contains the official data directory placeholder",
                }
            )
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "Vid2World main/inference.py is missing"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    devices = int(options.get("devices", options.get("num_gpus", 1)))
    master_port = str(options.get("master_port", 12869))
    return [
        context["python"],
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(devices),
        "--nnodes",
        "1",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        master_port,
        context["entrypoint"],
        "--base",
        str(_config_path(options)),
        "--val",
        "--name",
        str(options.get("name", "worldfoundry_infer")),
        "--logdir",
        context["output_dir"],
        "--devices",
        str(devices),
        "lightning.trainer.num_nodes=1",
    ]
