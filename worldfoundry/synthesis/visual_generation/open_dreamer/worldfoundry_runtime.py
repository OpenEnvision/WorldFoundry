"""Runtime adapter binding WorldFoundry to the official Open Dreamer rollout script.

Open Dreamer is a JAX/Flax implementation of the Dreamer 4 world model: a causal
video tokenizer plus an action-conditioned latent dynamics model, trained on
Minecraft/VPT gameplay. WorldFoundry drives the official rollout entrypoint as a
subprocess rather than importing it, because Open Dreamer needs its own CUDA-12
JAX environment and because its source is published under an all-rights-reserved
notice that does not permit redistribution inside this Apache-2.0 repository.

Staging the upstream checkout is therefore the user's step. WorldFoundry looks
for it at ``$WORLDFOUNDRY_OPEN_DREAMER_SOURCE`` first, then under
``$WORLDFOUNDRY_MODEL_SOURCE_DIR``::

    git clone https://github.com/reactor-team/open-dreamer \\
        "$WORLDFOUNDRY_MODEL_SOURCE_DIR/open-dreamer-inference"
    cd "$WORLDFOUNDRY_MODEL_SOURCE_DIR/open-dreamer-inference" && uv sync

Until the checkout, its JAX environment, and a trained Orbax checkpoint are all
present, the pipeline reports a blocked plan with per-requirement diagnostics
instead of attempting to run.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, official_runtime_repo_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings, plan_payload

from .vpt_actions import (
    DEFAULT_CAMERA_STEP_DEGREES,
    build_action_dicts,
    count_action_entries,
    write_action_jsonl,
)

RUNTIME_DIR = Path(__file__).resolve().parent

OFFICIAL_REPO_URL = "https://github.com/next-state/open-dreamer"
OFFICIAL_INFERENCE_REPO_URL = "https://github.com/reactor-team/open-dreamer"

# Environment override for a checkout that lives outside WORLDFOUNDRY_MODEL_SOURCE_DIR.
SOURCE_ENV_VAR = "WORLDFOUNDRY_OPEN_DREAMER_SOURCE"

# Checkout directory names probed under WORLDFOUNDRY_MODEL_SOURCE_DIR, in order.
# The inference harness is the rollout route; the training repo is kept as a
# fallback for users who cloned that one and copied `inference.py` beside it.
SOURCE_DIR_NAMES: tuple[str, ...] = ("open-dreamer-inference", "open-dreamer")

ENTRYPOINT_RELATIVE = "inference.py"

# Upstream package that `inference.py` imports; its presence distinguishes a real
# checkout from an empty directory.
RUNTIME_PACKAGE_RELATIVE = "pipeline"

DEFAULT_CHECKPOINT_DIR = checkpoint_root_path("open-dreamer")

# The model was trained at this resolution; the entrypoint also accepts 360x640
# and zero-pads it. Surfaced here so the plan can report the expected input shape.
MODEL_HEIGHT = 368
MODEL_WIDTH = 640

DEFAULT_CONTEXT_FRAMES = 16
DEFAULT_HORIZON = 64
DEFAULT_NUM_STEPS = 4
DEFAULT_DECODE_CHUNK_SIZE = 16
DEFAULT_PARALLEL_STRATEGY = "data"

# The official README runs rollouts with XLA preallocation disabled so JAX does
# not claim the whole device up front. The runtime facade builds the subprocess
# environment before it calls this module, so the setting rides on the command.
DEFAULT_ENV_OVERRIDES: dict[str, str] = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}

# Human-readable summary of what a rollout needs. The runtime manifest gates on
# `missing_requirements` rather than on this string, so a fully staged install
# runs instead of returning a permanently blocked plan.
BLOCKED_REASON = (
    "Open Dreamer is published under an all-rights-reserved notice, so WorldFoundry binds a "
    "user-staged official checkout instead of vendoring the source. Execution requires the "
    f"checkout ({OFFICIAL_INFERENCE_REPO_URL}), its CUDA-12 JAX environment, a trained Orbax "
    "checkpoint, and a 368x640 (or 360x640) Minecraft/VPT context clip."
)


def runtime_root() -> Path:
    """Resolve the staged Open Dreamer checkout used as the subprocess working directory."""
    override = os.getenv(SOURCE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    for name in SOURCE_DIR_NAMES:
        candidate = official_runtime_repo_path(name, specific_env=SOURCE_ENV_VAR)
        if (candidate / ENTRYPOINT_RELATIVE).is_file():
            return candidate
    # Nothing staged yet: report the canonical location so diagnostics stay actionable.
    return official_runtime_repo_path(SOURCE_DIR_NAMES[0], specific_env=SOURCE_ENV_VAR)


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first option present under any of ``names``."""
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce config values that may arrive as strings from a serialized plan."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _checkpoint_path(options: Mapping[str, Any]) -> Path:
    """Resolve the Orbax checkpoint directory for a rollout."""
    value = _option(
        options,
        "checkpoint_path",
        "checkpoint_dir",
        "ckpt_path",
        "model_path",
        "pretrained_model_path",
        default=DEFAULT_CHECKPOINT_DIR,
    )
    return Path(str(value)).expanduser()


def _input_video(options: Mapping[str, Any]) -> Path | None:
    """Resolve the context clip the rollout is seeded from."""
    value = _option(options, "input_mp4", "video", "video_path", "input_video", "input_path")
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _actions_path(options: Mapping[str, Any]) -> Path | None:
    """Resolve a caller-supplied VPT action file, if any."""
    value = _option(options, "actions_path", "actions_file", "action_path")
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _python_executable(options: Mapping[str, Any], runtime_root_path: Path) -> str:
    """Pick the interpreter that owns the Open Dreamer JAX environment.

    ``uv sync`` creates ``.venv`` inside the checkout, so prefer that over the
    WorldFoundry interpreter, which is a torch environment without JAX.
    """
    explicit = _option(options, "python_executable", "python_bin", "python")
    if explicit:
        return str(explicit)
    venv_python = runtime_root_path / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def _uses_worldfoundry_interpreter(options: Mapping[str, Any], runtime_root_path: Path) -> bool:
    """Report whether the rollout would run inside WorldFoundry's own interpreter."""
    return _python_executable(options, runtime_root_path) == sys.executable


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report every staging gap that blocks an Open Dreamer rollout."""
    del profile
    options = dict(options or {})
    root = Path(str(runtime_root)).expanduser()
    missing: list[dict[str, str]] = []

    if not root.is_dir():
        missing.append(
            {
                "kind": "source_repo",
                "path": str(root),
                "reason": (
                    f"Open Dreamer checkout is not staged; clone {OFFICIAL_INFERENCE_REPO_URL} to this path "
                    f"or point {SOURCE_ENV_VAR} at an existing checkout"
                ),
            }
        )
    elif not (root / RUNTIME_PACKAGE_RELATIVE).is_dir():
        missing.append(
            {
                "kind": "source_repo",
                "path": str(root / RUNTIME_PACKAGE_RELATIVE),
                "reason": "staged directory does not contain the Open Dreamer `pipeline` package",
            }
        )

    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": f"Open Dreamer {ENTRYPOINT_RELATIVE} is missing from the staged checkout",
            }
        )

    checkpoint = _checkpoint_path(options)
    if not checkpoint.is_dir():
        missing.append(
            {
                "kind": "checkpoint",
                "path": str(checkpoint),
                "reason": (
                    "Open Dreamer Orbax checkpoint directory does not exist; train one with "
                    f"{OFFICIAL_REPO_URL} or set checkpoint_path"
                ),
            }
        )

    video = _input_video(options)
    if video is None:
        missing.append(
            {
                "kind": "option",
                "path": "input_mp4",
                "reason": (
                    f"Open Dreamer needs a {MODEL_HEIGHT}x{MODEL_WIDTH} (or 360x{MODEL_WIDTH}) Minecraft/VPT "
                    "context clip; set input_mp4"
                ),
            }
        )
    elif not video.is_file():
        missing.append({"kind": "asset", "path": str(video), "reason": "Open Dreamer context clip does not exist"})

    actions_path = _actions_path(options)
    if actions_path is not None and not actions_path.is_file():
        missing.append(
            {
                "kind": "asset",
                "path": str(actions_path),
                "reason": "Open Dreamer VPT action file does not exist",
            }
        )

    if _uses_worldfoundry_interpreter(options, root):
        # WorldFoundry's own environment is a torch environment; JAX lives in the
        # checkout's uv venv or a dedicated conda env.
        for module_name in ("jax", "flax", "orbax.checkpoint", "imageio"):
            if importlib.util.find_spec(module_name) is None:
                missing.append(
                    {
                        "kind": "python_module",
                        "path": module_name,
                        "reason": (
                            "Open Dreamer requires a CUDA-12 JAX environment; run `uv sync` in the checkout "
                            "or set python_executable to an interpreter that has it"
                        ),
                    }
                )
                break

    return missing


def _resolve_actions_path(
    settings: Mapping[str, Any],
    interactions: Any,
    *,
    context_frames: int,
    horizon: int,
    output_dir: Path,
) -> Path:
    """Return the VPT action file for this rollout, synthesizing one when needed."""
    supplied = _actions_path(settings)
    if supplied is not None:
        available = count_action_entries(supplied)
        required = context_frames + horizon
        if available < required:
            raise ValueError(
                f"Open Dreamer needs {required} actions for context_frames={context_frames} and "
                f"horizon={horizon}, but {supplied} holds {available}"
            )
        return supplied.resolve()

    camera_step = float(_option(settings, "camera_step_degrees", default=DEFAULT_CAMERA_STEP_DEGREES))
    actions = build_action_dicts(
        list(interactions or ()),
        context_frames=context_frames,
        horizon=horizon,
        camera_step_degrees=camera_step,
    )
    return write_action_jsonl(output_dir / "open_dreamer_actions.jsonl", actions)


def build_command(context) -> list[str]:
    """Build the official Open Dreamer rollout command for one WorldFoundry request."""
    plan = plan_payload(context)
    # Call-time kwargs win over load-time options so a single loaded pipeline can
    # serve several rollouts.
    settings = command_settings(context)

    root = Path(str(context["runtime_root"])).expanduser()
    output_path = Path(str(context["output_path"]))
    output_dir = Path(str(context.get("output_dir") or output_path.parent))
    output_dir.mkdir(parents=True, exist_ok=True)

    context_frames = max(int(_option(settings, "context_frames", "num_context_frames", default=DEFAULT_CONTEXT_FRAMES)), 1)
    horizon = max(int(_option(settings, "horizon", "num_frames", default=DEFAULT_HORIZON)), 1)
    num_steps = max(int(_option(settings, "num_steps", "num_inference_steps", default=DEFAULT_NUM_STEPS)), 1)
    decode_chunk_size = max(int(_option(settings, "decode_chunk_size", default=DEFAULT_DECODE_CHUNK_SIZE)), 1)
    seed = int(_option(settings, "seed", default=0))
    parallel_strategy = str(_option(settings, "parallel_strategy", default=DEFAULT_PARALLEL_STRATEGY))

    video = _input_video(settings)
    if video is None:
        raise ValueError("Open Dreamer requires input_mp4 (or video=) pointing at a Minecraft/VPT context clip.")

    interactions = plan.get("interactions") or settings.get("interactions")
    actions_path = _resolve_actions_path(
        settings,
        interactions,
        context_frames=context_frames,
        horizon=horizon,
        output_dir=output_dir,
    )

    command = ["env"]
    env_overrides = {**DEFAULT_ENV_OVERRIDES, **dict(_option(settings, "env_overrides", default={}) or {})}
    command.extend(f"{key}={value}" for key, value in sorted(env_overrides.items()))
    command.extend(
        [
            _python_executable(settings, root),
            str(context["entrypoint"]),
            "--checkpoint_path",
            str(_checkpoint_path(settings)),
            "--input_mp4",
            str(video),
            "--actions_path",
            str(actions_path),
            "--output_mp4",
            str(output_path),
            "--context_frames",
            str(context_frames),
            "--horizon",
            str(horizon),
            "--num_steps",
            str(num_steps),
            "--decode_chunk_size",
            str(decode_chunk_size),
            "--seed",
            str(seed),
            "--parallel_strategy",
            parallel_strategy,
        ]
    )
    if _as_bool(_option(settings, "use_ema", default=True), True):
        command.append("--use_ema")
    if _as_bool(_option(settings, "no_kv_cache", default=False), False):
        command.append("--no_kv_cache")
    device = str(_option(settings, "device", default=context.get("device") or "cuda"))
    if _as_bool(_option(settings, "allow_cpu", default=False), False) or device.startswith("cpu"):
        command.append("--allow_cpu")
    return command


def resolved_runtime_report() -> dict[str, Any]:
    """Summarize how the adapter currently resolves the staged runtime.

    Used by diagnostics and tests that need to explain where WorldFoundry expects
    the Open Dreamer checkout and checkpoint to live.
    """
    root = runtime_root()
    entrypoint = root / ENTRYPOINT_RELATIVE
    return {
        "runtime_root": str(root),
        "runtime_root_exists": root.is_dir(),
        "entrypoint": str(entrypoint),
        "entrypoint_exists": entrypoint.is_file(),
        "checkpoint_dir": str(DEFAULT_CHECKPOINT_DIR),
        "checkpoint_dir_exists": DEFAULT_CHECKPOINT_DIR.is_dir(),
        "staging_summary": BLOCKED_REASON,
        "source_env_var": SOURCE_ENV_VAR,
        "official_repo_url": OFFICIAL_REPO_URL,
        "official_inference_repo_url": OFFICIAL_INFERENCE_REPO_URL,
        "python_executable": _python_executable({}, root),
        "env_binary": shutil.which("env") or "",
    }


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_CONTEXT_FRAMES",
    "DEFAULT_HORIZON",
    "ENTRYPOINT_RELATIVE",
    "MODEL_HEIGHT",
    "MODEL_WIDTH",
    "OFFICIAL_INFERENCE_REPO_URL",
    "OFFICIAL_REPO_URL",
    "RUNTIME_DIR",
    "SOURCE_DIR_NAMES",
    "SOURCE_ENV_VAR",
    "build_command",
    "missing_requirements",
    "resolved_runtime_report",
    "runtime_root",
]
