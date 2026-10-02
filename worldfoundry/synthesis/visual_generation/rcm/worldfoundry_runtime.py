"""Runtime adapter for the vendored Causal-rCM streaming video runtime.

Causal-rCM is NVIDIA's block-causal extension of rCM: a Wan2.1 backbone
fine-tuned under block-causal attention, then distilled to 1-4 steps per chunk
with teacher-forcing consistency and self-forcing DMD. Generation is
autoregressive over latent-frame chunks, which is what makes it usable as a
streaming/interactive world model rather than a one-shot clip generator.

The official source is Apache-2.0 and vendored under ``rcm_runtime`` (see its
``THIRD_PARTY_NOTICES.md``), so unlike source-bound manifests nothing has to be
staged from GitHub. Execution is still checkpoint gated: Wan2.1 VAE, the umT5
text encoder, and a Causal-rCM DiT checkpoint must be present.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings

RUNTIME_DIR = Path(__file__).resolve().parent
INFERENCE_ENTRYPOINT = RUNTIME_DIR / "rcm_runtime" / "inference" / "wan2pt1_t2v_causal_infer.py"

OFFICIAL_REPO_URL = "https://github.com/NVlabs/rcm"

DEFAULT_CHECKPOINT_DIR = checkpoint_root_path("rcm")
DEFAULT_DIT_FILENAME = "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt"
DEFAULT_VAE_FILENAME = "Wan2.1_VAE.pth"
DEFAULT_TEXT_ENCODER_FILENAME = "models_t5_umt5-xxl-enc-bf16.pth"
RUNTIME_ENV_NAME = "causal-rcm"

# Exact byte counts published by the public Hub files. Checking the size keeps a
# partially-resumed download from passing the runtime gate as a valid checkpoint.
PUBLIC_CHECKPOINT_SIZES = {
    DEFAULT_DIT_FILENAME: 2_838_292_823,
    DEFAULT_VAE_FILENAME: 507_609_880,
    DEFAULT_TEXT_ENCODER_FILENAME: 11_361_920_418,
}

# Distilled chunk schedule from the Causal-rCM recipe. Frame-wise `c1-1` chunks
# give the lowest streaming latency; `_c3-3` trades latency for throughput.
DEFAULT_FIRST_CHUNK_T = 1
DEFAULT_CHUNK_T = 1
DEFAULT_NUM_STEPS = 4
DEFAULT_MID_T = ("15/16", "5/6", "5/8")
DEFAULT_NUM_FRAMES = 81
DEFAULT_RESOLUTION = "480p"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_SIGMA_MAX = 1600.0

# Python packages the vendored runtime imports beyond WorldFoundry's core set.
REQUIRED_MODULES = (
    "numpy",
    "PIL",
    "torch",
    "torchvision",
    "einops",
    "transformers",
    "ftfy",
    "regex",
    "safetensors",
    "imageio",
    "tqdm",
    "omegaconf",
)

BLOCKED_REASON = (
    "Causal-rCM source is vendored in-tree under Apache-2.0; execution requires a Causal-rCM DiT "
    f"checkpoint plus the Wan2.1 VAE and umT5 text encoder staged under {DEFAULT_CHECKPOINT_DIR}."
)


def _option(settings: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first setting present under any of ``names``."""
    for name in names:
        value = settings.get(name)
        if value not in (None, ""):
            return value
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    """Coerce values that may arrive as strings from a serialized plan."""
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


def _checkpoint_dir(settings: Mapping[str, Any]) -> Path:
    """Resolve the directory holding the Wan2.1 VAE and umT5 encoder."""
    value = _option(settings, "checkpoint_dir", "checkpoint_root", default=DEFAULT_CHECKPOINT_DIR)
    return Path(str(value)).expanduser()


def _dit_path(settings: Mapping[str, Any]) -> Path | None:
    """Resolve the Causal-rCM DiT checkpoint."""
    value = _option(
        settings,
        "dit_path",
        "checkpoint_path",
        "ckpt_path",
        "model_path",
        "pretrained_model_path",
        default=_checkpoint_dir(settings) / DEFAULT_DIT_FILENAME,
    )
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _vae_path(settings: Mapping[str, Any]) -> Path:
    """Resolve the Wan2.1 VAE checkpoint."""
    value = _option(settings, "vae_path", default=_checkpoint_dir(settings) / DEFAULT_VAE_FILENAME)
    return Path(str(value)).expanduser()


def _text_encoder_path(settings: Mapping[str, Any]) -> Path:
    """Resolve the umT5 text-encoder checkpoint."""
    value = _option(
        settings,
        "text_encoder_path",
        default=_checkpoint_dir(settings) / DEFAULT_TEXT_ENCODER_FILENAME,
    )
    return Path(str(value)).expanduser()


def _image_path(settings: Mapping[str, Any]) -> Path | None:
    """Resolve the optional image-to-video conditioning frame."""
    value = _option(settings, "image_path", "images", "input_image")
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _checkpoint_missing_reason(label: str, path: Path) -> str | None:
    """Return a checkpoint gate reason, including incomplete public downloads."""
    if not path.is_file():
        return f"{label} checkpoint does not exist"
    expected_size = PUBLIC_CHECKPOINT_SIZES.get(path.name)
    if expected_size is not None and path.stat().st_size != expected_size:
        return (
            f"{label} checkpoint is incomplete: expected {expected_size} bytes, "
            f"found {path.stat().st_size} bytes"
        )
    return None


def _python_executable(settings: Mapping[str, Any]) -> str:
    """Pick the interpreter that runs the rollout.

    The vendored runtime pins torch 2.11 / CUDA 12.6 and can live in the
    dedicated ``causal-rcm`` conda environment instead of WorldFoundry's shared
    one. An explicit ``python_executable`` wins; otherwise use that environment
    automatically when it has been created under ``WORLDFOUNDRY_CONDA_ENVS_ROOT``.
    """
    explicit = _option(settings, "python_executable", "python_bin")
    if explicit:
        return str(explicit)
    env_root = os.getenv("WORLDFOUNDRY_CONDA_ENVS_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser() / RUNTIME_ENV_NAME / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report the checkpoints and packages a Causal-rCM rollout still needs."""
    del runtime_root, profile
    settings = dict(options or {})
    missing: list[dict[str, str]] = []

    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": "vendored Causal-rCM inference entrypoint is missing",
            }
        )

    dit_path = _dit_path(settings)
    if dit_path is None:
        missing.append(
            {
                "kind": "checkpoint",
                "path": "dit_path",
                "reason": (
                    "Causal-rCM requires dit_path (or checkpoint_path) pointing at a distilled causal DiT "
                    f"checkpoint; see {OFFICIAL_REPO_URL}"
                ),
            }
        )
    else:
        reason = _checkpoint_missing_reason("Causal-rCM DiT", dit_path)
        if reason:
            missing.append({"kind": "checkpoint", "path": str(dit_path), "reason": reason})

    for label, path in (("Wan2.1 VAE", _vae_path(settings)), ("umT5 text encoder", _text_encoder_path(settings))):
        reason = _checkpoint_missing_reason(label, path)
        if reason:
            missing.append({"kind": "checkpoint", "path": str(path), "reason": reason})

    image_path = _image_path(settings)
    if image_path is not None and not image_path.is_file():
        missing.append({"kind": "asset", "path": str(image_path), "reason": "Causal-rCM I2V conditioning image does not exist"})

    # Only meaningful when the rollout runs in this interpreter. A configured
    # python_executable points at the dedicated causal-rcm environment, whose
    # packages cannot be probed from here.
    if _python_executable(settings) == sys.executable:
        for module_name in REQUIRED_MODULES:
            if importlib.util.find_spec(module_name) is None:
                missing.append(
                    {
                        "kind": "python_module",
                        "path": module_name,
                        "reason": (
                            "required Causal-rCM runtime package is not importable; install it or set "
                            "python_executable to the causal-rcm environment interpreter"
                        ),
                    }
                )

    return missing


def _mid_t_arguments(settings: Mapping[str, Any]) -> list[str]:
    """Normalize the intermediate-timestep schedule into CLI tokens."""
    value = _option(settings, "mid_t", default=list(DEFAULT_MID_T))
    if isinstance(value, str):
        # Accept "15/16,5/6,5/8" as well as a real sequence.
        items = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        items = [str(item).strip() for item in value]
    return [item for item in items if item]


def build_command(context) -> list[str]:
    """Build the vendored Causal-rCM rollout command for one WorldFoundry request."""
    settings = command_settings(context)
    output_path = Path(str(context["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dit_path = _dit_path(settings)
    if dit_path is None:
        raise ValueError("Causal-rCM requires dit_path (or checkpoint_path) pointing at a distilled causal DiT checkpoint.")

    command = [
        _python_executable(settings),
        str(context["entrypoint"]),
        "--dit_path",
        str(dit_path),
        "--vae_path",
        str(_vae_path(settings)),
        "--text_encoder_path",
        str(_text_encoder_path(settings)),
        "--save_path",
        str(output_path),
        "--prompt",
        str(_option(settings, "prompt", default=context.get("prompt") or "")),
        "--model_size",
        str(_option(settings, "model_size", default="1.3B")),
        "--first_chunk_t",
        str(int(_option(settings, "first_chunk_t", default=DEFAULT_FIRST_CHUNK_T))),
        "--chunk_t",
        str(int(_option(settings, "chunk_t", default=DEFAULT_CHUNK_T))),
        "--num_frames",
        str(int(_option(settings, "num_frames", default=DEFAULT_NUM_FRAMES))),
        "--num_steps",
        str(int(_option(settings, "num_steps", default=DEFAULT_NUM_STEPS))),
        "--resolution",
        str(_option(settings, "resolution", default=DEFAULT_RESOLUTION)),
        "--aspect_ratio",
        str(_option(settings, "aspect_ratio", default=DEFAULT_ASPECT_RATIO)),
        "--seed",
        str(int(_option(settings, "seed", default=0))),
        "--num_samples",
        str(int(_option(settings, "num_samples", default=1))),
    ]

    # Distilled few-step sampling is the point of Causal-rCM; the multi-step
    # causal-diffusion teacher path stays reachable with distilled=False.
    if _as_bool(_option(settings, "distilled", default=True), True):
        command.append("--distilled")
        command.append("--sigma_max")
        command.append(str(float(_option(settings, "sigma_max", default=DEFAULT_SIGMA_MAX))))
        mid_t = _mid_t_arguments(settings)
        if mid_t:
            command.append("--mid_t")
            command.extend(mid_t)
        steps_per_chunk = _option(settings, "steps_per_chunk")
        if steps_per_chunk:
            values = steps_per_chunk.split() if isinstance(steps_per_chunk, str) else list(steps_per_chunk)
            command.append("--steps_per_chunk")
            command.extend(str(int(item)) for item in values)
        mid_t_schedules = _option(settings, "mid_t_schedules")
        if mid_t_schedules:
            command.extend(["--mid_t_schedules", str(mid_t_schedules)])
    else:
        command.extend(["--guidance_scale", str(float(_option(settings, "guidance_scale", default=3.0)))])
        command.extend(["--timestep_shift", str(float(_option(settings, "timestep_shift", default=3.0)))])

    negative_prompt = _option(settings, "negative_prompt")
    if negative_prompt:
        command.extend(["--negative_prompt", str(negative_prompt)])

    image_path = _image_path(settings)
    if image_path is not None:
        command.extend(["--image_path", str(image_path)])
        if _as_bool(_option(settings, "adaptive_resolution", default=False)):
            command.append("--adaptive_resolution")

    # Noisy context caches the final denoising forward instead of running an
    # extra clean pass per chunk, saving one forward at matched quality.
    if _as_bool(_option(settings, "context_from_last_step", default=False)):
        command.append("--context_from_last_step")
        command.extend(
            ["--context_from_last_step_start_chunk", str(int(_option(settings, "context_from_last_step_start_chunk", default=0)))]
        )

    # The upstream cache remains the default. A bounded sliding window is the
    # only cache policy exposed by this post-RoPE causal entrypoint; policies
    # that remap temporal RoPE positions require the separate extrapolation
    # runner and are deliberately rejected by its CLI.
    kv_cache_policy = str(_option(settings, "kv_cache_policy", default="keep_all")).strip().lower().replace("-", "_")
    if kv_cache_policy not in {"keep_all", "sliding_window"}:
        raise ValueError(
            "Causal-rCM supports kv_cache_policy=keep_all or sliding_window. "
            "RoPE-remapping policies require a pre-RoPE extrapolation runtime."
        )
    if kv_cache_policy != "keep_all":
        command.extend(["--kv_cache_policy", kv_cache_policy])
        command.extend(
            ["--kv_cache_window_blocks", str(int(_option(settings, "kv_cache_window_blocks", default=6)))]
        )
        command.extend(
            ["--kv_cache_sink_blocks", str(int(_option(settings, "kv_cache_sink_blocks", default=0)))]
        )

    warmup_iters = int(_option(settings, "warmup_iters", default=0))
    num_runs = int(_option(settings, "num_runs", default=1))
    if warmup_iters or num_runs > 1:
        command.extend(["--warmup_iters", str(warmup_iters), "--num_runs", str(num_runs)])

    return command


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_DIT_FILENAME",
    "INFERENCE_ENTRYPOINT",
    "OFFICIAL_REPO_URL",
    "PUBLIC_CHECKPOINT_SIZES",
    "REQUIRED_MODULES",
    "RUNTIME_ENV_NAME",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
]
