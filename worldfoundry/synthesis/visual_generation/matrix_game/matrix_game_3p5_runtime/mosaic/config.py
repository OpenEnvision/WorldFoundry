"""Compact configuration front door for Matrix-Game 3.5 inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from ..config_paths import MATRIX_GAME_35_CONFIG_ROOT, matrix_game_35_infer_config_path

MOSAIC_INTRINSICS_MODES = ("per_frame", "episode_mean", "first_frame")
MOSAIC_FUSE_MODES = ("masked_mean", "fill_stop", "fill_stop_zbuffer")
_CONFIG_ROOT = MATRIX_GAME_35_CONFIG_ROOT
_DEFAULT_CONFIG = matrix_game_35_infer_config_path("common")
_OPTION_TYPES = {
    "max_data_items": int,
    "max_frames_per_scene": int,
    "max_scan_items": int,
    "min_frame_count": int,
}


def _normalize_mosaic_fuse_mode(mode):
    value = str(mode or "fill_stop_zbuffer")
    if value not in MOSAIC_FUSE_MODES:
        raise ValueError(f"Mosaic fuse mode must be one of {MOSAIC_FUSE_MODES}, got {value!r}")
    return value


def _normalize_mosaic_intrinsics_mode(mode):
    value = str(mode or "per_frame").strip().lower()
    aliases = {
        "per-frame": "per_frame",
        "perframe": "per_frame",
        "frame": "per_frame",
        "mean": "episode_mean",
        "episode-mean": "episode_mean",
        "kmean": "episode_mean",
        "first": "first_frame",
        "first-frame": "first_frame",
        "frame0": "first_frame",
    }
    value = aliases.get(value, value)
    if value not in MOSAIC_INTRINSICS_MODES:
        raise ValueError(f"Mosaic intrinsics mode must be one of {MOSAIC_INTRINSICS_MODES}, got {value!r}")
    return value


def _intrinsics_mode_arg(args, name, default):
    return _normalize_mosaic_intrinsics_mode(getattr(args, name, default))


def _apply_intrinsics_mode(args):
    args.mosaic_intrinsics_mode = _normalize_mosaic_intrinsics_mode(
        getattr(args, "mosaic_intrinsics_mode", "episode_mean")
    )
    return args


def _load_config(path, *, stack=()):
    config_path = Path(path).expanduser().resolve()
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Cyclic Matrix-Game config inheritance: {chain}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Matrix-Game config must be a mapping: {config_path}")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    merged = _load_config(parent_path, stack=(*stack, config_path))
    merged.update(payload)
    return merged


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _option_type(key, value):
    if key in _OPTION_TYPES:
        return _OPTION_TYPES[key]
    if key == "trans_scale":
        return str
    if isinstance(value, bool):
        return _parse_bool
    if isinstance(value, int) and not isinstance(value, bool):
        return int
    if isinstance(value, float):
        return float
    return str


def _add_config_option(parser, key, value):
    option_strings = [f"--{key}"]
    hyphenated = key.replace("_", "-")
    if hyphenated != key:
        option_strings.append(f"--{hyphenated}")
    kwargs = {
        "dest": key,
        "default": argparse.SUPPRESS,
        "type": _option_type(key, value),
    }
    if isinstance(value, bool):
        kwargs.update({"nargs": "?", "const": True})
    parser.add_argument(*option_strings, **kwargs)
    if isinstance(value, bool):
        inverse = hyphenated.removeprefix("no-") if hyphenated.startswith("no-") else f"no-{hyphenated}"
        parser.add_argument(
            f"--{inverse}",
            dest=key,
            action="store_false",
            default=argparse.SUPPRESS,
        )


def wan_mosaic_parser(config=None):
    values = dict(_load_config(_DEFAULT_CONFIG) if config is None else config)
    parser = argparse.ArgumentParser(description="WorldFoundry-native Matrix-Game 3.5 inference runner.")
    parser.add_argument(
        "--omegaconf_config",
        "--config",
        dest="omegaconf_config",
        default=argparse.SUPPRESS,
        help="Inference YAML; supports an optional relative 'extends' key.",
    )
    for key, value in sorted(values.items()):
        if key == "omegaconf_config":
            continue
        _add_config_option(parser, key, value)
    return parser


def _normalize_trans_scale(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"log", "logd4", "tanh"}:
            return normalized
        try:
            return float(normalized)
        except ValueError as exc:
            raise ValueError("trans_scale must be numeric, 'log', 'logd4', or 'tanh'") from exc
    return float(value)


def _validate_inference_config(args):
    if int(args.latent_window_size) <= 0:
        raise ValueError("latent_window_size must be positive")
    if int(args.num_inference_batches) <= 0:
        raise ValueError("num_inference_batches must be positive")
    if int(args.num_inference_blocks) <= 0:
        raise ValueError("num_inference_blocks must be positive")
    if int(args.num_inference_steps) <= 0:
        raise ValueError("num_inference_steps must be positive")
    if float(args.guidance_scale) <= 0:
        raise ValueError("guidance_scale must be positive")
    if args.prope_disable_native_rope and args.prope_disable_t_rope:
        raise ValueError("prope_disable_native_rope and prope_disable_t_rope are mutually exclusive")
    if (args.prope_disable_native_rope or args.prope_disable_t_rope) and int(args.prope_attention_interval) <= 1:
        raise ValueError("Disabling native/temporal RoPE requires prope_attention_interval > 1")
    if str(args.prope_camera_layout) != "full" and not args.prope_disable_t_rope:
        raise ValueError("A non-default prope_camera_layout requires prope_disable_t_rope")
    args.trans_scale = _normalize_trans_scale(args.trans_scale)
    args.mosaic_fuse_mode = _normalize_mosaic_fuse_mode(args.mosaic_fuse_mode)
    return _apply_intrinsics_mode(args)


def parse_pipeline_args(argv=None):
    argv = list(argv) if argv is not None else None
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--omegaconf_config",
        "--config",
        dest="omegaconf_config",
    )
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_path = Path(pre_args.omegaconf_config or _DEFAULT_CONFIG)
    values = _load_config(config_path)
    parser = wan_mosaic_parser(values)
    explicit = vars(parser.parse_args(argv))
    values.update(explicit)
    values["omegaconf_config"] = str(config_path.expanduser().resolve())
    return _validate_inference_config(SimpleNamespace(**values))


def _dump_run_args_yaml(args, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            vars(args),
            handle,
            allow_unicode=True,
            sort_keys=True,
        )
    return str(destination)


__all__ = [
    "MOSAIC_INTRINSICS_MODES",
    "_apply_intrinsics_mode",
    "_dump_run_args_yaml",
    "_intrinsics_mode_arg",
    "_normalize_mosaic_fuse_mode",
    "parse_pipeline_args",
    "wan_mosaic_parser",
]
