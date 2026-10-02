"""CLI options and launch configuration for Studio browser frontends."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence

from worldfoundry.studio.inference.catalog import CatalogEntry, find_entry
from worldfoundry.studio.inference.config import (
    DREAMX_WORLD_MODEL_ID,
    HELIOS_MODEL_ID,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
    LINGBOT_WORLD_V2_MODEL_ID,
    LONGVIE2_MODEL_ID,
    MATRIX_GAME3_MODEL_ID,
    _torchrun_world_size,
    env_first,
)
from worldfoundry.studio.inference.paths import studio_path_summary
from worldfoundry.studio.visualization.core.registry import (
    EMBODIED_VISUALIZATION,
    INTERACTIVE_WORLD_VISUALIZATION,
    MEDIA_VISUALIZATION,
    RERUN_VISUALIZATION,
    SPARK_VISUALIZATION,
    VISER_VISUALIZATION,
)

CLI_BACKEND_CHOICES = frozenset({"auto", "from_pretrained", "api_init"})
CLI_FRONTEND_CHOICES = frozenset(
    {
        "auto",
        INTERACTIVE_WORLD_VISUALIZATION,
        "interactive-world",
        "world-model",
        VISER_VISUALIZATION,
        "viser",
        "geometry",
        "pointcloud",
        "point-cloud",
        EMBODIED_VISUALIZATION,
        "sim",
        "simulator",
        MEDIA_VISUALIZATION,
        "preview",
        "video",
        "image",
        RERUN_VISUALIZATION,
        "rrd",
        SPARK_VISUALIZATION,
        "3dgs",
        "splat",
        "gaussian-splat",
    }
)


@dataclass(frozen=True)
class StudioLaunchConfig:
    model_id: str
    variant_id: str | None = None
    model_ref: str = ""
    device: str = "cuda"
    backend: str = "auto"
    endpoint: str = ""
    frontend: str = "auto"
    asset_path: str = ""
    simulator_url: str = ""
    host: str = ""
    port: int | None = None


def build_launch_argument_parser(prog: str = "worldfoundry-studio") -> argparse.ArgumentParser:
    from worldfoundry.cli.help import WorldFoundryArgumentParser

    parser = WorldFoundryArgumentParser(
        prog=prog,
        description="Launch a WorldFoundry Studio frontend with one fixed model session.",
        epilog=(
            "Path roots: WORLDFOUNDRY_STUDIO_WORKSPACE_DIR controls Studio runs; "
            "WORLDFOUNDRY_MODEL_DIR and WORLDFOUNDRY_CACHE_DIR are searched for local checkpoints/repos."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Studio model id or alias to lock at launch, for example `lingbot-world`.",
    )
    parser.add_argument(
        "--model",
        dest="model_flag",
        help="Studio model id or alias to lock at launch.",
    )
    parser.add_argument(
        "--variant",
        help="Optional variant id or alias, for example `fast` or `base-camera`.",
    )
    parser.add_argument(
        "--model-ref",
        "--ckpt",
        "--checkpoint",
        dest="model_ref",
        help="Checkpoint path or repo id override for the fixed model.",
    )
    parser.add_argument(
        "--device",
        help="Default runtime device shown in the UI, for example `cuda:1`.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(CLI_BACKEND_CHOICES),
        help="Loader override for the fixed session.",
    )
    parser.add_argument(
        "--endpoint",
        help="Endpoint override for hosted or API-backed models.",
    )
    parser.add_argument(
        "--frontend",
        choices=sorted(CLI_FRONTEND_CHOICES),
        help=(
            "Frontend surface to launch. "
            "`world` is the game-console shell, "
            "`points` is native Viser, `spark` is a standalone 3DGS viewer, "
            "`embodied` preserves an external simulator UI, "
            "and `auto` routes by model type."
        ),
    )
    parser.add_argument(
        "--asset",
        dest="asset_path",
        help="Frontend asset path for native viewers, e.g. a .ply/.npz point cloud or .splat/.spz/.ply 3DGS file.",
    )
    parser.add_argument(
        "--simulator-url",
        help="Native embodied simulator URL to advertise instead of embedding in the game-console frontend.",
    )
    parser.add_argument(
        "--host",
        help="Frontend bind host. Defaults to loopback unless environment variables override it.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Frontend bind port. Defaults depend on the selected frontend.",
    )
    return parser


def launch_environment_summary() -> dict[str, str | list[str]]:
    """Return startup path discovery details for logs, tests, and readiness checks.

    Args:
        None.
    """

    return studio_path_summary()


def launch_uses_lingbot_torchrun_rollout(launch_config: StudioLaunchConfig) -> bool:
    world_size = _torchrun_world_size()
    if launch_config.model_id == LONGVIE2_MODEL_ID:
        if world_size not in {1, 4}:
            raise ValueError(
                "LongVie 2 supports either one GPU or the official four-rank USP topology; "
                f"got WORLD_SIZE={world_size}."
            )
        return world_size == 4
    if world_size <= 1:
        return False
    if launch_config.model_id == LINGBOT_WORLD_V2_MODEL_ID:
        if world_size not in {4, 8}:
            raise ValueError(
                "LingBot-World-V2 Workspace supports four or eight torchrun ranks; "
                f"got WORLD_SIZE={world_size}."
            )
        return True
    if (
        launch_config.model_id == LINGBOT_WORLD_MODEL_ID
        and launch_config.variant_id == LINGBOT_VARIANT_FAST
    ):
        if world_size not in {4, 8}:
            raise ValueError(
                "LingBot-World Fast Workspace supports four or eight torchrun ranks; "
                f"got WORLD_SIZE={world_size}."
            )
        return True
    if launch_config.model_id in {
        MATRIX_GAME3_MODEL_ID,
        HELIOS_MODEL_ID,
        DREAMX_WORLD_MODEL_ID,
    }:
        return True
    return False


def parse_launch_config(
    argv: Sequence[str] | None = None,
    *,
    entries: Sequence[CatalogEntry] | None = None,
    studio_catalog: Callable[[Sequence[CatalogEntry] | None], tuple[CatalogEntry, ...]] | None = None,
    resolve_cli_variant_id: Callable[[CatalogEntry, str | None], str | None] | None = None,
) -> StudioLaunchConfig:
    """Parse argv plus env overrides into a frozen launch configuration.

    Args:
        argv: Optional CLI tokens; forwarded to argparse (defaults match ``parse_args(None)``).
        entries: Optional catalog rows for fallback model ordering when env is unset.
        studio_catalog: Sorted catalog factory; lazily resolves to the lightweight Studio catalog helper when omitted.
        resolve_cli_variant_id: Variant resolver; lazily resolves to the lightweight variant helper when omitted.
    """
    if studio_catalog is None or resolve_cli_variant_id is None:
        if studio_catalog is None:
            from worldfoundry.studio.ui.catalog import _studio_catalog

            studio_catalog = _studio_catalog
        if resolve_cli_variant_id is None:
            from worldfoundry.studio.inference.variants import resolve_cli_variant_id

    parser = build_launch_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    raw_model_flag = (args.model_flag or "").strip()
    raw_model_positional = (args.model or "").strip()
    if raw_model_flag and raw_model_positional and raw_model_flag.lower() != raw_model_positional.lower():
        parser.error("Use either the positional model argument or `--model`, not two different model ids.")

    raw_model = (
        raw_model_flag
        or raw_model_positional
        or env_first("WORLDFOUNDRY_STUDIO_MODEL").strip()
    )
    if not raw_model:
        catalog = tuple(entries) if entries is not None else studio_catalog()
        fallback_model_id = catalog[0].model_id if catalog else ""
        raw_model = fallback_model_id
    if not raw_model:
        parser.error("Studio catalog is empty; no launchable model is available.")

    try:
        entry = find_entry(raw_model)
    except Exception as exc:
        parser.error(str(exc))

    try:
        variant_id = resolve_cli_variant_id(
            entry,
            (
                args.variant
                or env_first("WORLDFOUNDRY_STUDIO_VARIANT")
            ).strip()
            or None,
        )
    except Exception as exc:
        parser.error(str(exc))

    backend = (
        (
            args.backend
            or env_first("WORLDFOUNDRY_STUDIO_BACKEND")
        ).strip()
        or entry.default_backend
        or "auto"
    )
    if backend not in CLI_BACKEND_CHOICES:
        parser.error(
            f"Unsupported backend `{backend}`. Use one of: {', '.join(sorted(CLI_BACKEND_CHOICES))}."
        )

    frontend = (
        args.frontend
        or env_first("WORLDFOUNDRY_STUDIO_FRONTEND")
        or "auto"
    ).strip().lower()
    if frontend not in CLI_FRONTEND_CHOICES:
        parser.error(
            f"Unsupported frontend `{frontend}`. Use one of: {', '.join(sorted(CLI_FRONTEND_CHOICES))}."
        )

    port_text = (
        str(args.port)
        if args.port is not None
        else env_first("WORLDFOUNDRY_STUDIO_PORT")
        or ""
    ).strip()
    try:
        port = int(port_text) if port_text else None
    except ValueError:
        parser.error(f"Invalid Studio port `{port_text}`.")

    return StudioLaunchConfig(
        model_id=entry.model_id,
        variant_id=variant_id,
        model_ref=(
            args.model_ref
            or env_first("WORLDFOUNDRY_STUDIO_FIXED_MODEL_REF")
        ).strip(),
        device=(
            args.device
            or env_first("WORLDFOUNDRY_STUDIO_DEVICE")
        ).strip()
        or "cuda",
        backend=backend,
        endpoint=(
            args.endpoint
            or env_first("WORLDFOUNDRY_STUDIO_ENDPOINT")
        ).strip()
        or entry.default_endpoint
        or "",
        frontend=frontend,
        asset_path=(
            args.asset_path
            or env_first("WORLDFOUNDRY_STUDIO_ASSET")
        ).strip(),
        simulator_url=(
            args.simulator_url
            or env_first("WORLDFOUNDRY_STUDIO_SIMULATOR_URL")
        ).strip(),
        host=(
            args.host
            or env_first("WORLDFOUNDRY_STUDIO_HOST")
            or ""
        ).strip(),
        port=port,
    )
