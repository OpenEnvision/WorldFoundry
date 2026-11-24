"""Command-line entrypoint for the in-tree ViPE inference runtime."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from worldfoundry.base_models.three_dimensions.general_3d.vipe.assets import prepare_pose_assets
from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.build import native_extension_status
from worldfoundry.base_models.three_dimensions.general_3d.vipe.runtime import infer_pose, preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldFoundry in-tree NVIDIA ViPE runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-native", help="Compile and import the pinned vipe_ext CUDA sources")
    build.add_argument("--verbose", action="store_true")

    check = subparsers.add_parser("preflight", help="Check native extension, CUDA toolchain, and checkpoints")
    check.add_argument("--build", action="store_true", help="Build vipe_ext when the ABI-keyed cache is empty")
    check.add_argument("--no-assets", action="store_true", help="Do not require pose checkpoints")
    check.add_argument("--verbose", action="store_true")

    assets = subparsers.add_parser("prepare-assets", help="Validate or explicitly download public pose checkpoints")
    assets.add_argument("--download", action="store_true")

    infer = subparsers.add_parser("infer-pose", help="Write pose/<video-stem>.npz from a real MP4")
    infer.add_argument("video")
    infer.add_argument("--output", required=True)
    infer.add_argument("--frame-start", type=int, default=0)
    infer.add_argument("--frame-end", type=int, default=-1)
    infer.add_argument("--frame-skip", type=int, default=1)
    infer.add_argument("--override", action="append", default=[], help="Additional Hydra pipeline override")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-native":
        result = native_extension_status(build_if_missing=True, verbose=args.verbose)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.ready else 1
    if args.command == "preflight":
        result = preflight(build_native=args.build, require_assets=not args.no_assets, verbose=args.verbose)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready"] else 1
    if args.command == "prepare-assets":
        resolved = prepare_pose_assets(download=args.download)
        print(json.dumps({"ready": True, "assets": [asset.to_dict() for asset in resolved]}, indent=2, sort_keys=True))
        return 0
    if args.command == "infer-pose":
        result = infer_pose(
            args.video,
            args.output,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            frame_skip=args.frame_skip,
            hydra_overrides=args.override,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
