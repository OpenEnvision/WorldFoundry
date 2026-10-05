import argparse
import json

from .da3_video_index import build_da3_video_index


def build_parser():
    parser = argparse.ArgumentParser(description="Pre-scan Matrix-Game DA3 inference inputs into SQLite.")
    parser.add_argument(
        "--camera_params_path",
        required=True,
        help="Comma-separated DA3 output roots. Each root may contain many scenes.",
    )
    parser.add_argument(
        "--prompt_path",
        default="",
        help="Comma-separated prompt roots matching DA3 scene names.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output SQLite index path, e.g. da3_video_index.sqlite.",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--worker_backend",
        type=str,
        default="process",
        choices=["process", "thread", "serial"],
        help=(
            "Parallel backend for root discovery and per-scene probes. "
            "Default process uses multiple Python processes; thread is useful "
            "for lighter IO-only scans; serial disables parallelism."
        ),
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help=(
            "Stop DA3 scene discovery after this many unique scenes. Useful "
            "for smoke tests on large/NFS-mounted datasets. None/<=0 scans all."
        ),
    )
    parser.add_argument(
        "--max_scenes_per_root",
        type=int,
        default=None,
        help=(
            "Optional cap applied independently to each comma-separated "
            "camera_params_path root. None/<=0 disables it. If --max_scenes "
            "is also set, the global cap still stops discovery once reached."
        ),
    )
    parser.add_argument("--min_frame_count", type=int, default=1)
    parser.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="Disable progress bars and scan summaries.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Replace an existing index atomically.",
    )
    parser.add_argument(
        "--append",
        default=False,
        action="store_true",
        help=(
            "Incrementally append newly scanned roots into an existing index. "
            "Existing clean_name records are preserved."
        ),
    )
    parser.add_argument(
        "--require_dynamic_masks",
        default=False,
        action="store_true",
        help=(
            "Require each complete scene to contain the offline dynamic object "
            "mask JSONL before it is kept in the inference index."
        ),
    )
    parser.add_argument(
        "--dynamic_mask_subdir",
        default="objects",
        help="Scene-relative directory containing dynamic object masks.",
    )
    parser.add_argument(
        "--dynamic_mask_filename",
        default="dynamic_masks.jsonl",
        help="Dynamic object mask filename inside --dynamic_mask_subdir.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    stats = build_da3_video_index(
        camera_params_path=args.camera_params_path,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        min_frame_count=args.min_frame_count,
        require_complete=True,
        overwrite=args.overwrite,
        append=args.append,
        workers=args.workers,
        worker_backend=args.worker_backend,
        max_scenes=args.max_scenes,
        max_scenes_per_root=args.max_scenes_per_root,
        require_dynamic_masks=args.require_dynamic_masks,
        dynamic_mask_subdir=args.dynamic_mask_subdir,
        dynamic_mask_filename=args.dynamic_mask_filename,
        verbose=not args.quiet,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
