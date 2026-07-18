import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = "worldfoundry.evaluation.tasks.execution.runners.larybench"
SPLIT_DATASETS = {"agibotbeta", "robocoin"}
STATS_JSON_NAME = {
    "agibotbeta": "agibotbeta_stats.json",
    "robocoin": "robocoin_stats.json",
}
REGRESSION_DATA_SUBDIR = {
    "calvin": "calvin",
    "vlabench": "vlabench",
    "vlabench_15": "vlabench",
    "vlabench_30": "vlabench",
    "agibotbeta": "agibot_45",
    "robocoin": "robocoin_10",
}


def create_parser():
    parser = argparse.ArgumentParser(prog="lary")
    commands = parser.add_subparsers(dest="command")

    extract = commands.add_parser("extract")
    extract.add_argument("--model", required=True)
    extract.add_argument("--dataset", required=True)
    extract.add_argument("--input")
    extract.add_argument("--output")
    extract.add_argument("--split", default="all")
    extract.add_argument("--batch-size", type=int, default=16)
    extract.add_argument("--num-workers", type=int, default=8)
    extract.add_argument("--gpus", default="0")
    extract.add_argument("--mode", choices=("video", "image"), default="video")
    extract.add_argument("--stride", type=int, default=5)
    extract.add_argument("--perspective", default="1st")
    extract.add_argument("--partition", type=int, default=0)
    extract.add_argument("--num-partitions", type=int, default=1)

    classify = commands.add_parser("classify")
    classify.add_argument("--model", required=True)
    classify.add_argument("--dataset", required=True)
    classify.add_argument("--dim", type=int, required=True)
    classify.add_argument("--classes", type=int, required=True)
    classify.add_argument("--config")
    classify.add_argument("--batch-size", type=int, default=256)
    classify.add_argument("--num-workers", type=int, default=8)
    classify.add_argument("--loader-timeout", type=int, default=0)
    classify.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    classify.add_argument("--master-port", type=int)

    regress = commands.add_parser("regress")
    regress.add_argument("--model", required=True)
    regress.add_argument("--dataset", required=True)
    regress.add_argument("--stride", type=int, required=True)
    regress.add_argument("--model-type", choices=("mlp", "dit"), required=True)
    regress.add_argument("--action-mode", choices=("absolute", "relative"), default="absolute")
    regress.add_argument("--action-data-root")
    regress.add_argument("--batch-size", type=int, default=256)
    regress.add_argument("--num-workers", type=int, default=8)
    regress.add_argument("--epochs", type=int, default=20)
    regress.add_argument("--lr", type=float, default=1e-4)
    regress.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    regress.add_argument("--global-stats-json")
    regress.add_argument("--val-unseen-csv")
    return parser


def _parse_gpu_ids(value):
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("--gpus must contain at least one GPU id")
    normalized = []
    for item in ids:
        if item.lower() == "cpu":
            if len(ids) != 1:
                raise ValueError("--gpus cpu cannot be combined with CUDA devices")
            normalized.append("cpu")
        elif item.isdigit() or item.startswith(("GPU-", "MIG-")):
            normalized.append(item)
        else:
            raise ValueError(f"unsupported device selector {item!r}; use CUDA indices, GPU/MIG UUIDs, or cpu")
    return normalized


def _free_port(start, end):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            if handle.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free local port in [{start}, {end})")


def _extract_csv_base_name(args):
    suffix = f"_{args.stride}" if args.mode == "image" else ""
    return f"{args.split}_la_{args.dataset}{suffix}_{args.model}"


def _merge_extract_partition_csvs(args, count):
    import pandas as pd

    from .config import get_config

    output_dir = get_config().data_dir
    base_name = _extract_csv_base_name(args)
    partition_paths = [output_dir / f"{base_name}_{index}.csv" for index in range(count)]
    missing = [path for path in partition_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing extraction partition CSVs:\n" + "\n".join(f"  {path}" for path in missing))
    output_path = output_dir / f"{base_name}.csv"
    pd.concat((pd.read_csv(path) for path in partition_paths), ignore_index=True).to_csv(output_path, index=False)
    return output_path


def _spawn_extract_partitions(args, gpus):
    processes = []
    for partition, gpu in enumerate(gpus):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else str(gpu)
        if gpu == "cpu":
            environment["LARY_DEVICE"] = "cpu"
        command = [
            sys.executable,
            "-m",
            f"{MODULE_ROOT}.cli",
            "extract",
            "--model",
            args.model,
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--gpus",
            str(gpu),
            "--mode",
            args.mode,
            "--stride",
            str(args.stride),
            "--perspective",
            args.perspective,
            "--partition",
            str(partition),
            "--num-partitions",
            str(len(gpus)),
        ]
        if args.input:
            command.extend(("--input", args.input))
        if args.output:
            command.extend(("--output", args.output))
        processes.append((partition, subprocess.Popen(command, env=environment)))

    failures = [(partition, process.wait()) for partition, process in processes]
    failures = [(partition, code) for partition, code in failures if code]
    if failures:
        raise RuntimeError(f"Extraction partitions failed: {failures}")
    print(_merge_extract_partition_csvs(args, len(gpus)))


def run_extract(args):
    from .extract import extract_latent_actions

    gpus = _parse_gpu_ids(args.gpus)
    if len(gpus) > 1 and args.num_partitions == 1 and args.partition == 0:
        _spawn_extract_partitions(args, gpus)
        return
    extract_latent_actions(
        model=args.model,
        dataset=args.dataset,
        input_file=args.input,
        output_dir=args.output,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpus=gpus,
        mode=args.mode,
        stride=args.stride,
        perspective=args.perspective,
        partition=args.partition,
        num_partitions=args.num_partitions,
    )


def run_classify(args):
    from .config import get_config

    config = get_config()
    environment = os.environ.copy()
    environment["MASTER_ADDR"] = "127.0.0.1"
    environment["MASTER_PORT"] = str(args.master_port or _free_port(11325, 11425))
    config_file = args.config or str(config.config_root / "classification" / "eval" / "vitl" / "manipulation.yaml")
    command = [
        sys.executable,
        "-m",
        f"{MODULE_ROOT}.classification_cli",
        "--fname",
        config_file,
        "--lam",
        args.model,
        "--dataset",
        args.dataset,
        "--dim",
        str(args.dim),
        "--classes",
        str(args.classes),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--loader_timeout",
        str(args.loader_timeout),
        "--devices",
        *(gpu if gpu == "cpu" else f"cuda:{gpu}" for gpu in _parse_gpu_ids(args.gpus)),
    ]
    subprocess.run(command, cwd=config.project_root, env=environment, check=True)


def _relative_stats_path(action_data_root, dataset):
    root = os.path.normpath(action_data_root)
    if os.path.basename(root) != "regression_relative":
        root = os.path.join(root, "regression_relative")
    dataset = dataset.lower()
    subdir = REGRESSION_DATA_SUBDIR.get(dataset, dataset)
    return os.path.join(root, subdir, f"relative_action_stats_{dataset}.json")


def run_regress(args):
    from .config import get_config
    from .paths import get_data_root

    config = get_config()
    dataset = args.dataset.lower()
    if dataset in SPLIT_DATASETS:
        train_csv = config.data_dir / f"seen_train_la_{dataset}_{args.stride}_{args.model}.csv"
        val_csv = config.data_dir / f"seen_val_la_{dataset}_{args.stride}_{args.model}.csv"
        unseen_csv = config.data_dir / f"unseen_la_{dataset}_{args.stride}_{args.model}.csv"
    else:
        train_csv = config.data_dir / f"train_la_{dataset}_{args.stride}_{args.model}.csv"
        val_csv = config.data_dir / f"val_la_{dataset}_{args.stride}_{args.model}.csv"
        unseen_csv = None
    if args.val_unseen_csv:
        unseen_csv = Path(args.val_unseen_csv)

    stats_json = args.global_stats_json
    if not stats_json and args.action_mode == "relative":
        action_root = args.action_data_root or os.environ.get("DATA_DIR")
        if action_root:
            stats_json = _relative_stats_path(action_root, dataset)
    if not stats_json and dataset in STATS_JSON_NAME:
        data_root = get_data_root(dataset, "seen_train")
        if data_root:
            stats_json = os.path.join(data_root, STATS_JSON_NAME[dataset])

    run_name = f"{dataset}_{args.stride}_{args.model}_{args.model_type}_{args.action_mode}"
    save_dir = config.log_dir / "regression" / "logs" / run_name
    environment = os.environ.copy()
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        visible_devices = "0,1,2,3,4,5,6,7"
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_machines=1",
        f"--mixed_precision={args.mixed_precision}",
        "--dynamo_backend=no",
        f"--num_processes={len(visible_devices.split(','))}",
        f"--main_process_port={_free_port(29500, 29600)}",
        "--module",
        f"{MODULE_ROOT}.regression",
        "--model_type",
        args.model_type,
        "--train_csv",
        str(train_csv),
        "--val_csv",
        str(val_csv),
        "--save_dir",
        str(save_dir),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--dataset",
        dataset,
        "--latent_action_model",
        args.model,
        "--stride",
        str(args.stride),
        "--action_mode",
        args.action_mode,
    ]
    for flag, value in (
        ("--action_data_root", args.action_data_root),
        ("--val_unseen_csv", unseen_csv),
        ("--global_stats_json", stats_json),
    ):
        if value:
            command.extend((flag, str(value)))
    subprocess.run(command, cwd=config.project_root, env=environment, check=True)


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
    {"extract": run_extract, "classify": run_classify, "regress": run_regress}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
