# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "VideoMAEv2"))

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import utils
from dataset import build_dataset

# isort: split
from bench_utils.cas_utils import final_merge, final_test


def get_args():
    parser = argparse.ArgumentParser(
        "VideoMAE fine-tuning and evaluation script for action classification", add_help=False
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=100, type=int)

    # Model parameters
    parser.add_argument(
        "--model", default="vit_base_patch16_224", type=str, metavar="MODEL", help="Name of model to train"
    )
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument("--with_checkpoint", action="store_true", default=False)

    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)")
    parser.add_argument(
        "--attn_drop_rate", type=float, default=0.0, metavar="PCT", help="Attention dropout rate (default: 0.)"
    )
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)")
    parser.add_argument(
        "--head_drop_rate", type=float, default=0.0, metavar="PCT", help="cls head dropout rate (default: 0.)"
    )

    parser.add_argument("--disable_eval_during_finetuning", action="store_true", default=False)

    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument("--model_ema_force_cpu", action="store_true", default=False, help="")

    # Optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw"')
    parser.add_argument(
        "--opt_eps", default=1e-8, type=float, metavar="EPSILON", help="Optimizer Epsilon (default: 1e-8)"
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)"
    )
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)")
    parser.add_argument("--layer_decay", type=float, default=0.75)

    parser.add_argument(
        "--warmup_lr", type=float, default=1e-8, metavar="LR", help="warmup learning rate (default: 1e-6)"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR, if scheduler supports"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter", type=float, default=0.4, metavar="PCT", help="Color jitter factor (default: 0.4)"
    )
    parser.add_argument("--num_sample", type=int, default=2, help="Repeated_aug (default: 2)")
    (
        parser.add_argument(
            "--aa",
            type=str,
            default="rand-m7-n4-mstd0.5-inc1",
            metavar="NAME",
            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)',
        ),
    )
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--short_side_size", type=int, default=224)
    parser.add_argument("--test_num_segment", type=int, default=10)
    parser.add_argument("--test_num_crop", type=int, default=3)

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument(
        "--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split"
    )

    # * Mixup params
    parser.add_argument("--mixup", type=float, default=0.8, help="mixup alpha, mixup enabled if > 0.")
    parser.add_argument("--cutmix", type=float, default=1.0, help="cutmix alpha, cutmix enabled if > 0.")
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)
    parser.add_argument("--init_scale", default=0.001, type=float)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")

    # Dataset parameters
    parser.add_argument("--meta_info_path", default="/your/data/path/", type=str, help="dataset path")
    parser.add_argument(
        "--data_set",
        default="Kinetics-400",
        choices=[
            "Kinetics-400",
            "Kinetics-600",
            "Kinetics-700",
            "SSV2",
            "UCF101",
            "HMDB51",
            "Diving48",
            "Kinetics-710",
            "MIT",
            "Commonsense-Adherence",
        ],
        type=str,
        help="dataset",
    )
    parser.add_argument("--data_path", default="/your/data/path/", type=str, help="dataset path")
    parser.add_argument("--data_root", default="", type=str, help="dataset path root")
    parser.add_argument("--eval_data_path", default=None, type=str, help="dataset path for evaluation")
    parser.add_argument("--nb_classes", default=400, type=int, help="number of the classification types")
    parser.add_argument("--imagenet_default_mean_and_std", default=True, action="store_true")
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--sparse_sample", default=False, action="store_true")
    parser.add_argument("--fname_tmpl", default="img_{:05}.jpg", type=str, help="filename_tmpl for rawframe dataset")
    parser.add_argument("--start_idx", default=1, type=int, help="start_idx for rwaframe dataset")

    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--validation", action="store_true", help="Perform validation only")
    parser.add_argument("--dist_eval", action="store_true", default=False, help="Enabling distributed evaluation")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--enable_deepspeed", action="store_true", default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        parser = deepspeed.add_config_arguments(parser)
        ds_init = deepspeed.initialize
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # build csv from meta data
    with open(args.meta_info_path) as f:
        data = json.load(f)

    with open(f"{args.data_path}/test.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        for item in data:
            writer.writerow([item["filepath"], 0])

    print(f"CSV file has been created: {args.data_path}/test.csv")

    # build test dataset
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
        )
    else:
        data_loader_test = None

    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
    )

    patch_size = model.patch_embed.patch_size
    print(f"Patch size = {patch_size!s}")

    args.window_size = (
        args.num_frames // args.tubelet_size,
        args.input_size // patch_size[0],
        args.input_size // patch_size[1],
    )

    args.patch_size = patch_size

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = {model_without_ddp!s}")
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * num_tasks
    print(f"Batch size = {total_batch_size:d}")
    print(f"Update frequent = {args.update_freq:d}")

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + ".txt")
        _test_stats = final_test(data_loader_test, model, device, preds_file)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            results = final_merge(args.output_dir, num_tasks, args.meta_info_path)
            print("computing successfully")
            with open(args.meta_info_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Writing updates to {args.meta_info_path}")
        exit(0)


if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
