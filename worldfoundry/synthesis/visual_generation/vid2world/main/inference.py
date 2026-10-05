from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from transformers import logging as transf_logging

from worldfoundry.core.model_loading.factory import instantiate_from_config


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vid2World inference-only validation runner")
    parser.add_argument("--seed", "-s", type=int, default=20230211)
    parser.add_argument("--name", "-n", type=str, default="worldfoundry_infer")
    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", default=list())
    parser.add_argument("--val", "-v", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--logdir", "-l", type=str, default="logs")
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    return parser


def get_nondefault_trainer_args(args: argparse.Namespace) -> list[str]:
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))


def resolve_checkpoint(config) -> None:
    checkpoint = str(config.model.get("pretrained_checkpoint", "") or "").strip()
    if not checkpoint or checkpoint.startswith("|<"):
        return
    if "params" not in config.model:
        config.model.params = OmegaConf.create()
    params = config.model.params
    if not params.get("ckpt_path"):
        params.ckpt_path = checkpoint


def main() -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    del now
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    del local_rank

    parser = Trainer.add_argparse_args(get_parser())
    args, unknown = parser.parse_known_args()
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    resolve_checkpoint(config)

    output_root = Path(args.logdir).expanduser().resolve() / args.name
    output_root.mkdir(parents=True, exist_ok=True)
    config.model.params.logdir = str(output_root)

    model = instantiate_from_config(config.model)
    if getattr(model, "rescale_betas_zero_snr", False):
        model.register_schedule(
            given_betas=model.given_betas,
            beta_schedule=model.beta_schedule,
            timesteps=model.timesteps,
            linear_start=model.linear_start,
            linear_end=model.linear_end,
            cosine_s=model.cosine_s,
        )

    for key in get_nondefault_trainer_args(args):
        trainer_config[key] = getattr(args, key)
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"

    data = instantiate_from_config(config.data)
    data.setup()

    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(
        trainer_args,
        logger=False,
        callbacks=[],
        num_sanity_val_steps=0,
        sync_batchnorm=False,
        precision=lightning_config.get("precision", 32),
    )
    if global_rank == 0:
        print(f"Vid2World inference validating with WORLD_SIZE={world_size}.")
    trainer.validate(model, data)


if __name__ == "__main__":
    main()
