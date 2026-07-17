import gc
import json
import logging
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from .classification_data import (
    LatentActionVideoDataset,
    make_latent_action_dataloader,
)
from .classification_model import FeatureEvaluator
from torch import nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


def _clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _gather_predictions(predictions, labels, world_size, rank, device):
    local_size = torch.tensor([len(predictions)], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    sizes = [int(size.item()) for size in all_sizes]
    padded_size = max(1, max(sizes))

    def gather(values):
        source = torch.as_tensor(values, dtype=torch.long, device=device)
        padded = torch.full((padded_size,), -1, dtype=torch.long, device=device)
        padded[: len(source)] = source
        output = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(output, padded)
        return output

    all_predictions = gather(predictions)
    all_labels = gather(labels)
    if rank != 0:
        return None, None
    return (
        [value for rank_values, size in zip(all_predictions, sizes) for value in rank_values[:size].cpu().tolist()],
        [value for rank_values, size in zip(all_labels, sizes) for value in rank_values[:size].cpu().tolist()],
    )


def _save_metrics(predictions, labels, class_count, output_dir) -> None:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels), np.asarray(predictions)), 1)

    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = matrix.sum(axis=1)
    total = int(support.sum())

    summary = {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_precision": float(precision.mean()) if class_count else 0.0,
        "macro_recall": float(recall.mean()) if class_count else 0.0,
        "macro_f1": float(f1.mean()) if class_count else 0.0,
        "weighted_f1": float(np.average(f1, weights=support)) if total else 0.0,
        "sample_count": total,
    }
    with open(os.path.join(output_dir, "classification_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, total_steps):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.current_step = 0

    def step(self):
        self.current_step += 1
        for group in self.optimizer.param_groups:
            warmup = group["schedule_warmup_steps"]
            if self.current_step < warmup:
                progress = self.current_step / max(1, warmup)
                value = group["schedule_start_lr"] + progress * (group["schedule_peak_lr"] - group["schedule_start_lr"])
            else:
                progress = (self.current_step - warmup) / max(1, self.total_steps - warmup)
                progress = min(1.0, progress)
                final = group["schedule_final_lr"]
                value = final + (group["schedule_peak_lr"] - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
            group["lr"] = value


class CosineWDSchedule:
    def __init__(self, optimizer, total_steps):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.current_step = 0

    def step(self):
        self.current_step += 1
        progress = min(1.0, self.current_step / self.total_steps)
        for group in self.optimizer.param_groups:
            start = group["schedule_start_wd"]
            final = group["schedule_final_wd"]
            group["weight_decay"] = final + (start - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _run_epoch(
    device,
    models,
    data_loader,
    training=False,
    optimizers=None,
    schedulers=None,
    use_bfloat16=False,
    collect_predictions=False,
):
    for model in models:
        model.train(training)

    correct = [0.0] * len(models)
    losses = [0.0] * len(models)
    sample_counts = [0] * len(models)
    predictions = [[] for _ in models]
    labels = []
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for step, batch in enumerate(data_loader):
        target = batch[0].to(device, non_blocking=True)
        features = batch[1].to(device, non_blocking=True)

        if training:
            for schedule in schedulers or []:
                schedule.step()

        autocast = torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=training and use_bfloat16 and device.type == "cuda",
        )
        with torch.set_grad_enabled(training), autocast:
            output = [model(features) for model in models]
            batch_losses = [criterion(value, target) for value in output]

        finite = torch.tensor(
            [
                torch.isfinite(loss).item() and torch.isfinite(value).all().item()
                for loss, value in zip(batch_losses, output)
            ],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)

        if training:
            for index, (loss, optimizer) in enumerate(zip(batch_losses, optimizers or [])):
                optimizer.zero_grad(set_to_none=True)
                if finite[index]:
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            for index, (value, loss) in enumerate(zip(output, batch_losses)):
                if not finite[index]:
                    continue
                predicted = value.argmax(dim=1)
                count = len(target)
                correct[index] += predicted.eq(target).sum().item()
                losses[index] += loss.item() * count
                sample_counts[index] += count
                if collect_predictions:
                    predictions[index].extend(predicted.cpu().tolist())
            if collect_predictions:
                labels.extend(target.cpu().tolist())

        if step % 10 == 0 and models:
            local_accuracy = [100.0 * value / count if count else 0.0 for value, count in zip(correct, sample_counts)]
            logger.info("step=%d best_local_accuracy=%.3f", step, max(local_accuracy))

    aggregate = []
    for values in zip(correct, losses, sample_counts):
        stats = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(stats)
        aggregate.append(stats.tolist())
    accuracies = [100.0 * value[0] / value[2] if value[2] else 0.0 for value in aggregate]
    average_losses = [value[1] / value[2] if value[2] else float("inf") for value in aggregate]
    best_index = int(np.argmax(accuracies))
    return (
        accuracies[best_index],
        average_losses[best_index],
        predictions[best_index] if collect_predictions else None,
        labels if collect_predictions else None,
    )


def main(args_eval):
    experiment = args_eval["experiment"]
    data_config = experiment["data"]
    optimization = experiment["optimization"]
    classifier_config = experiment["classifier"]
    output_dir = args_eval["folder"]
    os.makedirs(output_dir, exist_ok=True)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_dataset = LatentActionVideoDataset(
        data_config["dataset_train"],
        args_eval["dataset"],
        "train",
        args_eval["lam"],
    )
    val_dataset = LatentActionVideoDataset(
        data_config["dataset_val"],
        args_eval["dataset"],
        "val",
        args_eval["lam"],
        action_to_id=train_dataset.action_to_id,
    )
    class_names = [action for action, _ in sorted(train_dataset.action_to_id.items(), key=lambda item: item[1])]
    if data_config["num_classes"] != len(class_names):
        raise ValueError(
            f"Configured {data_config['num_classes']} classes, but training data has {len(class_names)}: {class_names}"
        )

    loader_kwargs = {
        "batch_size": optimization["batch_size"],
        "world_size": world_size,
        "rank": rank,
        "num_workers": args_eval.get("num_workers", 8),
        "timeout": args_eval.get("loader_timeout", 0),
    }
    train_loader, train_sampler = make_latent_action_dataloader(train_dataset, training=True, **loader_kwargs)
    val_loader, _ = make_latent_action_dataloader(val_dataset, training=False, **loader_kwargs)

    optimizer_configs = optimization["multihead_kwargs"]
    if not optimizer_configs:
        raise ValueError("At least one classifier optimizer configuration is required")
    models = []
    for _ in optimizer_configs:
        model = FeatureEvaluator(
            input_dim=classifier_config["dim"],
            num_heads=classifier_config["num_heads"],
            depth=classifier_config["num_probe_blocks"],
            num_classes=len(class_names),
            use_activation_checkpointing=True,
        ).to(device)
        if device.type == "cuda":
            model = DistributedDataParallel(model, device_ids=[0], static_graph=True)
        else:
            model = DistributedDataParallel(model, static_graph=True)
        models.append(model)
    steps_per_epoch = len(train_loader)
    epoch_count = optimization["num_epochs"]
    optimizers = []
    lr_schedulers = []
    wd_schedulers = []
    for model, config in zip(models, optimizer_configs):
        groups = [
            {
                "params": model.parameters(),
                "schedule_warmup_steps": int(config["warmup"] * steps_per_epoch),
                "schedule_start_lr": config["start_lr"],
                "schedule_peak_lr": config["lr"],
                "schedule_final_lr": config["final_lr"],
                "schedule_start_wd": config["weight_decay"],
                "schedule_final_wd": config["final_weight_decay"],
            }
        ]
        optimizer = torch.optim.AdamW(groups)
        optimizers.append(optimizer)
        total_steps = epoch_count * steps_per_epoch
        lr_schedulers.append(WarmupCosineLRSchedule(optimizer, total_steps))
        wd_schedulers.append(CosineWDSchedule(optimizer, total_steps))

    checkpoint_path = os.path.join(output_dir, "latest.pt")
    start_epoch = 0
    if args_eval.get("resume_checkpoint") and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_epoch = checkpoint["epoch"]
        for model, state in zip(models, checkpoint["classifiers"]):
            model.load_state_dict(state)
        for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)
        for _ in range(start_epoch * steps_per_epoch):
            for schedule in lr_schedulers + wd_schedulers:
                schedule.step()

    for epoch in range(start_epoch, epoch_count):
        train_sampler.set_epoch(epoch)
        _clear_memory()
        train_result = _run_epoch(
            device,
            models,
            train_loader,
            training=True,
            optimizers=optimizers,
            schedulers=lr_schedulers + wd_schedulers,
            use_bfloat16=optimization.get("use_bfloat16", False),
        )
        val_result = _run_epoch(
            device,
            models,
            val_loader,
            collect_predictions=True,
        )
        logger.info(
            "epoch=%d train_accuracy=%.3f train_loss=%.5f val_accuracy=%.3f val_loss=%.5f",
            epoch + 1,
            train_result[0],
            train_result[1],
            val_result[0],
            val_result[1],
        )

        all_predictions, all_labels = _gather_predictions(
            val_result[2],
            val_result[3],
            world_size,
            rank,
            device,
        )
        if rank == 0:
            torch.save(
                {
                    "classifiers": [model.state_dict() for model in models],
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
            _save_metrics(
                all_predictions,
                all_labels,
                len(class_names),
                output_dir,
            )
        dist.barrier()
