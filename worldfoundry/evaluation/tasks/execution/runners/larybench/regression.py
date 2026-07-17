import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_constant_schedule_with_warmup
from .paths import get_data_root, resolve_la_path
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .regression_dit import DiT

DIM_LABELS = {
    "calvin": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "agibotbeta": [
        "l_x",
        "l_y",
        "l_z",
        "r_x",
        "r_y",
        "r_z",
        "l_qx",
        "l_qy",
        "l_qz",
        "l_qw",
        "r_qx",
        "r_qy",
        "r_qz",
        "r_qw",
        "r_g",
        "l_g",
    ],
    "robocoin": [
        "l_x",
        "l_y",
        "l_z",
        "l_roll",
        "l_pitch",
        "l_yaw",
        "r_x",
        "r_y",
        "r_z",
        "r_roll",
        "r_pitch",
        "r_yaw",
    ],
}
GROUP_INDICES = {
    "calvin": {
        "position": [0, 1, 2],
        "orientation": [3, 4, 5],
        "gripper": [6],
    },
    "agibotbeta": {
        "position": [0, 1, 2, 3, 4, 5],
        "orientation": [6, 7, 8, 9, 10, 11, 12, 13],
        "gripper": [14, 15],
    },
    "robocoin": {
        "position": [0, 1, 2, 6, 7, 8],
        "orientation": [3, 4, 5, 9, 10, 11],
    },
}
REGRESSION_DATA_SUBDIR = {
    "calvin": "calvin",
    "vlabench": "vlabench",
    "vlabench_15": "vlabench",
    "vlabench_30": "vlabench",
    "agibotbeta": "agibot_45",
    "robocoin": "robocoin_10",
}
REGRESSION_SPLIT_DIR = {
    ("calvin", "train"): "train_stride5",
    ("calvin", "val"): "val_stride5",
}
ACTION_SUBDIR_DATASETS = {"calvin", "agibotbeta", "robocoin"}
CALVIN_MEAN = np.array([0.03993005, -0.1113833, 0.50033228, 1.04580053, -0.08165425, 1.58390577, -0.08441296])
CALVIN_STD = np.array([0.14403107, 0.09919957, 0.05518382, 2.89455128, 0.13053949, 0.57474015, 0.99643086])
DEFAULT_MEAN = np.array([0.02959172, 0.39965126, 0.25788001, 1.0136418, -0.38189239, -0.13961033, 0.54842541])
DEFAULT_STD = np.array([0.15966601, 0.13813421, 0.1362726, 2.70449661, 0.63210694, 1.72885403, 0.49764945])


def _pandas():
    import pandas

    return pandas


def get_dim_labels(dataset_name):
    return DIM_LABELS.get(dataset_name.lower(), DIM_LABELS["calvin"])


def get_action_dim(dataset_name):
    return len(get_dim_labels(dataset_name))


def get_action_steps(chunk_size, action_mode):
    return 1 if action_mode == "relative" else chunk_size


def get_regression_root(action_data_root, action_mode="absolute"):
    root = action_data_root or os.environ.get("DATA_DIR")
    if not root:
        return None
    root = os.path.normpath(root)
    expected_leaf = "regression_relative" if action_mode == "relative" else "regression"
    return root if os.path.basename(root) == expected_leaf else os.path.join(root, expected_leaf)


def get_regression_data_subdir(dataset_name):
    dataset_name = dataset_name.lower()
    return REGRESSION_DATA_SUBDIR.get(dataset_name, dataset_name)


def get_relative_stats_path(action_data_root, dataset_name):
    regression_root = get_regression_root(action_data_root, "relative")
    if not regression_root:
        return None
    dataset_name = dataset_name.lower()
    stats_dir = os.path.join(regression_root, get_regression_data_subdir(dataset_name))
    dataset_specific = os.path.join(stats_dir, f"relative_action_stats_{dataset_name}.json")
    default = os.path.join(stats_dir, "relative_action_stats.json")
    return default if not os.path.exists(dataset_specific) and os.path.exists(default) else dataset_specific


def get_action_data_root(action_data_root, dataset_name, split, action_mode="absolute"):
    if action_mode != "relative" and not action_data_root:
        return None
    regression_root = get_regression_root(action_data_root, action_mode)
    if not regression_root:
        return None
    dataset_name = dataset_name.lower()
    root = os.path.join(regression_root, get_regression_data_subdir(dataset_name))
    split_dir = REGRESSION_SPLIT_DIR.get((dataset_name, split.lower()))
    return os.path.join(root, split_dir) if split_dir else root


def resolve_under_root(root, subdir, value):
    pd = _pandas()
    if value is None or pd.isna(value):
        return value
    path = str(value)
    if os.path.isabs(path):
        return path
    if subdir and subdir in Path(path).parts:
        return os.path.join(root, path)
    return os.path.join(root, subdir, path) if subdir else os.path.join(root, path)


def to_action_target(action, action_dim, action_mode):
    flat = np.asarray(action).reshape(-1)
    if action_mode != "relative" or flat.size == action_dim:
        return flat.astype(np.float32, copy=False)
    if flat.size % action_dim:
        raise ValueError(f"Action size {flat.size} is not divisible by action_dim={action_dim}.")
    sequence = flat.reshape(-1, action_dim)
    return (sequence[-1] - sequence[0]).astype(np.float32, copy=False)


def compute_per_dim_mse(prediction, target, action_steps, labels):
    prediction = prediction.view(-1, action_steps, len(labels))
    target = target.view(-1, action_steps, len(labels))
    values = ((prediction - target) ** 2).mean(dim=(0, 1))
    return {label: values[index].item() for index, label in enumerate(labels)}


def compute_group_mse(prediction, target, action_steps, dataset_name):
    dataset_name = dataset_name.lower()
    groups = GROUP_INDICES.get(dataset_name)
    if not groups:
        return {}
    action_dim = get_action_dim(dataset_name)
    squared_error = (prediction.view(-1, action_steps, action_dim) - target.view(-1, action_steps, action_dim)) ** 2
    return {f"mse_{name}": squared_error[:, :, indices].mean().item() for name, indices in groups.items()}


def _split_from_csv(csv_path):
    name = Path(csv_path).name.lower()
    for split in ("seen_train", "seen_val", "unseen", "train", "val"):
        if split in name:
            return split
    return "train"


class ActionExpertDataset(Dataset):
    def __init__(
        self,
        csv_path,
        dataset_name,
        chunk_size,
        global_stats_json=None,
        action_mean=None,
        action_std=None,
        action_mode="absolute",
        action_data_root=None,
        latent_action_model="dinov2",
    ):
        pd = _pandas()
        self.data = pd.read_csv(csv_path)
        self.dataset_name = dataset_name.lower()
        self.action_dim = get_action_dim(self.dataset_name)
        self.action_mode = action_mode
        self.action_steps = get_action_steps(chunk_size, action_mode)
        self.action_mean = action_mean
        self.action_std = action_std
        self.robot_stats = self._load_robot_stats(global_stats_json)

        required = {"la_path", "action"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        split = _split_from_csv(csv_path)
        data_root = get_data_root(self.dataset_name, split)
        action_root = get_action_data_root(action_data_root, self.dataset_name, split, action_mode)
        root = action_root or data_root
        if root:
            subdir = "actions" if self.dataset_name in ACTION_SUBDIR_DATASETS else ""
            self.data["action"] = self.data["action"].apply(lambda value: resolve_under_root(root, subdir, value))
        self.data["la_path"] = self.data["la_path"].apply(
            lambda value: (
                resolve_la_path(str(value), self.dataset_name, split, latent_action_model) if pd.notna(value) else value
            )
        )

    @staticmethod
    def _load_robot_stats(path):
        if not path or not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        stats_by_robot = payload.get("robot_stats", payload)
        result = {}
        for robot_type, stats in stats_by_robot.items():
            if not isinstance(stats, dict) or not {"mean", "std"} <= stats.keys():
                continue
            mean = np.asarray(stats["mean"])
            std = np.where(np.asarray(stats["std"]) < 1e-6, 1.0, stats["std"])
            result[str(robot_type)] = (mean, std)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with np.load(row["la_path"]) as latent_data:
            latent_action = np.asarray(latent_data["tokens"]).reshape(-1)
        action = to_action_target(np.load(row["action"]), self.action_dim, self.action_mode)

        if self.dataset_name in {"agibotbeta", "robocoin"}:
            mean, std = self.robot_stats.get(
                str(row["robot_type"]),
                (np.zeros(self.action_dim), np.ones(self.action_dim)),
            )
        else:
            mean, std = self.action_mean, self.action_std
        normalized = (action - np.tile(mean, self.action_steps)) / np.tile(std, self.action_steps)
        return (
            torch.from_numpy(latent_action).float(),
            torch.from_numpy(normalized).float(),
        )


class MLPResNetBlock(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.ReLU())

    def forward(self, value):
        return self.layers(value) + value


class ActionExpertMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim=7,
        hidden_dim=4096,
        num_blocks=2,
        action_steps=5,
    ):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.blocks = nn.ModuleList(MLPResNetBlock(hidden_dim) for _ in range(num_blocks))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim * action_steps))

    def forward(self, latent_action):
        value = self.input(latent_action)
        for block in self.blocks:
            value = block(value)
        return self.output(value)

    @staticmethod
    def loss(prediction, target):
        return F.huber_loss(prediction, target, delta=1.0)


class ActionExpertDiT(nn.Module):
    def __init__(
        self,
        latent_dim,
        action_dim=7,
        action_steps=5,
        hidden_size=512,
        depth=6,
        num_heads=8,
    ):
        super().__init__()
        self.action_steps = action_steps
        self.action_dim = action_dim
        self.dit = DiT(
            in_channels=action_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            token_size=latent_dim,
            future_action_window_size=action_steps,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def forward(self, latent_action, noisy_actions, timesteps):
        return self.dit(noisy_actions, timesteps, latent_action.unsqueeze(1))

    def loss(self, latent_action, action_target):
        batch_size = latent_action.shape[0]
        action_sequence = action_target.view(batch_size, self.action_steps, self.action_dim)
        timesteps = torch.randint(
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latent_action.device,
        )
        noise = torch.randn_like(action_sequence)
        noisy_actions = self.noise_scheduler.add_noise(action_sequence, noise, timesteps)
        return F.huber_loss(self(latent_action, noisy_actions, timesteps), noise, delta=1.0)

    @torch.no_grad()
    def sample(self, latent_action):
        batch_size = latent_action.shape[0]
        action_sequence = torch.randn(
            (batch_size, self.action_steps, self.action_dim),
            device=latent_action.device,
        )
        self.noise_scheduler.set_timesteps(500)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.dit(
                action_sequence,
                timestep.unsqueeze(0).to(latent_action.device),
                latent_action.unsqueeze(1),
            )
            action_sequence = self.noise_scheduler.step(prediction, timestep, action_sequence).prev_sample
        return action_sequence.view(batch_size, -1)


def train_one_epoch(model, loader, optimizer, scheduler, accelerator, epoch, model_type):
    model.train()
    unwrapped = accelerator.unwrap_model(model)
    total_loss = 0.0
    steps = 0
    progress = tqdm(
        loader,
        desc=f"Train Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )
    for latent_action, action in progress:
        optimizer.zero_grad(set_to_none=True)
        loss = (
            unwrapped.loss(latent_action, action)
            if model_type == "dit"
            else unwrapped.loss(model(latent_action), action)
        )
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        value = accelerator.gather(loss.detach()).mean().item()
        total_loss += value
        steps += 1
        progress.set_postfix(loss=f"{value:.4f}")
    return total_loss / max(1, steps)


def evaluate(
    model,
    loader,
    accelerator,
    model_type,
    action_steps,
    dim_labels,
    dataset_name,
    prefix,
):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0.0
    steps = 0
    unwrapped = accelerator.unwrap_model(model)
    with torch.no_grad():
        for latent_action, action in tqdm(
            loader,
            desc=f"Evaluating {prefix}",
            disable=not accelerator.is_local_main_process,
        ):
            if model_type == "dit":
                loss = unwrapped.loss(latent_action, action)
                prediction = unwrapped.sample(latent_action)
            else:
                prediction = model(latent_action)
                loss = unwrapped.loss(prediction, action)
            total_loss += accelerator.gather(loss).mean().item()
            steps += 1
            predictions.append(accelerator.gather_for_metrics(prediction))
            targets.append(accelerator.gather_for_metrics(action))

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return (
        total_loss / max(1, steps),
        F.mse_loss(prediction, target).item(),
        compute_per_dim_mse(prediction, target, action_steps, dim_labels),
        compute_group_mse(prediction, target, action_steps, dataset_name),
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--val_unseen_csv")
    parser.add_argument("--global_stats_json")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dataset", default="calvin")
    parser.add_argument("--latent_action_model", required=True)
    parser.add_argument("--model_type", choices=("mlp", "dit"), default="mlp")
    parser.add_argument("--action_mode", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--action_data_root")
    parser.add_argument("--dit_hidden_size", type=int, default=512)
    parser.add_argument("--dit_depth", type=int, default=6)
    return parser.parse_args()


def _action_stats(args, global_stats_json):
    dataset = args.dataset.lower()
    if args.action_mode == "relative":
        if not global_stats_json or not os.path.exists(global_stats_json):
            raise FileNotFoundError(
                "Relative action mode requires relative-action statistics. "
                f"Expected: {global_stats_json}. Pass --global_stats_json explicitly."
            )
        if dataset in {"agibotbeta", "robocoin"}:
            return None, None
        with open(global_stats_json, encoding="utf-8") as handle:
            stats = json.load(handle)
        mean = np.asarray(stats["mean"])
        std = np.where(np.asarray(stats["std"]) < 1e-6, 1.0, stats["std"])
        return mean, std
    if dataset in {"agibotbeta", "robocoin"}:
        return None, None
    return (CALVIN_MEAN, CALVIN_STD) if dataset == "calvin" else (DEFAULT_MEAN, DEFAULT_STD)


def _make_dataset(args, csv_path, global_stats_json, action_mean, action_std):
    return ActionExpertDataset(
        csv_path,
        args.dataset,
        args.stride,
        global_stats_json,
        action_mean,
        action_std,
        action_mode=args.action_mode,
        action_data_root=args.action_data_root,
        latent_action_model=args.latent_action_model,
    )


def _make_loader(dataset, args, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def main():
    args = _parse_args()
    args.dataset = args.dataset.lower()
    global_stats_json = args.global_stats_json
    if args.action_mode == "relative" and not global_stats_json:
        global_stats_json = get_relative_stats_path(args.action_data_root, args.dataset)

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    accelerator = Accelerator(project_dir=args.save_dir)
    if accelerator.is_local_main_process:
        os.makedirs(args.save_dir, exist_ok=True)

    action_dim = get_action_dim(args.dataset)
    action_steps = get_action_steps(args.stride, args.action_mode)
    dim_labels = get_dim_labels(args.dataset)
    action_mean, action_std = _action_stats(args, global_stats_json)

    train_dataset = _make_dataset(args, args.train_csv, global_stats_json, action_mean, action_std)
    val_dataset = _make_dataset(args, args.val_csv, global_stats_json, action_mean, action_std)
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    val_loader = _make_loader(val_dataset, args)

    unseen_loader = None
    if args.val_unseen_csv and os.path.exists(args.val_unseen_csv):
        unseen_loader = _make_loader(
            _make_dataset(
                args,
                args.val_unseen_csv,
                global_stats_json,
                action_mean,
                action_std,
            ),
            args,
        )

    latent_dim = train_dataset[0][0].numel()
    if args.model_type == "dit":
        model = ActionExpertDiT(
            latent_dim,
            action_dim,
            action_steps,
            args.dit_hidden_size,
            args.dit_depth,
        )
    else:
        model = ActionExpertMLP(latent_dim, action_dim, action_steps=action_steps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.epochs * len(train_loader) * 0.1),
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    if unseen_loader is not None:
        unseen_loader = accelerator.prepare(unseen_loader)

    best_mse = float("inf")
    best_result = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            accelerator,
            epoch,
            args.model_type,
        )
        seen = evaluate(
            model,
            val_loader,
            accelerator,
            args.model_type,
            action_steps,
            dim_labels,
            args.dataset,
            "val_seen",
        )
        unseen = (
            evaluate(
                model,
                unseen_loader,
                accelerator,
                args.model_type,
                action_steps,
                dim_labels,
                args.dataset,
                "val_unseen",
            )
            if unseen_loader is not None
            else None
        )
        if accelerator.is_local_main_process:
            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"Val Seen Loss: {seen[0]:.4f} | Val Seen MSE: {seen[1]:.4f}"
            )
            if seen[1] < best_mse:
                best_mse = seen[1]
                best_result = {
                    "best_epoch": epoch,
                    "train_loss": train_loss,
                    "val_seen_loss": seen[0],
                    "val_seen_mse": seen[1],
                    **{f"val_seen_mse_{key}": value for key, value in seen[2].items()},
                    **{f"val_seen_{key}": value for key, value in seen[3].items()},
                }
                if unseen is not None:
                    best_result.update(
                        {
                            "val_unseen_loss": unseen[0],
                            "val_unseen_mse": unseen[1],
                            **{f"val_unseen_mse_{key}": value for key, value in unseen[2].items()},
                            **{f"val_unseen_{key}": value for key, value in unseen[3].items()},
                        }
                    )

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process and best_result:
        output_path = os.path.join(args.save_dir, "best_result.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(best_result, handle, indent=2)
        print(output_path)


if __name__ == "__main__":
    main()
