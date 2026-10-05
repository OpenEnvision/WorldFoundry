import argparse
from pathlib import Path
from typing import Tuple

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from eval_inputs import BatchSampler, collate_segments_to_batch, Dataset
from envs import make_atari_env, WorldModelEnv
from game import ActionNames, DatasetEnv, Game, get_keymap_and_action_names, Keymap, NamedEnv, PlayEnv
from utils import get_path_agent_ckpt, prompt_atari_game

from worldfoundry.core.io.paths import checkpoint_root_path, resolve_data_path


OmegaConf.register_new_resolver("eval", eval)


def download(filename: str, pretrained_dir: str | None = None) -> Path:
    """Resolve one released DIAMOND asset without runtime network access."""

    if not pretrained_dir:
        raise FileNotFoundError(
            "DIAMOND pretrained inference requires --pretrained-dir pointing to a "
            "local eloialonso/diamond snapshot. Runtime downloads are disabled."
        )
    local_path = Path(pretrained_dir).expanduser() / filename
    if not local_path.is_file():
        raise FileNotFoundError(
            f"DIAMOND local pretrained asset is missing: {local_path.resolve()}"
        )
    return local_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained", action="store_true", help="Use a local pretrained world model and agent.")
    parser.add_argument("--pretrained-game", type=str, default=None, help="Atari game id for non-interactive pretrained mode.")
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default=str(checkpoint_root_path("diamond")) if checkpoint_root_path("diamond").is_dir() else None,
        help="Local eloialonso/diamond mirror with atari_100k/config and atari_100k/models.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Local DIAMOND agent/world-model checkpoint.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(resolve_data_path("models", "runtime", "configs", "diamond", "config")),
        help="Directory containing trainer.yaml plus agent/env config groups.",
    )
    parser.add_argument("-d", "--dataset-mode", action="store_true", help="Dataset visualization mode.")
    parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
    parser.add_argument("-n", "--num-steps-initial-collect", type=int, default=1000, help="Num steps initial collect.")
    parser.add_argument("--store-denoising-trajectory", action="store_true", help="Save denoising steps in info.")
    parser.add_argument("--store-original-obs", action="store_true", help="Save original obs (pre resizing) in info.")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate.")
    parser.add_argument("--size", type=int, default=640, help="Window size.")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Run a finite policy-controlled world-model rollout without opening a Pygame window.",
    )
    parser.add_argument("--output-path", type=str, default=None, help="MP4 destination for a headless rollout.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Directory for the short real-environment initialization dataset.",
    )
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.dataset_mode:
        if not Path("dataset").is_dir():
            print(f"Error: {str(Path('dataset').absolute())} not found, cannot use dataset mode.")
            return False
        if Path(".git").is_dir():
            print("Error: cannot run dataset mode the root of the repository.")
            return False
        if args.pretrained or args.record:
            print("Warning: dataset mode, ignoring --pretrained and --record")
    else:
        if not args.record and (args.store_denoising_trajectory or args.store_original_obs):
            print("Warning: not in recording mode, ignoring --store* options")
        if args.headless_steps < 0:
            print("Error: --headless-steps must be non-negative.")
            return False
        if args.headless_steps and not args.output_path:
            print("Error: --output-path is required with --headless-steps.")
            return False
    return True


def prepare_dataset_mode(cfg: DictConfig) -> Tuple[DatasetEnv, Keymap, ActionNames]:
    datasets = []
    for p in Path("dataset").iterdir():
        if p.is_dir():
            d = Dataset(p, p.stem)
            d.load_from_default_path()
            datasets.append(d)
    _, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    dataset_env = DatasetEnv(datasets, env_action_names)
    keymap, _ = get_keymap_and_action_names("dataset_mode")
    return dataset_env, keymap


def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace) -> Tuple[PlayEnv, Keymap, ActionNames]:
    # Checkpoint
    if args.pretrained:
        name = args.pretrained_game or prompt_atari_game()
        path_ckpt = download(f"atari_100k/models/{name}.pt", args.pretrained_dir)

        # Override config
        cfg.agent = OmegaConf.load(download("atari_100k/config/agent/default.yaml", args.pretrained_dir))
        cfg.env = OmegaConf.load(download("atari_100k/config/env/atari.yaml", args.pretrained_dir))
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
        cfg.world_model_env.horizon = 50
    elif args.checkpoint:
        path_ckpt = Path(args.checkpoint).expanduser().resolve()
    else:
        path_ckpt = get_path_agent_ckpt("checkpoints", epoch=-1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Real envs
    train_env = make_atari_env(num_envs=1, device=device, **cfg.env.train)
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    agent.load(path_ckpt)

    # Collect for imagination's initialization
    n = args.num_steps_initial_collect
    dataset_root = Path(args.dataset_dir).expanduser() if args.dataset_dir else Path("dataset")
    dataset = Dataset(dataset_root / f"{path_ckpt.stem}_{n}")
    dataset.load_from_default_path()
    if len(dataset) == 0:
        print(f"Collecting {n} steps in real environment for world model initialization.")
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
        collector.send(NumToCollect(steps=n))
        dataset.save_to_default_path()

    # World model environment
    bs = BatchSampler(dataset, 0, 1, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)

    envs = [
        NamedEnv("wm", wm_env),
        NamedEnv("test", test_env),
        NamedEnv("train", train_env),
    ]

    env_keymap, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    play_env = PlayEnv(
        agent,
        envs,
        env_action_names,
        env_keymap,
        args.record,
        args.store_denoising_trajectory,
        args.store_original_obs,
    )

    return play_env, env_keymap


def run_headless(play_env: PlayEnv, output_path: str, *, num_steps: int, fps: int) -> None:
    """Run a finite imagined rollout and encode its observations as MP4."""

    import cv2
    import numpy as np

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    def to_rgb(observation: torch.Tensor) -> np.ndarray:
        if observation.ndim != 4 or observation.size(0) != 1:
            raise ValueError(f"Expected one BCHW observation, got {tuple(observation.shape)}")
        return (
            observation[0]
            .detach()
            .clamp(-1, 1)
            .add(1)
            .div(2)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

    observation, _ = play_env.reset()
    frames = [to_rgb(observation)]
    for _ in range(num_steps):
        next_observation, _, end, trunc, _ = play_env.step(0)
        frames.append(to_rgb(next_observation))
        if bool(end) or bool(trunc):
            observation, _ = play_env.reset()
        else:
            observation = next_observation

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {destination}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(f"Saved {len(frames)} DIAMOND world-model frames to {destination}")


@torch.no_grad()
def main():
    args = parse_args()
    ok = check_args(args)
    if not ok:
        return

    config_dir = str(Path(args.config_dir).expanduser().resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="trainer")

    env, keymap = prepare_dataset_mode(cfg) if args.dataset_mode else prepare_play_mode(cfg, args)
    if args.headless_steps:
        run_headless(env, args.output_path, num_steps=args.headless_steps, fps=args.fps)
    else:
        size = (args.size // cfg.env.train.size) * cfg.env.train.size  # window size
        game = Game(env, keymap, (size, size), fps=args.fps, verbose=not args.no_header)
        game.run()


if __name__ == "__main__":
    main()
