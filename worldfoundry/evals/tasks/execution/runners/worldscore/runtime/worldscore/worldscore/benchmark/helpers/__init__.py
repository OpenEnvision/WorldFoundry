import os

from omegaconf import OmegaConf

from worldfoundry.evals.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evals.tasks.execution.runners.worldscore.runtime.worldscore.worldscore.benchmark.helpers.dataloaders import (
    dataloader_general,
)


def config_path(*relative):
    root = os.environ.get("WORLDSCORE_CONFIG_ROOT")
    if root:
        return os.path.join(os.path.expanduser(root), *relative)
    return str(bundled_benchmark_asset("worldscore", "config", *relative))


def GetDataloader(visual_movement, json_file=None, noise=False, noise_type="simple"):
    if json_file is None:
        json_file = f"{visual_movement}.json"
    dataset_root = os.getenv("DATA_PATH")
    json_path = os.path.join(dataset_root, "WorldScore-Dataset", visual_movement, json_file)

    config = OmegaConf.load(config_path("base_config.yaml"))

    config.json_path = json_path
    config.visual_movement = visual_movement
    config.noise = noise
    config.noise_type = noise_type
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    loader = dataloader_general(config)

    return loader.data
