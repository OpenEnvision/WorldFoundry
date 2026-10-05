import hydra
from absl import logging

from src.utils.config import instantiate_from_config, resolve_device_paths
from src.utils.jax import init_jax_distributed
from worldfoundry.core.io.paths import package_data_path


@hydra.main(
    config_path=str(package_data_path('models', 'runtime', 'configs', 'solaris')), config_name="inference", version_base=None
)  # no version to avoid warnings in cli
def main(cfg):
    # Hydra changes cwd, so resolve relative paths before anything else.
    resolve_device_paths(cfg)

    # Enable JAX compilation cache
    # NOTE: we should import jax cache AFTER setting up the environment variables.
    import jax

    from src.utils.jax import setup_jax_cache

    if cfg.enable_jax_cache:
        setup_jax_cache(cfg.device.jax_cache_dir)

    init_jax_distributed(cfg)

    if not (jax.process_index() == 0):  # not first process
        logging.set_verbosity(logging.ERROR)  # disable info/warning

    inference = instantiate_from_config(cfg.runner)
    inference.run()


if __name__ == "__main__":
    main()
