"""Small compatibility fixes needed before importing the JAX/Flax runtime."""

from __future__ import annotations


def install_jax_compatibility() -> None:
    """Bridge APIs used by the released Octo checkpoint environment."""
    try:
        import jax
        from jax._src import config as jax_config

        if not hasattr(jax.config, "define_bool_state") and hasattr(jax_config, "define_bool_state"):
            jax.config.define_bool_state = jax_config.define_bool_state
        if not hasattr(jax.random, "KeyArray"):
            jax.random.KeyArray = jax.Array
    except (ImportError, AttributeError):
        return


__all__ = ["install_jax_compatibility"]
