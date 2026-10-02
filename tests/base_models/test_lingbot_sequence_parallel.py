import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot.model import (
    sinusoidal_embedding_1d as model_timestep_embedding,
)
from worldfoundry.core.attention.causal_rope_sequence_parallel import (
    sinusoidal_embedding_1d as sequence_parallel_timestep_embedding,
)


def test_sequence_parallel_preserves_integer_scheduler_timestep_precision() -> None:
    timesteps = torch.tensor([999, 820, 641, 320], dtype=torch.int64)

    actual = sequence_parallel_timestep_embedding(256, timesteps)
    expected = model_timestep_embedding(256, timesteps)

    assert actual.is_floating_point()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
