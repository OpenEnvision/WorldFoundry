"""Formula gates against Doubiiu/DynamiCrafter commit 859021927d8e0f8eb4d91d16f86711b8c25a2023.

Compared source paths:
  released 512-pixel training config
  released 512-pixel interpolation training config
  lvdm/models/ddpm3d.py
  lvdm/models/utils_diffusion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api.contracts import TrainingBatch  # noqa: E402
from worldfoundry.training.engine.dynamicrafter.sft import _load_native_dynamicrafter  # noqa: E402
from worldfoundry.training.engine.single_device import SingleDeviceTrainEngine  # noqa: E402
from worldfoundry.training.models.dynamicrafter import (  # noqa: E402
    DynamiCrafterTrainAdapter,
    dynamicrafter_objective,
)
from worldfoundry.training.objectives.classic_diffusion import (  # noqa: E402
    dynamic_latent_scale,
    lvdm_linear_beta_schedule,
    rescale_betas_to_zero_terminal_snr,
)
from worldfoundry.training.recipes.spec import TrainingRecipe  # noqa: E402


class _HybridDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.input_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    def forward(self, noisy, timesteps, *, c_concat, c_crossattn, fs):
        condition = c_concat[0].mean() + c_crossattn[0].mean() + fs.float().mean()
        return noisy * self.scale + condition * 0.0 + timesteps.float().mean() * 0.0


class _LoaderDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, noisy, timesteps, *, c_concat, c_crossattn, fs):
        del timesteps, c_concat, c_crossattn, fs
        return noisy * self.weight


def _batch(batch_size: int = 4) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=tuple(f"clip-{index}" for index in range(batch_size)),
        prompts=tuple(f"prompt-{index}" for index in range(batch_size)),
        conditions={
            "clean_latents": torch.arange(batch_size * 4 * 4 * 2 * 2, dtype=torch.float32).reshape(
                batch_size, 4, 4, 2, 2
            ),
            "text_context": torch.arange(batch_size * 2 * 2, dtype=torch.float32).reshape(batch_size, 2, 2),
            "empty_text_context": torch.full((1, 2, 2), -3.0),
            "image_features_by_frame": torch.randn(batch_size, 4, 1, 2),
            "zero_image_features": torch.zeros(1, 1, 2),
            "fps": 10,
        },
    )


def _adapter(*, interpolation: bool) -> DynamiCrafterTrainAdapter:
    return DynamiCrafterTrainAdapter(
        denoiser=_HybridDenoiser(),
        image_projector=torch.nn.Linear(2, 2),
        interpolation=interpolation,
    )


def test_dynamicrafter_zero_terminal_snr_and_dynamic_scale_match_source() -> None:
    betas = lvdm_linear_beta_schedule(1000, beta_start=0.00085, beta_end=0.012)
    actual = rescale_betas_to_zero_terminal_snr(betas).numpy()
    base = np.linspace(np.sqrt(0.00085), np.sqrt(0.012), 1000, dtype=np.float64) ** 2
    alpha_bar_sqrt = np.sqrt(np.cumprod(1.0 - base))
    first, last = alpha_bar_sqrt[0], alpha_bar_sqrt[-1]
    alpha_bar_sqrt = (alpha_bar_sqrt - last) * first / (first - last)
    alpha_bar = alpha_bar_sqrt**2
    expected_alphas = np.concatenate((alpha_bar[:1], alpha_bar[1:] / alpha_bar[:-1]))
    np.testing.assert_allclose(actual, 1.0 - expected_alphas, rtol=1.0e-12, atol=1.0e-12)
    assert torch.cumprod(1.0 - torch.from_numpy(actual), dim=0)[-1].item() == 0.0

    reference = torch.zeros(3, 4, 4, 2, 2)
    scales = dynamic_latent_scale(
        torch.tensor([0, 399, 900]),
        reference,
        final_scale=0.7,
        transition_steps=400,
    ).flatten()
    torch.testing.assert_close(scales, torch.tensor([1.0, 0.7, 0.7]))


def test_dynamicrafter_interpolation_and_joint_dropout_match_source() -> None:
    adapter = _adapter(interpolation=True)
    prepared = adapter.prepare_batch(_batch())
    generator = torch.Generator().manual_seed(91)
    expected_draws = torch.rand(4, generator=torch.Generator().manual_seed(91))
    conditioning = adapter.build_objective_conditioning(
        prepared,
        torch.zeros(4, dtype=torch.long),
        prepared.clean_latents,
        generator,
    )
    expected_text_drop = expected_draws < 0.10
    expected_image_drop = (expected_draws >= 0.05) & (expected_draws < 0.15)
    text = prepared.conditioning["text_context"]
    empty = prepared.conditioning["empty_text_context"]
    expected_text = torch.where(expected_text_drop[:, None, None], empty, text)
    torch.testing.assert_close(conditioning["text_context"], expected_text)
    torch.testing.assert_close(conditioning["image_drop_mask"], expected_image_drop)
    concat = conditioning["c_concat"]
    clean = prepared.clean_latents
    torch.testing.assert_close(concat[:, :, 0], clean[:, :, 0])
    torch.testing.assert_close(concat[:, :, -1], clean[:, :, -1])
    assert torch.count_nonzero(concat[:, :, 1:-1]) == 0


def test_dynamicrafter_runs_on_generic_video_engine() -> None:
    adapter = _adapter(interpolation=False)
    objective = dynamicrafter_objective(adapter)
    optimizer = torch.optim.AdamW(adapter.trainable_module.parameters(), lr=0.01, weight_decay=0.0)
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer, max_grad_norm=10.0)
    before = adapter.denoiser.scale.detach().clone()
    result = engine.train_step(_batch(batch_size=2), generator=torch.Generator().manual_seed(13))
    assert result.sample_count == 2
    assert result.diagnostics["prediction_type"] == "v_prediction"
    assert not torch.equal(before, adapter.denoiser.scale.detach())


def test_dynamicrafter_fp16_profile_materializes_fp32_master_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.base_models.diffusion_model.models.networks.lvdm.latent_diffusion as lvdm_module
    import worldfoundry.core.model_loading.factory as factory_module
    import worldfoundry.core.model_loading.file as file_module

    recipe = TrainingRecipe.from_file(Path("configs/training/dynamicrafter_512_i2v.yaml"))
    monkeypatch.setattr(lvdm_module, "DiffusionWrapper", lambda *_args, **_kwargs: _LoaderDenoiser())
    monkeypatch.setattr(
        factory_module,
        "instantiate_from_config",
        lambda _config: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        file_module,
        "load_state_dict",
        lambda *_args, **_kwargs: {
            "model.weight": torch.ones(1, dtype=torch.float16),
            "image_proj_model.weight": torch.ones(1, 1, dtype=torch.float16),
            "image_proj_model.bias": torch.zeros(1, dtype=torch.float16),
        },
    )

    adapter = _load_native_dynamicrafter(
        recipe,
        root=Path.cwd(),
        checkpoint=None,
        device=torch.device("cpu"),
    )

    assert recipe.runtime.param_dtype == "float16"
    assert all(parameter.dtype is torch.float32 for parameter in adapter.trainable_module.parameters())
