# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free tensor contracts for the native Matrix-Game 4 renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers.matrix_game4 import (
    MatrixGame4Denoiser,
    build_matrix_game4_denoiser,
)
from worldfoundry.base_models.diffusion_model.models.initializers.wan.causal_control import WanCausalControlInitializer
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.matrix_game4 import MatrixGame4Model
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.base_models.diffusion_model.runners.autoregressive import AutoregressiveWindowRunner
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowMatchEulerScheduler
from worldfoundry.operators.matrix_game4_operator import MatrixGame4Operator

pytestmark = pytest.mark.unit

SMALL_CONFIG = {
    "in_dim": 148,
    "out_dim": 48,
    "dim": 24,
    "ffn_dim": 48,
    "num_heads": 2,
    "num_layers": 1,
    "text_dim": 16,
    "text_len": 8,
    "freq_dim": 12,
    "local_attn_size": 21,
    "sink_size": 1,
    "max_rope_frames": 128,
}


class _Codec:
    dtype = torch.float32

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.inputs.append(images.clone())
        indices = [0, *range(4, images.shape[2], 4)]
        return F.avg_pool3d(images[:, :, indices], (1, 16, 16)).repeat(1, 16, 1, 1, 1)

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        if request.inputs.get("return_latent"):
            return latents
        frames = [latents[:, :3, :1]]
        frames.extend(
            latents[:, :3, index : index + 1].repeat_interleave(4, dim=2) for index in range(1, latents.shape[2])
        )
        return F.interpolate(torch.cat(frames, dim=2), size=((latents.shape[2] - 1) * 4 + 1, 32, 32))


class _Conditioner:
    def encode(self, request: DiffusionRequest, **kwargs: Any) -> Conditioning:
        return Conditioning(positive={"context": torch.zeros(1, 8, 16, **kwargs)})


def _model() -> MatrixGame4Model:
    torch.manual_seed(31)
    model = MatrixGame4Model(**SMALL_CONFIG).eval()
    torch.nn.init.normal_(model.head.head.weight, std=0.05)
    return model


def _request(frames: int = 100, value: float = 0.25) -> DiffusionRequest:
    return DiffusionRequest(
        prompt="Egypt",
        height=32,
        width=32,
        num_frames=frames,
        sampling=SamplingConfig(num_inference_steps=2, guidance_scale=1.0, seed=7),
        inputs={"video": torch.full((1, 3, frames, 32, 32), value)},
    )


def _runner(tmp_path: Path) -> tuple[AutoregressiveWindowRunner, _Codec]:
    path = tmp_path / "sink.png"
    Image.new("RGB", (80, 40), color=(20, 100, 180)).save(path)
    codec = _Codec()
    runner = AutoregressiveWindowRunner(
        model_id="matrix-game-4-egypt",
        components=RunnerComponents(
            denoiser=MatrixGame4Denoiser(_model()),
            conditioner=_Conditioner(),
            latent_initializer=WanCausalControlInitializer(path),
            latent_encoder=codec,
            decoder=codec,
            scheduler=WanFlowMatchEulerScheduler(shift=5),
        ),
        device="cpu",
        dtype=torch.float32,
        prediction_mode="distilled-x0",
        fixed_timesteps=(1000, 500),
        renoise_dtype=torch.float32,
    )
    return runner, codec


def test_checkpoint_loader_preserves_released_names_and_runs_real_forward(tmp_path: Path) -> None:
    source = _model()
    path = tmp_path / "model.safetensors"
    save_file({f"model.{name}": tensor.contiguous() for name, tensor in source.state_dict().items()}, str(path))
    context = ComponentBuildContext(
        model_id="matrix-game-4-egypt",
        key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(device="cpu", dtype=torch.float32),
        checkpoints={"weights": CheckpointSpec(source=path)},
        component_options={"model_config": SMALL_CONFIG},
    )
    denoiser = build_matrix_game4_denoiser(context)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(denoiser.model.state_dict()[name], value, rtol=0, atol=0)
    cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=3,
        frame_sequence_length=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    pixels = torch.randn(1, 148, 1, 2, 2)
    arguments = dict(
        t=torch.full((1, 1), 1000.0),
        context=torch.randn(1, 8, 16),
        seq_len=1,
        kv_cache=cache["kv_cache"],
        crossattn_cache=cache["crossattn_cache"],
    )
    with torch.no_grad():
        actual = denoiser.model(pixels, **arguments)
    assert actual.shape == (1, 48, 1, 2, 2)
    assert actual.abs().max() > 0
    assert torch.isfinite(actual).all()


def _assert_bf16_x0_grid(device: str) -> None:
    model = _model().to(device=device, dtype=torch.bfloat16)
    torch.nn.init.zeros_(model.head.head.weight)
    torch.nn.init.constant_(model.head.head.bias, 3.0 / 256.0)
    denoiser = MatrixGame4Denoiser(model)
    latents = torch.ones(1, 48, 1, 2, 2, device=device, dtype=torch.bfloat16)
    cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=2,
        frame_sequence_length=1,
        device=latents.device,
        dtype=latents.dtype,
    )
    conditioning = {
        "context": torch.zeros(1, 8, 16, device=device, dtype=latents.dtype),
        "control_latents": torch.zeros_like(latents),
        "kv_cache": cache,
        "current_start": 0,
    }
    for timestep, branch, expected in (
        (1000.0, "positive", 0.98828125),
        (833.3333129882812, "positive", 0.9921875),
        (0.0, "positive-cache-commit", 1.0),
    ):
        with torch.no_grad():
            output = denoiser(
                DenoiserInput(
                    latents=latents,
                    timestep=torch.tensor(timestep, device=device),
                    next_timestep=torch.tensor(0.0, device=device),
                    conditioning=conditioning,
                    step_index=1,
                    total_steps=2,
                    branch=branch,
                )
            )
        torch.testing.assert_close(output.sample, torch.full_like(latents, expected), rtol=0, atol=0)


def test_bf16_x0_uses_released_sigma_grid_and_clean_commits() -> None:
    _assert_bf16_x0_grid("cpu")


@pytest.mark.gpu
def test_cuda_bf16_x0_uses_released_sigma_grid_and_clean_commits() -> None:
    if not torch.cuda.is_available():
        pytest.fail("Matrix-Game 4 CUDA sigma-grid contract requires CUDA")
    _assert_bf16_x0_grid("cuda")


def test_initializer_pads_tail_and_keeps_sink_separate_from_control_history(tmp_path: Path) -> None:
    runner, codec = _runner(tmp_path)
    request = _request(frames=4)
    initializer = runner.components.latent_initializer
    assert isinstance(initializer, WanCausalControlInitializer)
    initialized = initializer.initialize_with_encoder(
        request,
        latent_encoder=codec,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert [value.shape[2] for value in codec.inputs] == [1, 5]
    video = request.inputs["video"]
    assert isinstance(video, torch.Tensor)
    torch.testing.assert_close(codec.inputs[1][:, :, -1], video[:, :, -1])
    assert initialized.artifacts["padded_frames"] == 1
    prefix, control = initialized.conditioning["kv_prefix_latents"], initialized.conditioning["control_latents"]
    assert isinstance(prefix, torch.Tensor) and isinstance(control, torch.Tensor)
    assert prefix.shape[2] == 1
    assert control.shape[2] == 2
    rng = torch.Generator("cpu").manual_seed(8)
    expected = torch.cat([torch.randn(1, 1, 48, 2, 2, generator=rng) for _ in range(2)], dim=1)
    torch.testing.assert_close(initialized.latents, expected.permute(0, 2, 1, 3, 4).bfloat16(), rtol=0, atol=0)


def test_rollout_crosses_window_commits_without_control_and_trims_tail(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    denoiser = runner.components.denoiser
    assert isinstance(denoiser, MatrixGame4Denoiser)
    packed = []
    hook = denoiser.model.patch_embedding.register_forward_pre_hook(
        lambda _module, args: packed.append(args[0].clone())
    )
    try:
        output = runner.run(_request())
    finally:
        hook.remove()
    assert output.sample.shape == (1, 3, 100, 32, 32)
    assert output.latents.shape == (1, 48, 26, 2, 2)
    assert output.metadata["kv_prefix_frames"] == 1
    assert output.artifacts["padded_frames"] == 1
    assert len(packed) == 1 + 26 * 3
    assert torch.count_nonzero(packed[0][:, 48:]) == 0
    for index in range(26):
        positive, intermediate, commit = packed[1 + 3 * index : 4 + 3 * index]
        assert positive[:, 48:96].abs().max() > 0
        torch.testing.assert_close(positive[:, 48:96], intermediate[:, 48:96])
        assert torch.count_nonzero(commit[:, 48:]) == 0
        assert torch.count_nonzero(positive[:, 96:]) == 0


def test_rollout_is_seeded_request_local_and_recovers_after_failure(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    state = torch.random.get_rng_state().clone()
    first = runner.run(_request(frames=5))
    other = runner.run(_request(frames=5, value=-0.75))
    repeated = runner.run(_request(frames=5))
    torch.testing.assert_close(first.latents, repeated.latents, rtol=0, atol=0)
    assert not torch.equal(first.latents, other.latents)
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)

    def fail(_module: Any, _inputs: Any) -> None:
        raise RuntimeError("interrupted render")

    denoiser = runner.components.denoiser
    assert isinstance(denoiser, MatrixGame4Denoiser)
    hook = denoiser.model.patch_embedding.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="interrupted render"):
            runner.run(_request(frames=5))
    finally:
        hook.remove()
    recovered = runner.run(_request(frames=5))
    torch.testing.assert_close(first.latents, recovered.latents, rtol=0, atol=0)


def test_fixed_sink_rope_matches_independent_history_attention_after_eviction() -> None:
    model = _model()
    denoiser = MatrixGame4Denoiser(model)
    cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=30,
        frame_sequence_length=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )["kv_cache"][0]
    module = model.blocks[0].self_attn
    history = []
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for position in range(28):
            x = torch.randn(1, 1, 24, generator=generator)
            shape = (1, 1, 2, 12)
            raw_key = module.norm_k(module.k(x)).reshape(shape)
            value = module.v(x).reshape(shape)
            history.append((raw_key, value))

            def rotate(tensor: torch.Tensor, frame: int) -> torch.Tensor:
                # One spatial token: only the first temporal complex pairs rotate.
                half = tensor.shape[-1] // 2
                temporal = half - 2 * (half // 3)
                angles = model.freqs[0].clone()
                angles[:temporal] = model.freqs[frame, :temporal]
                paired = torch.view_as_complex(tensor.double().reshape(1, 1, 2, -1, 2))
                return torch.view_as_real(paired * angles).flatten(-2).to(tensor)

            retained = [0, *range(max(1, position - 19), position + 1)]
            keys = torch.cat(
                [rotate(history[index][0], max(0, position - 21) if index == 0 else index) for index in retained], dim=1
            )
            values = torch.cat([history[index][1] for index in retained], dim=1)
            query = rotate(module.norm_q(module.q(x)).reshape(shape), position)
            result = F.scaled_dot_product_attention(
                query.bfloat16().float().transpose(1, 2),
                keys.bfloat16().float().transpose(1, 2),
                values.bfloat16().float().transpose(1, 2),
            ).transpose(1, 2)
            expected = module.o(result.flatten(2))
            actual = module(x, torch.tensor([1]), torch.tensor([[1, 1, 1]]), model.freqs, None, cache, position)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            previous = cache["k"].clone()
            rewritten = module(x, torch.tensor([1]), torch.tensor([[1, 1, 1]]), model.freqs, None, cache, position)
            torch.testing.assert_close(actual, rewritten, rtol=0, atol=0)
            torch.testing.assert_close(cache["k"], previous, rtol=0, atol=0)
            assert cache["local_end_index"].item() == min(position + 1, 21)


def test_bf16_rollout_uses_fp32_intermediate_noise(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    runner.dtype = torch.bfloat16
    denoiser = runner.components.denoiser
    assert isinstance(denoiser, MatrixGame4Denoiser)
    model = denoiser.model.to(dtype=torch.bfloat16)
    inputs: list[torch.Tensor] = []
    flows: list[torch.Tensor] = []
    before = model.register_forward_pre_hook(lambda _module, args: inputs.append(args[0].clone()))
    after = model.register_forward_hook(lambda _module, _args, output: flows.append(output.clone()))
    try:
        runner.run(_request(frames=1))
    finally:
        before.remove()
        after.remove()
    initial = inputs[1][:, :48]
    clean = (initial.double() - flows[1].double()).bfloat16()
    sigma = torch.tensor(5 * 0.5 / 3, dtype=torch.float32)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float32)
    expected = ((1 - sigma) * clean.float() + sigma * noise).bfloat16()
    torch.testing.assert_close(inputs[2][:, :48], expected, rtol=0, atol=0)


def test_operator_declares_pixel_semantics_and_preserves_training_resize() -> None:
    source = np.arange(18 * 32 * 3, dtype=np.uint8).reshape(18, 32, 3)
    pixels, fps = MatrixGame4Operator().prepare_video(np.stack([source, source]))
    expected = np.asarray(Image.fromarray(source).resize((864, 486), Image.Resampling.BILINEAR).crop((0, 3, 864, 483)))
    expected = torch.from_numpy(expected.copy()).permute(2, 0, 1).float() * (2 / 255) - 1
    torch.testing.assert_close(pixels[0, :, 0], expected, rtol=0, atol=0)
    assert fps is None
    with pytest.raises(ValueError, match="uint8"):
        MatrixGame4Operator().prepare_video(np.zeros((2, 480, 864, 3), dtype=np.float32))
