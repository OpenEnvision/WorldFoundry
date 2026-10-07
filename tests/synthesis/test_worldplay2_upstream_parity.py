from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.synthesis.visual_generation.worldplay2.conditioning import WorldPlay2Conditioner
from worldfoundry.synthesis.visual_generation.worldplay2.denoiser import WorldPlay2Denoiser
from worldfoundry.synthesis.visual_generation.worldplay2.initializer import WorldPlay2LatentInitializer
from worldfoundry.synthesis.visual_generation.worldplay2.modeling.model import WorldPlay2Model
from worldfoundry.synthesis.visual_generation.worldplay2.runner import CompressedMemoryRunner
from worldfoundry.synthesis.visual_generation.worldplay2.scheduler import FixedPDD4Scheduler

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "worldplay2" / "upstream_cpu.json"


@pytest.fixture(scope="module")
def reference():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    fixtures = {}
    for name, record in json.loads(_FIXTURE.read_text()).items():
        dtype = record["dtype"]
        values = np.asarray(record["values"], dtype="float64" if dtype == "complex128" else dtype)
        fixtures[name] = torch.from_numpy(values.view(dtype).reshape(record["shape"]))
    yield fixtures
    torch.set_num_threads(threads)


def _model(seed=123, *, fixed_pdd=False):
    model = WorldPlay2Model(
        in_dim=36, out_dim=16, dim=32, ffn_dim=64, freq_dim=8,
        text_dim=12, num_heads=4, num_layers=2, text_len=8,
        eps=1e-6, patch_size=(1, 2, 2), has_image_input=False,
        require_vae_embedding=True, require_clip_embedding=False,
        compressor_options={"dims": (8, 8, 16, 32, 32, 64, 64), "attn_num": 0},
        fixed_pdd=fixed_pdd,
    ).eval()
    model.set_attention_compatibility_mode(True)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, tensor in sorted(model.state_dict().items()):
            tensor.copy_(torch.rand(tensor.shape, generator=generator) * 0.4 - 0.2)
    return model


def _memory(model, reference, frames):
    high, low = reference[f"input_hr{frames}"], reference[f"input_lr{frames}"]
    actions = reference["input_actions"].repeat(frames // 4, 1)
    return model.compress_memory(
        hr_input=high, lr_input=low, lr_actions=actions[::2],
        temporal_input=high[:, :, -1:], temporal_actions=actions[-1:], sink_actions=actions[:1],
    )


def _forward(model, reference, **kwargs):
    return model(
        reference["input_latent"], torch.tensor([930.0]), reference["input_context"],
        y=reference["input_condition"], actions=reference["input_actions"], **kwargs,
    )[0]


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("with_memory", [False, True])
def test_bidirectional_forward_matches_official_fp32_fixture(reference, with_memory):
    model = _model()
    with torch.no_grad():
        memory = _memory(model, reference, 4) if with_memory else None
        actual = _forward(model, reference, memory=memory)
    _close(actual, reference["bi_memory" if with_memory else "bi_first"])


def test_first_and_incremental_ar_cache_match_official_fp32_fixture(reference):
    model = _model()
    cache = model.new_cache()
    with torch.no_grad():
        for frames, prefix, name in ((4, "prefill", "ar_prefilled"), (8, "incremental", "ar_incremental")):
            memory = _memory(model, reference, frames)
            for key in ("tokens", "e", "e0", "vec_action"):
                _close(memory[key], reference[f"memory{frames}_{key}"])
            _close(memory["freqs"].squeeze(1), reference[f"memory{frames}_freqs"])
            model.prefill(memory, reference["input_context"], cache)
            for index, entry in enumerate(cache):
                for key in ("k_vision", "v_vision"):
                    _close(entry[key].unflatten(-1, (4, 8)), reference[f"{prefix}_{index}_{key}"])
            _close(_forward(model, reference, memory=memory, cache=cache, chunk_start=frames), reference[name])


@pytest.mark.parametrize("expert,seed", [("high", 456), ("low", 789)])
@pytest.mark.parametrize("block", [0, 1])
def test_compact_pdd_predictions_match_official_fp32_fixture(reference, expert, seed, block):
    model = _model(seed, fixed_pdd=True)
    with torch.no_grad():
        actual = _forward(model, reference, pdd_block_index=block, cache=model.new_cache())
    _close(actual[:16], reference[f"pdd_{expert}_{block}_displacement"])
    _close(actual[16:], reference[f"pdd_{expert}_{block}_endpoint"])


class _Codec:
    """Deterministic RGB/latent transforms shared by the official rollout fixture."""

    dtype = torch.float32

    def new_stream_state(self):
        return {}

    @staticmethod
    def _encode(pixels, first):
        pixels = pixels[:, :, torch.arange(0 if first else 3, pixels.shape[2], 4)]
        batch, channels, frames, height, width = pixels.shape
        values = F.avg_pool2d(pixels.transpose(1, 2).reshape(batch * frames, channels, height, width), 8)
        values = values.reshape(batch, frames, channels, height // 8, width // 8).transpose(1, 2)
        return values[:, torch.arange(16) % 3].contiguous()

    def encode(self, pixels):
        return self._encode(pixels, True)

    def encode_chunk(self, pixels, *, state, is_first_chunk):
        return self._encode(pixels, is_first_chunk)

    def decode_chunk(self, latents, *, state, is_first_chunk):
        values = latents[:, :3].repeat_interleave(4, dim=2)
        if is_first_chunk:
            values = values[:, :, 3:]
        batch, channels, frames, height, width = values.shape
        values = F.interpolate(
            values.transpose(1, 2).reshape(batch * frames, channels, height, width),
            scale_factor=8, mode="bilinear", align_corners=False,
        )
        return values.reshape(batch, frames, channels, height * 8, width * 8).transpose(1, 2).clamp(-1, 1)


class _TextEncoder:
    def __init__(self, context):
        self.context = context

    def _encode(self, prompts, *, device, dtype):
        return torch.cat([
            self.context if prompt == "p1" else self.context * -0.5 for prompt in prompts
        ]).to(device=device, dtype=dtype)


def test_three_chunk_fast_rollout_matches_official_fp32_fixture(reference):
    codec = _Codec()
    runner = CompressedMemoryRunner(
        model_id="worldplay2", dtype=torch.float32,
        components=RunnerComponents(
            denoiser=WorldPlay2Denoiser(
                _model(456, fixed_pdd=True), _model(789, fixed_pdd=True), mode="few_step", dtype=torch.float32,
            ),
            conditioner=WorldPlay2Conditioner(_TextEncoder(reference["input_context"]), mode="few_step"),
            latent_initializer=WorldPlay2LatentInitializer(), scheduler=FixedPDD4Scheduler(),
            decoder=codec, latent_encoder=codec,
        ),
    )
    output = runner.run(DiffusionRequest(
        prompt="p1", height=64, width=64, num_frames=45, sampling=SamplingConfig(4, 1, 42),
        inputs={
            "image": Image.new("RGB", (64, 64), color=(255, 255, 255)),
            "actions": "w-4,right+space-4,up+wa-4",
            "prompts": {"prompt1": "p1", "prompt2": "p2"},
            "prompt_event": "prompt1-8,prompt2-4",
        },
    ))
    _close(output.latents, reference["fast_3chunk_latents"])
    assert output.sample.shape == (45, 64, 64, 3)
    assert output.metadata["denoiser_calls"] == 12
    expected_chunks = [
        codec.decode_chunk(
            reference["fast_3chunk_latents"][:, :, index * 4:(index + 1) * 4],
            state={}, is_first_chunk=index == 0,
        )
        for index in range(3)
    ]
    expected = torch.cat(expected_chunks, dim=2)[0].permute(1, 2, 3, 0).add(1).mul(0.5)
    _close(output.sample, expected)
