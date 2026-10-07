import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import (
    DenoiserInput,
    DenoiserOutput,
    DiffusionOutput,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE
from worldfoundry.base_models.diffusion_model.optimizations import AttentionBackend, RuntimePolicy
from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowUniPCScheduler
from worldfoundry.pipelines.worldplay2 import WorldPlay2Pipeline
from worldfoundry.synthesis.visual_generation.worldplay2.conditioning import (
    WorldPlay2Conditioner,
    parse_worldplay2_actions,
)
from worldfoundry.synthesis.visual_generation.worldplay2.denoiser import build_worldplay2_denoiser
from worldfoundry.synthesis.visual_generation.worldplay2.initializer import WorldPlay2LatentInitializer
from worldfoundry.synthesis.visual_generation.worldplay2.modeling import WorldPlay2Model
from worldfoundry.synthesis.visual_generation.worldplay2.runner import CompressedMemoryRunner
from worldfoundry.synthesis.visual_generation.worldplay2.scheduler import FixedPDD4Scheduler

SMALL_CONFIG = {
    "dim": 32, "in_dim": 36, "ffn_dim": 64, "out_dim": 16, "text_dim": 12,
    "freq_dim": 8, "eps": 1e-6, "patch_size": (1, 2, 2), "num_heads": 4,
    "num_layers": 2, "has_image_input": False, "require_clip_embedding": False,
    "text_len": 3, "compressor_options": {"dims": (8, 8, 16, 32, 32, 64, 64), "attn_num": 0},
}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_action_durations_angles_perspective_and_stationary_first_latent():
    actions = parse_worldplay2_actions("wd+right+space-4,up-4", latent_frames=8, perspective="fps",
                                      yaw_speed=3, pitch_speed=1)
    torch.testing.assert_close(actions[0], torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32))
    torch.testing.assert_close(actions[1:4], torch.tensor([[0, 3, 1, 1, 1, 1]] * 3, dtype=torch.float32))
    torch.testing.assert_close(actions[4:], torch.tensor([[1, 0, 0, 0, 1, 0]] * 4, dtype=torch.float32))


@pytest.mark.parametrize("value", ["w-7", "w+s-8", "none+right-8", "w-0", "jump-8"])
def test_action_plan_rejects_unsupported_commands_or_wrong_duration(value):
    with pytest.raises(ValueError):
        parse_worldplay2_actions(value, latent_frames=8)


@pytest.mark.parametrize("mode", ["few_step", "ar", "bi"])
def test_released_expert_prefix_restores_native_graph_and_runs(tmp_path, mode):
    model = WorldPlay2Model(fixed_pdd=mode == "few_step", **SMALL_CONFIG).eval()
    if mode == "few_step":
        torch.nn.init.normal_(model.head.block_weight, std=0.01)
        torch.nn.init.zeros_(model.head.block_bias)
    state = {"model." + name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    checkpoint = tmp_path / "model.safetensors"
    save_file(state, checkpoint)
    context = ComponentBuildContext(
        model_id="worldplay2", key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(device=torch.device("cpu"), dtype=torch.float32, attention=AttentionBackend.SDPA),
        checkpoints={"high": CheckpointSpec(source=str(checkpoint)), "low": CheckpointSpec(source=str(checkpoint))},
        recipe_options={"mode": mode}, component_options={"model_config": SMALL_CONFIG},
    )
    denoiser = build_worldplay2_denoiser(context)
    for restored in denoiser.models.values():
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)
    latents = torch.randn(1, 16, 4, 8, 8)
    result = denoiser(DenoiserInput(
        latents=latents, timestep=torch.tensor(1000), next_timestep=torch.tensor(900),
        conditioning={"expert": "high", "context": torch.randn(1, 3, 12),
                      "condition_latents": torch.randn(1, 20, 4, 8, 8), "actions": torch.zeros(4, 6),
                      "chunk_start": 0, "pdd_block_index": 0}, step_index=0, total_steps=4,
    ))
    assert result.sample.shape == latents.shape
    assert torch.isfinite(result.sample).all()
    if mode == "few_step":
        assert result.extras["endpoint_velocity"].shape == latents.shape


def test_wan_streaming_codec_matches_dense_decode_and_encode_with_separate_states():
    vae = WanVideoVAE(base_dim=4).eval()
    codec = WanVideoDecoder(vae, device=torch.device("cpu"))
    latents = torch.randn(1, 16, 8, 2, 2)
    with torch.no_grad():
        expected_pixels = codec.decode(latents)
        consumed_pixels = torch.cat((expected_pixels[:, :, :13:2][:, :, :5],
                                     expected_pixels[:, :, 13::2]), dim=2)
        expected_low_latents = codec.encode(consumed_pixels)
        state = codec.new_stream_state()
        first = codec.decode_chunk(latents[:, :, :4], state=state, is_first_chunk=True)
        low_first = codec.encode_chunk(first[:, :, ::2], state=state, is_first_chunk=True)
        last = codec.decode_chunk(latents[:, :, 4:], state=state)
        low_last = codec.encode_chunk(last[:, :, ::2], state=state)
    assert first.shape[2] == 13 and last.shape[2] == 16
    torch.testing.assert_close(torch.cat((first, last), dim=2), expected_pixels)
    torch.testing.assert_close(torch.cat((low_first, low_last), dim=2), expected_low_latents)
    second_state = codec.new_stream_state()
    assert second_state["decode"] is not state["decode"]


class FakeText:
    def _encode(self, prompts, *, device, dtype):
        return torch.tensor([len(prompt) for prompt in prompts], device=device, dtype=dtype).view(-1, 1, 1)


class FakeCodec:
    dtype = torch.float32

    def encode(self, pixels):
        batch, _, frames, height, width = pixels.shape
        return pixels.new_zeros(batch, 16, (frames - 1) // 4 + 1, height // 8, width // 8)

    def new_stream_state(self):
        return {"decoded": 0}

    def decode_chunk(self, latents, *, state, is_first_chunk):
        state["decoded"] += 1
        frames = latents.shape[2] * 4 - (3 if is_first_chunk else 0)
        return latents.new_zeros(1, 3, frames, latents.shape[3] * 8, latents.shape[4] * 8)

    def encode_chunk(self, pixels, *, state, is_first_chunk):
        del state
        frames = 1 + (pixels.shape[2] - 1) // 4 if is_first_chunk else pixels.shape[2] // 4
        return pixels.new_zeros(1, 16, frames, pixels.shape[3] // 8, pixels.shape[4] // 8)


class FakeDenoiser:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.memories = []
        self.caches = []

    def new_cache(self):
        result = {name: {branch: [] for branch in ("positive", "negative")} for name in ("high", "low")}
        self.caches.append(result)
        return result

    def __call__(self, model_input):
        self.calls.append((model_input.branch, model_input.conditioning["expert"],
                           float(model_input.conditioning["context"].flatten()[0]), model_input.step_index))
        sample = model_input.latents * 0.1
        return DenoiserOutput(sample, {"endpoint_velocity": torch.zeros_like(sample)})

    def prepare_memory(self, **kwargs):
        self.memories.append(kwargs)
        return {"high": {"history": len(self.memories)}, "low": {"history": len(self.memories)}}


@pytest.mark.parametrize("mode,steps,guidance,frames", [("few_step", 4, 1.0, 29), ("ar", 4, 3.5, 29), ("bi", 4, 3.5, 253)])
@pytest.mark.parametrize("return_latent", [False, True])
def test_chunk_runner_preserves_video_cadence_prompt_switch_cfg_and_request_local_caches(mode, steps, guidance, frames, return_latent):
    codec = FakeCodec()
    denoiser = FakeDenoiser(mode)
    scheduler = FixedPDD4Scheduler() if mode == "few_step" else WanFlowUniPCScheduler()
    runner = CompressedMemoryRunner(
        model_id="worldplay2", components=RunnerComponents(
            denoiser=denoiser, conditioner=WorldPlay2Conditioner(FakeText(), mode=mode),
            latent_initializer=WorldPlay2LatentInitializer(), scheduler=scheduler, decoder=codec, latent_encoder=codec,
        ),
    )
    length = 32 if mode == "bi" else 4
    request = DiffusionRequest(
        prompt="one", height=64, width=64, num_frames=frames,
        sampling=SamplingConfig(steps, guidance, 42),
        inputs={"images": Image.new("RGB", (64, 64)), "actions": f"w-{length},right-{length}",
                "prompts": {"a": "one", "b": "second"}, "prompt_event": f"a-{length},b-{length}",
                "return_latent": return_latent},
    )
    first = runner.run(request)
    second = runner.run(request)
    if return_latent:
        torch.testing.assert_close(first.sample, first.latents)
    else:
        assert first.sample.shape == (frames, 64, 64, 3)
    torch.testing.assert_close(first.latents, second.latents)
    assert first.metadata["chunks"] == 2
    assert first.metadata["denoiser_calls"] == (8 if guidance == 1.0 else 16)
    assert {call[2] for call in denoiser.calls[:first.metadata["denoiser_calls"]] if call[0] == "positive"} == {3, 6}
    inputs = denoiser.memories[0]["memory_inputs"]
    assert inputs["hr_input"].shape == (1, 36, length, 8, 8)
    assert inputs["lr_input"].shape == (1, 36, length // 2, 2, 2)
    if mode != "bi":
        assert denoiser.caches[0] is not denoiser.caches[1]


@pytest.mark.parametrize("variant,mode,steps,guidance,chunk_length", [
    ("worldplay2-fast", "few_step", 4, 1.0, 4),
    ("worldplay2-ar", "ar", 40, 3.5, 4),
    ("worldplay2-bi", "bi", 40, 3.5, 32),
])
@pytest.mark.parametrize("output_type", ["auto", "latent"])
def test_public_adapter_selects_variant_and_reuses_separate_base_assets(monkeypatch, variant, mode, steps, guidance, chunk_length, output_type):
    calls = []
    requests = []

    class Native:
        def __init__(self, model_id):
            self.model_id = model_id
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        def __call__(self, request):
            requests.append(request)
            latents = torch.zeros(1, 16, 1, 1, 1)
            sample = latents if request.inputs["return_latent"] else torch.zeros(1, 1, 1, 3)
            return DiffusionOutput(sample, latents)

    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return Native(model_id)

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", load)
    pipeline = WorldPlay2Pipeline.from_pretrained(model_path="/weights/worldplay2", base_model_path="/weights/wan", variant=variant, device="cpu")
    result = pipeline(images=Image.new("RGB", (64, 64)), prompt="scene", output_type=output_type, return_dict=True)
    assert result["sample"].shape == ((1, 16, 1, 1, 1) if output_type == "latent" else (1, 3, 1, 1, 1))
    assert calls[0][0] == ("worldplay2" if mode == "few_step" else variant)
    assert calls[0][1]["checkpoint_overrides"] == {
        "high": "/weights/worldplay2", "low": "/weights/worldplay2", "t5": "/weights/wan",
        "tokenizer": "/weights/wan", "vae": "/weights/wan",
    }
    assert requests[0].sampling.num_inference_steps == steps
    assert requests[0].sampling.guidance_scale == guidance
    assert requests[0].inputs["chunk_length"] == chunk_length
