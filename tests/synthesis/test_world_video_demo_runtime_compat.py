from pathlib import Path

import pytest
import torch

from worldfoundry.synthesis.visual_generation.video_x_fun.worldfoundry_runtime import (
    Wan22Fun5BCameraRuntime,
    Wan22FunA14BCameraRuntime,
)


def test_official_video_runtime_resolves_package_relative_config_from_tmp(monkeypatch, tmp_path):
    from worldfoundry.synthesis.visual_generation.official_video_runtime import (
        _existing_path,
    )

    monkeypatch.chdir(tmp_path)
    resolved = _existing_path("worldfoundry/data/models/runtime/configs/video_official/krea-realtime-video.yaml")

    assert resolved is not None
    assert resolved.name == "krea-realtime-video.yaml"


def test_official_video_runtime_skips_incomplete_checkpoint_candidate(tmp_path):
    from worldfoundry.synthesis.visual_generation.official_video_runtime import (
        OfficialVideoRuntime,
    )

    incomplete = tmp_path / "legacy-checkpoint"
    incomplete.mkdir()
    complete = tmp_path / "normalized-checkpoint"
    (complete / "transformer").mkdir(parents=True)
    (complete / "vae").mkdir()

    runtime = OfficialVideoRuntime.from_model_id("helios")
    runtime.runtime.update(
        {
            "checkpoint_candidates": [str(incomplete), str(complete)],
            "required_paths": [
                {"id": "transformer", "path": "transformer"},
                {"id": "vae", "path": "vae"},
            ],
        }
    )

    report = runtime._requirement_report()

    assert report.ready
    assert report.checkpoint_path == complete.resolve()


def test_worldgen_lora_resolves_normalized_hugging_face_directory(monkeypatch, tmp_path):
    from worldfoundry.synthesis.visual_generation.worldgen.worldgen_runtime.src.worldgen import pano_gen

    checkpoint = tmp_path / "LeoXie--WorldGen" / "models--WorldGen-Flux-Lora" / "worldgen_text2scene.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"worldgen lora")
    monkeypatch.setattr(pano_gen, "CKPT_ROOT", tmp_path)

    assert pano_gen._worldgen_lora(checkpoint.name) == str(checkpoint)


def test_worldgen_direct_panorama_skips_flux_initialization(monkeypatch):
    from worldfoundry.synthesis.visual_generation.worldgen.worldgen_runtime.src.worldgen import (
        worldgen as worldgen_module,
    )

    monkeypatch.setattr(worldgen_module, "build_depth_model", lambda _device: object())

    def _reject_flux_initialization(**_kwargs):
        raise AssertionError("direct panorama requests must not initialize FLUX")

    monkeypatch.setattr(worldgen_module, "build_pano_gen_model", _reject_flux_initialization)
    runtime = worldgen_module.WorldGen(
        mode="t2s",
        device=torch.device("cpu"),
        low_vram=False,
        initialize_pano_generator=False,
    )

    assert runtime.pano_gen_model is None


def test_studio_persists_external_encoded_video_in_run(tmp_path):
    from worldfoundry.studio.inference.execution import _persist_video_artifact_in_run

    source = tmp_path / "temporary" / "generated.mp4"
    source.parent.mkdir()
    source.write_bytes(b"encoded-video")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    output_path = output_dir / "demo.mp4"

    persisted = Path(
        _persist_video_artifact_in_run(
            source,
            output_dir=output_dir,
            output_path=output_path,
        )
    )

    assert persisted == output_path
    assert persisted.read_bytes() == b"encoded-video"


def test_lingbot_flash_attention_accepts_fa3_tuple_output():
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot.attention import (
        _flash_attention_output_tensor,
    )

    expected = torch.zeros(2, 3)
    assert _flash_attention_output_tensor(expected) is expected
    assert _flash_attention_output_tensor((expected, torch.ones(1))) is expected


def test_mmaudio_euler_flow_does_not_require_torchdiffeq():
    from worldfoundry.synthesis.visual_generation.mmaudio.mmaudio.model.flow_matching import (
        FlowMatching,
    )

    flow_matching = FlowMatching(inference_mode="euler", num_steps=2)
    result = flow_matching.to_data(
        lambda _time, value: torch.ones_like(value),
        torch.zeros(1, 2, 3),
    )

    torch.testing.assert_close(result, torch.ones_like(result))


def test_mmaudio_resolves_dfn5b_from_alternate_plural_root(monkeypatch, tmp_path):
    from worldfoundry.synthesis.visual_generation.mmaudio.checkpoints import (
        dfn5b_open_clip_ref,
        resolve_bigvgan_v2_checkpoint,
        resolve_dfn5b_checkpoint,
    )

    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    checkpoint = tmp_path / "ckpts" / "apple--DFN5B-CLIP-ViT-H-14-384"
    checkpoint.mkdir(parents=True)
    (checkpoint / "open_clip_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "open_clip_pytorch_model.bin").write_bytes(b"weights")

    assert resolve_dfn5b_checkpoint() == checkpoint.resolve()
    assert dfn5b_open_clip_ref() == f"local-dir:{checkpoint.resolve()}"

    bigvgan = tmp_path / "ckpts" / "nvidia--bigvgan_v2_44khz_128band_512x"
    bigvgan.mkdir()
    (bigvgan / "config.json").write_text("{}", encoding="utf-8")
    (bigvgan / "bigvgan_generator.pt").write_bytes(b"weights")
    assert resolve_bigvgan_v2_checkpoint() == bigvgan.resolve()


def test_echo_and_infinite_world_catalog_expose_runtime_load_controls(monkeypatch, tmp_path):
    from worldfoundry.studio.inference import catalog as catalog

    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    (tmp_path / "Wan-AI--Wan2.1-T2V-1.3B").mkdir()

    echo = catalog.CURATED_OVERRIDES["echo-infinity"]
    echo_load_params = set(echo["load_params"])
    assert {
        "num_output_frames",
        "denoising_step_list",
        "num_frame_per_block",
    } <= echo_load_params
    echo_defaults = echo["default_load_kwargs"]()
    assert echo_defaults["wan_model_name"] == "Wan-AI--Wan2.1-T2V-1.3B"
    assert echo_defaults["model_kwargs"]["model_name"] == "Wan-AI--Wan2.1-T2V-1.3B"

    infinite_load_params = set(catalog.CURATED_OVERRIDES["infinite-world"]["load_params"])
    assert {"num_sampling_steps", "guidance_scale", "shift"} <= infinite_load_params


def test_echo_catalog_discovers_wan_base_in_plural_checkpoint_sibling(monkeypatch, tmp_path):
    from worldfoundry.studio.inference import catalog as catalog

    configured_root = tmp_path / "ckpt"
    staged_root = tmp_path / "ckpts"
    configured_root.mkdir()
    (staged_root / "Wan-AI--Wan2.1-T2V-1.3B").mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(configured_root))

    defaults = catalog.CURATED_OVERRIDES["echo-infinity"]["default_load_kwargs"]()

    assert defaults["wan_root"] == str(staged_root)
    assert defaults["wan_model_name"] == "Wan-AI--Wan2.1-T2V-1.3B"
    assert defaults["model_kwargs"]["wan_root"] == str(staged_root)


def test_vmem_trajectory_export_does_not_require_evo():
    import numpy as np

    from worldfoundry.synthesis.visual_generation.vmem.vmem_runtime.surfel_alignment.cloud_opt.dust3r_opt.base_opt import (
        make_traj,
    )

    poses = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    timestamps = np.array([0.0], dtype=np.float64)
    trajectory = make_traj([poses, timestamps])

    np.testing.assert_array_equal(trajectory.positions_xyz, poses[:, :3])
    np.testing.assert_array_equal(trajectory.orientations_quat_wxyz, poses[:, 3:])
    assert trajectory.num_poses == 1


def test_step_video_loads_trusted_local_legacy_bert_checkpoint(monkeypatch, tmp_path):
    from transformers import BertConfig, BertModel

    from worldfoundry.base_models.diffusion_model.models.encoders.step_video import clip

    config = BertConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=16,
    )
    expected = BertModel(config, add_pooling_layer=False)
    config.save_pretrained(tmp_path)
    state_dict = {f"bert.{key}": value for key, value in expected.state_dict().items()}
    state_dict["visual.unused"] = torch.zeros(1)
    torch.save(state_dict, tmp_path / "pytorch_model.bin")

    def _reject_legacy_bin(*args, **kwargs):
        del args, kwargs
        raise ValueError("upgrade torch to at least v2.6")

    monkeypatch.setattr(clip.BertModel, "from_pretrained", _reject_legacy_bin)
    loaded = clip._load_local_bert_model(str(tmp_path))

    torch.testing.assert_close(
        loaded.embeddings.word_embeddings.weight,
        expected.embeddings.word_embeddings.weight,
    )
    assert loaded.pooler is not None


def test_unianimate_runner_passes_requested_frame_count_to_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from worldfoundry.synthesis.visual_generation.unianimate_dit.unianimate_dit_runtime.examples.unianimate_wan import (
        inference_unianimate_wan_480p as worker,
    )

    captured = {}
    monkeypatch.setattr(worker, "run_inference", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["unianimate", "--max-frames", "9", "--num-inference-steps", "1"])
    worker.main()

    assert captured["max_frames"] == 9
    assert captured["steps"] == 1


def test_wan_vace_adapter_normalizes_standard_generation_kwargs(monkeypatch):
    from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline
    from worldfoundry.pipelines.wan.pipeline_wan_vace import Wan2p1VACEPipeline
    from worldfoundry.studio.inference.catalog import CURATED_OVERRIDES

    assert CURATED_OVERRIDES["wan2.1-vace"]["default_load_kwargs"] == {"torch_dtype": "bfloat16"}

    captured = {}

    def _capture_call(self, **kwargs):
        del self
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(NativeTextToVideoPipeline, "__call__", _capture_call)
    pipeline = object.__new__(Wan2p1VACEPipeline)
    result = pipeline(
        prompt="demo",
        size="448*256",
        num_frames=5,
        num_inference_steps=1,
        guidance_scale=1.0,
        shift=5.0,
        seed=42,
        nproc_per_node=4,
        ulysses_size=4,
        ring_size=1,
    )

    assert result["height"] == 256
    assert result["width"] == 448
    assert result["num_frames"] == 5
    assert result["num_inference_steps"] == 1
    assert result["guidance_scale"] == 1.0
    assert result["shift"] == 5.0
    assert result["seed"] == 42
    assert "nproc_per_node" not in result
    assert "ulysses_size" not in result
    assert "ring_size" not in result
    assert captured == result


def test_wan_vace_denoiser_uses_checkpoint_compute_dtype():
    from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
    from worldfoundry.base_models.diffusion_model.models.denoisers.wan_vace import (
        WanVaceDenoiser,
    )

    class _FakeVace(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        def forward(self, *, x, timestep, context, vace_context, vace_context_scale):
            assert x.dtype == torch.bfloat16
            assert timestep.dtype == torch.bfloat16
            assert context.dtype == torch.bfloat16
            assert vace_context.dtype == torch.bfloat16
            assert vace_context_scale == 1.0
            return x

    latents = torch.zeros(1, 16, 2, 4, 4, dtype=torch.float32)
    output = WanVaceDenoiser(_FakeVace())(  # ty: ignore[invalid-argument-type]  # duck-typed stand-in for VaceWanModel
        DenoiserInput(
            latents=latents,
            timestep=torch.ones(1, dtype=torch.float32),
            next_timestep=torch.zeros(1, dtype=torch.float32),
            conditioning={
                "context": torch.zeros(1, 2, 8, dtype=torch.float32),
                "vace_context": torch.zeros(1, 96, 2, 4, 4, dtype=torch.float32),
            },
            step_index=0,
            total_steps=1,
            branch="positive",
        )
    )

    assert output.sample.dtype == latents.dtype


def test_mvdiffusion_resolves_hfd_layout_without_hub_access(monkeypatch, tmp_path):
    from worldfoundry.base_models.three_dimensions.general_3d.mvdiffusion.mvdiffusion_runtime.src.pano_outpaint_generator import (
        _resolve_model_id,
    )

    model_root = tmp_path / "stabilityai--stable-diffusion-2-inpainting"
    model_root.mkdir()
    (model_root / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path))

    assert _resolve_model_id("stabilityai/stable-diffusion-2-inpainting") == str(model_root.resolve())


def test_mvdiffusion_selects_local_fp16_only_diffusers_components(tmp_path):
    from worldfoundry.base_models.three_dimensions.general_3d.mvdiffusion.mvdiffusion_runtime.src.pano_outpaint_generator import (
        _local_fp16_variant_kwargs,
    )

    component = tmp_path / "text_encoder"
    component.mkdir()
    fp16_weight = component / "model.fp16.safetensors"
    fp16_weight.write_bytes(b"weights")

    assert _local_fp16_variant_kwargs(
        str(tmp_path),
        subfolder="text_encoder",
        weight_stem="model",
    ) == {"variant": "fp16"}

    (component / "model.safetensors").write_bytes(b"weights")
    assert (
        _local_fp16_variant_kwargs(
            str(tmp_path),
            subfolder="text_encoder",
            weight_stem="model",
        )
        == {}
    )
    assert (
        _local_fp16_variant_kwargs(
            "stabilityai/stable-diffusion-2-inpainting",
            subfolder="text_encoder",
            weight_stem="model",
        )
        == {}
    )


def test_mvdiffusion_checkpoint_resolves_alternate_plural_root(monkeypatch, tmp_path):
    from worldfoundry.base_models.three_dimensions.general_3d.mvdiffusion.mvdiffusion_runtime.src.checkpoint_utils import (
        resolve_mvdiffusion_checkpoint,
    )

    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    checkpoint = tmp_path / "ckpts" / "MVDiffusion" / "weights" / "pano_outpaint.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_mvdiffusion_checkpoint("MVDiffusion", "weights", "pano_outpaint.ckpt") == checkpoint.resolve()


def test_mvdiffusion_plan_forwards_steps_and_skips_missing_prompt_file(tmp_path):
    from worldfoundry.base_models.three_dimensions.three_d_four_d.runtime import (
        ThreeDFourDRuntimeSynthesis,
        three_d_four_d_runtime_spec,
    )
    from worldfoundry.evals.utils import REPO_ROOT
    from worldfoundry.studio.inference.catalog import CURATED_OVERRIDES

    assert CURATED_OVERRIDES["mvdiffusion"]["default_call_kwargs"]["steps"] == 20

    image = REPO_ROOT / "worldfoundry/data/test_cases/mvdiffusion/outpaint_example.png"
    runtime = ThreeDFourDRuntimeSynthesis(
        spec=three_d_four_d_runtime_spec("mvdiffusion"),
        device="cuda",
        options={"steps": 1, "text_path": "assets/prompts.txt", "gen_video": True},
    )

    result = runtime.predict(
        prompt="A coherent panorama.",
        images=[image],
        output_path=tmp_path / "mvdiffusion.png",
        run_dir=tmp_path / "run",
        plan_only=True,
    )

    command = result["command"]
    assert command[command.index("--steps") + 1] == "1"
    assert "--text_path" not in command
    assert "--gen_video" in command


@pytest.mark.parametrize(
    "runtime_class",
    (
        pytest.param(Wan22Fun5BCameraRuntime, id="wan22-5b"),
        pytest.param(Wan22FunA14BCameraRuntime, id="wan22-a14b"),
    ),
)
def test_wan22_fun_complete_checkpoint_layout_does_not_require_wan21_clip(
    runtime_class,
    tmp_path,
):
    checkpoint = tmp_path / runtime_class.MODEL_ID
    for relative in runtime_class.REQUIRED_COMPONENT_FILES:
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"component")
    runtime = runtime_class.__new__(runtime_class)

    assert runtime._materialize_checkpoint_view(tmp_path / "view", checkpoint) == checkpoint
    assert "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" not in (runtime_class.REQUIRED_COMPONENT_FILES)
