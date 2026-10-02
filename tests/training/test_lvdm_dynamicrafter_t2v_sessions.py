from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from worldfoundry.base_models.diffusion_model.models.denoisers.t2v_turbo import (  # noqa: E402
    _unet_state_dict as _t2v_turbo_inference_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.t2v_turbo import (  # noqa: E402
    t2v_turbo_guidance_projection,
)
from worldfoundry.training.data.manifest import MediaReference, TrainingSample  # noqa: E402
from worldfoundry.training.data.video_bucketing import VideoLatentGeometry  # noqa: E402
from worldfoundry.training.data.video_cache import (  # noqa: E402
    VideoCachedDataset,
    VideoCacheProvenance,
    VideoCacheStore,
)
from worldfoundry.training.data.video_tensor_contracts import (  # noqa: E402
    dynamicrafter_latent_normalization,
    lvdm_short_latent_normalization,
    t2v_turbo_latent_normalization,
)
from worldfoundry.training.data.video_tensor_import import (  # noqa: E402
    materialize_precomputed_video_training_cache,
)
from worldfoundry.training.engine.dynamicrafter.sft import (  # noqa: E402
    _dynamicrafter_component_state_dicts,
    build_dynamicrafter_single_device_session,
)
from worldfoundry.training.engine.lvdm.sft import (  # noqa: E402
    _short_unet_state_dict,
    build_lvdm_short_single_device_session,
)
from worldfoundry.training.models.dynamicrafter import DynamiCrafterTrainAdapter  # noqa: E402
from worldfoundry.training.models.lvdm import LVDMUnconditionalTrainAdapter  # noqa: E402
from worldfoundry.training.post_training.distillation.t2v_turbo import (  # noqa: E402
    LVDMEpsilonPredictor,
    T2VTurboRoles,
    T2VTurboTrainAdapter,
    build_t2v_turbo_single_device_session,
)
from worldfoundry.training.post_training.distillation.t2v_turbo.builder import (  # noqa: E402
    _base_state_dict,
    build_t2v_turbo_adapter,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe  # noqa: E402
from worldfoundry.training.recipes.spec import TrainingRecipe  # noqa: E402
from worldfoundry.training.safety.shieldgemma import (  # noqa: E402
    SHIELDGEMMA_PROMPT_POLICIES,
    PromptSafetyAudit,
)


class _TinyLVDM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv3d(4, 4, 1, bias=False)

    def forward(self, noisy, timesteps):
        return self.projection(noisy) + timesteps.reshape(-1, 1, 1, 1, 1) * 0.0


class _TinyDynamiDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv3d(4, 4, 1, bias=False)

    def forward(self, noisy, timesteps, *, c_concat, c_crossattn, fs):
        del c_concat
        context = c_crossattn[0].mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return self.projection(noisy) + context + (timesteps + fs).reshape(-1, 1, 1, 1, 1) * 0.0


class _TinyDynamiCheckpointDenoiser(_TinyDynamiDenoiser):
    def __init__(self) -> None:
        super().__init__()
        self.fps_embedding = torch.nn.Linear(1, 1)


class _TinyTurboUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(1, 1))])
        self.time_cond_proj = torch.nn.Linear(4, 1, bias=False)

    def forward(self, noisy, timesteps, *, context, fps=None, timestep_cond=None):
        value = noisy.flatten(2).transpose(1, 2).unsqueeze(-1)
        value = self.input_blocks[0](value).squeeze(-1).transpose(1, 2).reshape_as(noisy)
        guidance = noisy.new_zeros((noisy.shape[0], 1, 1, 1, 1))
        if timestep_cond is not None:
            guidance = self.time_cond_proj(timestep_cond).reshape(-1, 1, 1, 1, 1)
        context_value = context.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        fps_value = 0.0 if fps is None else fps.reshape(-1, 1, 1, 1, 1) * 0.0
        return value + guidance + context_value * 0.0 + fps_value + timesteps.reshape(-1, 1, 1, 1, 1) * 0.0


class _TinyDynamiWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = _TinyDynamiCheckpointDenoiser()


class _TinyTurboCheckpointUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.time_cond_proj = torch.nn.Linear(256, 320, bias=False)


def _provenance(
    model_recipe: str,
    *,
    layout: str,
    normalization: dict[str, object],
    target_fps: float | None = None,
) -> VideoCacheProvenance:
    temporal = 4 if model_recipe == "lvdm-short-unconditional" else 1
    if target_fps is None:
        target_fps = 16.0 if model_recipe == "t2v-turbo" else 8.0
    return VideoCacheProvenance(
        media_uri="tiny.mp4",
        prompt="a moving square",
        model_recipe=model_recipe,
        codec={"name": "tiny"},
        conditioner={"name": "tiny"},
        tokenizer={"name": "tiny"},
        conditioning_inputs={},
        safety_audit={"safe": True},
        frame_sampling={"mode": "tiny"},
        spatial_transform={"mode": "tiny"},
        latent_normalization=normalization,
        task="t2v",
        conditioning_layout=layout,
        aspect_bin="1:1",
        source_num_frames=2 * temporal,
        source_height=16,
        source_width=16,
        source_fps=8.0,
        target_num_frames=2 * temporal,
        target_height=16,
        target_width=16,
        target_fps=target_fps,
        latent_geometry=VideoLatentGeometry(8, 8, temporal, "uniform"),
    )


def _cache(
    root: Path,
    family: str,
    *,
    target_fps: float | None = None,
    condition_fps: int | None = None,
) -> VideoCachedDataset:
    store = VideoCacheStore(root)
    generator = torch.Generator().manual_seed(7)
    conditions = {}
    layouts = {}
    if family == "dynamicrafter-512-i2v":
        conditions = {
            "text_context": torch.randn(2, 2, generator=generator),
            "empty_text_context": torch.zeros(2, 2),
            "image_features_by_frame": torch.randn(2, 1, 3, generator=generator),
            "zero_image_features": torch.zeros(1, 3),
            "fps": torch.tensor([10]),
        }
        layouts = {
            "text_context": "sequence-features",
            "empty_text_context": "sequence-features",
            "image_features_by_frame": "frames-sequence-features",
            "zero_image_features": "sequence-features",
            "fps": "scalar",
        }
        layout = "dynamicrafter-hybrid"
        normalization = dynamicrafter_latent_normalization()
    elif family == "t2v-turbo":
        conditions = {
            "context": torch.randn(2, 1024, generator=generator),
            "unconditional_context": torch.zeros(2, 1024),
        }
        layouts = {"context": "sequence-features", "unconditional_context": "sequence-features"}
        if condition_fps is not None:
            conditions["fps"] = torch.tensor([condition_fps])
            layouts["fps"] = "scalar"
        layout = "videocrafter-text"
        normalization = t2v_turbo_latent_normalization()
    else:
        layout = "none"
        normalization = lvdm_short_latent_normalization()
    entry = store.write_sample(
        sample_id="tiny",
        provenance=_provenance(
            family,
            layout=layout,
            normalization=normalization,
            target_fps=target_fps,
        ),
        clean_latents=torch.randn(4, 2, 2, 2, generator=generator),
        conditioning=conditions,
        conditioning_layouts=layouts,
        latent_loss_mask=torch.ones(1, 2, 2, 2),
        valid_latent_mask=torch.ones(1, 2, 2, 2, dtype=torch.bool),
    )
    store.write_index(entries=(entry,))
    return VideoCachedDataset(root)


def _training_recipe(family: str) -> TrainingRecipe:
    dynamic = family.startswith("dynamicrafter")
    return TrainingRecipe.from_mapping(
        {
            "run": {"id": f"tiny-{family}", "output_dir": "unused"},
            "model": {
                "recipe": family,
                "options": {"training_mode": "image-to-video", "default_fps": 10} if dynamic else {},
            },
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused",
                "cache": "unused",
                "max_latent_tokens_per_microbatch": 8,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {"num_workers": 0, "pin_memory": False},
            },
            "objective": {
                "type": "classic-diffusion",
                "prediction_type": "v_prediction" if dynamic else "epsilon",
                "timestep_sampler": "uniform",
                "conditioning_dropout": 0.05 if dynamic else 0.0,
                "options": (
                    {
                        "num_train_timesteps": 1000,
                        "beta_start": 0.00085,
                        "beta_end": 0.012,
                        "loss_type": "l2",
                        "zero_terminal_snr": True,
                        "dynamic_rescale_final": 0.7,
                        "dynamic_rescale_transition_steps": 400,
                    }
                    if dynamic
                    else {
                        "num_train_timesteps": 1000,
                        "beta_start": 0.0015,
                        "beta_end": 0.0155,
                        "loss_type": "l1",
                    }
                ),
            },
            "optimizer": {"type": "adamw", "learning_rate": 0.001, "weight_decay": 0.0},
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "checkpoint": {"save_every_steps": 1, "async": False},
        }
    )


def _post_recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "tiny-t2v-turbo", "output_dir": "unused"},
            "model": {"recipe": "t2v-turbo"},
            "tuning": {
                "mode": "lora",
                "preset": "t2v-turbo-unet",
                "rank": 64,
                "alpha": 1,
                "dropout": 0.1,
            },
            "data": {
                "manifest": "unused",
                "cache": "unused",
                "max_latent_tokens_per_microbatch": 8,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {"num_workers": 0, "pin_memory": False},
            },
            "algorithm": {"type": "t2v-turbo-distillation", "guidance_embedding_dim": 4},
            "optimizer": {"type": "adamw", "learning_rate": 0.00001, "weight_decay": 0.0},
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "checkpoint": {"save_every_steps": 1, "async": False},
            "export": {"format": "native-lora"},
        }
    )


def _state(session) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in session.engine.adapter.trainable_module.named_parameters()
        if parameter.requires_grad
    }


def _assert_exact_resume(build, tmp_path: Path) -> None:
    continuous = build(tmp_path / "continuous")
    continuous.run(max_steps=2, seed=31)
    expected = _state(continuous)
    checkpoint = continuous.output_dir / "checkpoints" / "step-00000001"
    resumed = build(tmp_path / "resumed")
    summary = resumed.run(max_steps=1, seed=31, resume_checkpoint=checkpoint)
    assert summary.changed_parameter_tensors > 0
    actual = _state(resumed)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_lvdm_cached_session_optimizer_step_and_exact_resume(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache", "lvdm-short-unconditional")
    recipe = _training_recipe("lvdm-short-unconditional")

    def build(output):
        torch.manual_seed(11)
        return build_lvdm_short_single_device_session(
            recipe=recipe,
            adapter=LVDMUnconditionalTrainAdapter(_TinyLVDM(), codec=None),
            dataset=cache,
            output_dir=output,
            fused_adamw=False,
        )

    _assert_exact_resume(build, tmp_path)


def test_dynamicrafter_cached_session_optimizer_step_and_exact_resume(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache", "dynamicrafter-512-i2v")
    recipe = _training_recipe("dynamicrafter-512-i2v")

    def build(output):
        torch.manual_seed(13)
        adapter = DynamiCrafterTrainAdapter(
            denoiser=_TinyDynamiDenoiser(),
            image_projector=torch.nn.Linear(3, 2),
            conditioning_dropout_probability=0.05,
            default_fps=10,
        )
        return build_dynamicrafter_single_device_session(
            recipe=recipe,
            adapter=adapter,
            dataset=cache,
            output_dir=output,
            fused_adamw=False,
        )

    _assert_exact_resume(build, tmp_path)


def test_t2v_turbo_cached_session_optimizer_step_and_exact_resume(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache", "t2v-turbo")
    recipe = _post_recipe()

    def build(output):
        torch.manual_seed(17)
        return build_t2v_turbo_single_device_session(
            recipe=recipe,
            roles=T2VTurboRoles(student=_TinyTurboUNet(), teacher=_TinyTurboUNet()),
            dataset=cache,
            output_dir=output,
            fused_adamw=False,
            initialization_seed=19,
        )

    _assert_exact_resume(build, tmp_path)


def test_t2v_turbo_cache_requires_released_fps_contract(tmp_path: Path) -> None:
    recipe = _post_recipe()
    roles = T2VTurboRoles(student=_TinyTurboUNet(), teacher=_TinyTurboUNet())
    wrong_target = _cache(tmp_path / "wrong-target", "t2v-turbo", target_fps=15.0)
    with pytest.raises(ValueError, match="must target 16 FPS"):
        build_t2v_turbo_single_device_session(
            recipe=recipe,
            roles=roles,
            dataset=wrong_target,
            output_dir=tmp_path / "wrong-target-run",
            fused_adamw=False,
        )

    wrong_condition = _cache(tmp_path / "wrong-condition", "t2v-turbo", condition_fps=15)
    with pytest.raises(ValueError, match="must condition on 16 FPS"):
        build_t2v_turbo_single_device_session(
            recipe=recipe,
            roles=T2VTurboRoles(student=_TinyTurboUNet(), teacher=_TinyTurboUNet()),
            dataset=wrong_condition,
            output_dir=tmp_path / "wrong-condition-run",
            fused_adamw=False,
        )


def test_t2v_turbo_builder_keeps_teacher_fp32_and_student_bf16_compute() -> None:
    root = Path(__file__).resolve().parents[2]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training/t2v_turbo_distillation.yaml")
    teacher = _TinyTurboUNet().to(dtype=torch.bfloat16)
    adapter = build_t2v_turbo_adapter(
        recipe,
        T2VTurboRoles(student=_TinyTurboUNet(), teacher=teacher),
    )

    assert {parameter.dtype for parameter in teacher.parameters()} == {torch.float32}
    assert adapter.student_autocast_dtype is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast contract requires a CUDA device")
def test_t2v_turbo_target_forwards_use_released_cuda_autocast_dtypes() -> None:
    student = _TinyTurboUNet().cuda()
    teacher = _TinyTurboUNet().cuda()
    adapter = T2VTurboTrainAdapter(
        student=LVDMEpsilonPredictor(student),
        teacher=LVDMEpsilonPredictor(teacher),
        student_autocast_dtype=torch.bfloat16,
    )
    observed: dict[str, tuple[bool, torch.dtype]] = {}

    def capture(name):
        def hook(_module, _inputs):
            observed[name] = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))

        return hook

    teacher_handle = teacher.register_forward_pre_hook(capture("teacher"))
    student_handle = student.register_forward_pre_hook(capture("student"))
    noisy = torch.randn(1, 4, 2, 2, 2, device="cuda")
    timesteps = torch.tensor([19], device="cuda")
    context = torch.randn(1, 2, 1, device="cuda")
    fps = torch.tensor([16], device="cuda")
    guidance = torch.randn(1, 4, device="cuda")
    try:
        with torch.no_grad():
            adapter.teacher_prediction(noisy, timesteps, context=context, fps=fps)
            adapter.student_target_prediction(
                noisy,
                timesteps,
                context=context,
                fps=fps,
                guidance_embedding=guidance,
            )
    finally:
        teacher_handle.remove()
        student_handle.remove()

    assert observed == {
        "teacher": (True, torch.float16),
        "student": (True, torch.bfloat16),
    }


def test_t2v_turbo_session_exports_native_ordered_lora(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache", "t2v-turbo")
    session = build_t2v_turbo_single_device_session(
        recipe=_post_recipe(),
        roles=T2VTurboRoles(student=_TinyTurboUNet(), teacher=_TinyTurboUNet()),
        dataset=cache,
        output_dir=tmp_path / "run",
        fused_adamw=False,
        initialization_seed=19,
    )
    session.run(max_steps=1, seed=31)
    artifact = session.export_adapter()
    weights = torch.load(artifact.path / "unet_lora.pt", map_location="cpu", weights_only=True)
    assert len(weights) == 4
    assert set(session._manifest["artifacts"]) == {"adapter"}


def test_native_checkpoint_converters_unwrap_lightning_state_dicts_strictly() -> None:
    lvdm = _TinyLVDM()
    lvdm_checkpoint = {
        "state_dict": {f"model.diffusion_model.{name}": value.clone() for name, value in lvdm.state_dict().items()},
        "optimizer_states": [],
    }
    lvdm.load_state_dict(_short_unet_state_dict(lvdm_checkpoint), strict=True)

    dynamicrafter_denoiser = _TinyDynamiWrapper()
    dynamicrafter_projector = torch.nn.Linear(3, 2)
    dynamicrafter_checkpoint = {
        "state_dict": {
            **{
                f"model.{name.replace('fps_embedding', 'framestride_embed')}": value.clone()
                for name, value in dynamicrafter_denoiser.state_dict().items()
            },
            **{
                f"image_proj_model.{name}": value.clone()
                for name, value in dynamicrafter_projector.state_dict().items()
            },
        },
        "global_step": 17,
    }
    denoiser_state, projector_state = _dynamicrafter_component_state_dicts(dynamicrafter_checkpoint)
    dynamicrafter_denoiser.load_state_dict(denoiser_state, strict=True)
    dynamicrafter_projector.load_state_dict(projector_state, strict=True)

    turbo = _TinyTurboCheckpointUNet()
    base_state = {name: value.clone() for name, value in turbo.state_dict().items() if name != "time_cond_proj.weight"}
    turbo_checkpoint = {
        "state_dict": {f"model.diffusion_model.{name}": value for name, value in base_state.items()},
        "global_step": 0,
    }
    incompatible = turbo.load_state_dict(_base_state_dict(turbo_checkpoint), strict=False)
    assert incompatible.missing_keys == ["time_cond_proj.weight"]
    assert incompatible.unexpected_keys == []

    with torch.no_grad():
        turbo.time_cond_proj.weight.copy_(t2v_turbo_guidance_projection(turbo.time_cond_proj.weight))
    inference_state = _t2v_turbo_inference_state_dict(turbo_checkpoint["state_dict"])
    fresh_turbo = _TinyTurboCheckpointUNet()
    fresh_turbo.load_state_dict(inference_state, strict=True)
    torch.testing.assert_close(
        turbo.time_cond_proj.weight,
        fresh_turbo.time_cond_proj.weight,
        rtol=0.0,
        atol=0.0,
    )


def test_video_tensor_import_has_no_engine_or_post_training_dependency() -> None:
    source = Path("worldfoundry/training/data/video_tensor_import.py").read_text(encoding="utf-8")
    assert "worldfoundry.training.engine" not in source
    assert "worldfoundry.training.post_training" not in source


def test_precomputed_safetensors_import_closes_lvdm_cache_path(tmp_path: Path) -> None:
    prompt = "a moving square"
    audit = PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={name: 0.0 for name in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )
    media = tmp_path / "tiny.mp4"
    media.write_bytes(b"tiny")
    sample = TrainingSample(
        sample_id="tiny",
        task="t2v",
        prompt=prompt,
        media=MediaReference(uri=media.name, mime_type="video/mp4", size_bytes=media.stat().st_size),
        width=16,
        height=16,
        num_frames=8,
        fps=8.0,
        conditions={},
        split="train",
        safety={"prompt_safe": True, "model_revision": audit.model_revision},
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    source = tmp_path / "precomputed"
    source.mkdir()
    from safetensors.torch import save_file

    save_file({"clean_latents": torch.randn(4, 2, 2, 2)}, source / "tiny.safetensors")
    result = materialize_precomputed_video_training_cache(
        _training_recipe("lvdm-short-unconditional"),
        manifest_path=manifest,
        cache_dir=tmp_path / "imported-cache",
        source_dir=source,
        device="cpu",
        safety_audits=(audit,),
    )
    cached = VideoCachedDataset(tmp_path / "imported-cache", expected_sample_ids=("tiny",))
    assert result.index == cached.index
    assert cached[0].tensors["clean_latents"].shape == (4, 2, 2, 2)


def test_t2v_turbo_precomputed_import_uses_profile_fps_as_provenance(tmp_path: Path) -> None:
    prompt = "a moving square"
    audit = PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={name: 0.0 for name in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )
    media = tmp_path / "tiny.mp4"
    media.write_bytes(b"tiny")
    sample = TrainingSample(
        sample_id="tiny",
        task="t2v",
        prompt=prompt,
        media=MediaReference(uri=media.name, mime_type="video/mp4", size_bytes=media.stat().st_size),
        width=16,
        height=16,
        num_frames=8,
        fps=8.0,
        conditions={},
        split="train",
        safety={"prompt_safe": True, "model_revision": audit.model_revision},
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    source = tmp_path / "precomputed"
    source.mkdir()
    from safetensors.torch import save_file

    save_file(
        {
            "clean_latents": torch.randn(4, 2, 2, 2),
            "context": torch.randn(2, 1024),
            "unconditional_context": torch.zeros(2, 1024),
        },
        source / "tiny.safetensors",
    )
    result = materialize_precomputed_video_training_cache(
        _post_recipe(),
        manifest_path=manifest,
        cache_dir=tmp_path / "imported-cache",
        source_dir=source,
        device="cpu",
        safety_audits=(audit,),
    )
    cached = VideoCachedDataset(tmp_path / "imported-cache", expected_sample_ids=("tiny",))

    assert result.index == cached.index
    assert cached.index.entries[0].provenance.target_fps == 16.0
    assert "condition.fps" not in cached.index.entries[0].tensors


def test_formal_dynamicrafter_1024_profile_uses_official_single_sample_budget() -> None:
    recipe = TrainingRecipe.from_file("configs/training/dynamicrafter_1024_i2v.yaml")
    assert recipe.data.max_latent_tokens_per_microbatch == 16 * 72 * 128


def test_formal_t2v_turbo_profile_uses_released_single_sample_microbatch() -> None:
    recipe = PostTrainingRecipe.from_file("configs/post_training/t2v_turbo_distillation.yaml")
    bucket = recipe.data.options["video_buckets"][0]
    assert (bucket["num_frames"], bucket["height"], bucket["width"]) == (16, 320, 512)
    assert recipe.data.max_latent_tokens_per_microbatch == 16 * 40 * 64
    assert recipe.algorithm.default_fps == 16
    assert recipe.checkpoint.save_every_steps == 2000
    assert recipe.tuning.preset == "t2v-turbo-unet"
    assert recipe.export.format == "native-lora"
