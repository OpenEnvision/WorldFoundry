from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.checkpoint.stateful import NamedStatefulCollection  # noqa: E402
from worldfoundry.training.engine.sana import sid as sana_sid_engine  # noqa: E402
from worldfoundry.training.engine.sana.sid_run import (  # noqa: E402
    SANA_SID_RUN_SCHEMA,
    SanaSIDTrainingRun,
)
from worldfoundry.training.models.sana_sid import (  # noqa: E402
    build_local_diffusers_sana_sid_adapter,
)
from worldfoundry.training.post_training import (  # noqa: E402
    NativeSIDLossAdapter,
    NativeSIDTrainEngine,
    NativeSIDTrainingSession,
    SIDConfig,
    SIDLossResult,
    SIDRunSummary,
    SIDTrainingBatch,
    build_native_sid_training_stack,
    simulate_sid_student,
)
from worldfoundry.training.post_training.distillation.dmd.objective import (  # noqa: E402
    FewStepSchedule,
)
from worldfoundry.training.recipes import (  # noqa: E402
    PostTrainingRecipe,
    SIDAlgorithmSpec,
)


def _expand(levels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return levels.reshape((levels.shape[0],) + (1,) * (reference.ndim - 1))


class _Scale(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


class _FakeScale(_Scale):
    def __init__(self, value: float) -> None:
        super().__init__(value)
        self.discriminator_weight = torch.nn.Parameter(torch.tensor(0.3))


class _Adapter:
    noise_process_kind = "flow-matching"

    def __init__(self, module: _Scale, checkpoint_identity: str) -> None:
        self.module = module
        self.checkpoint_identity = checkpoint_identity
        self.noises: list[torch.Tensor] = []
        self.inputs: list[torch.Tensor] = []
        self.cleans: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []

    def add_noise(self, clean_latents, noise, sigmas):
        self.noises.append(noise.detach().clone())
        levels = _expand(sigmas, clean_latents)
        return (1.0 - levels) * clean_latents + levels * noise

    def predict_velocity(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del sigmas, sample_ids, conditioning, training, branch
        self.grad_enabled.append(torch.is_grad_enabled())
        return noisy_latents * self.module.weight

    def predict_clean(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        self.inputs.append(noisy_latents.detach().clone())
        velocity = self.predict_velocity(
            noisy_latents,
            sigmas,
            sample_ids=sample_ids,
            conditioning=conditioning,
            training=training,
            branch=branch,
        )
        clean = noisy_latents - _expand(sigmas, noisy_latents) * velocity
        self.cleans.append(clean.detach().clone())
        return clean


class _DiscriminatorAdapter(_Adapter):
    module: _FakeScale

    def discriminator_logits(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del sigmas, sample_ids, conditioning, training
        return noisy_latents.float().reshape(noisy_latents.shape[0], -1).mean(1) * (
            self.module.discriminator_weight
        )


def _batch(values=(0.0, 0.0), *, real: bool = False, weights=None, prefix="sample") -> SIDTrainingBatch:
    template = torch.tensor(values, dtype=torch.float32).reshape(len(values), 1)
    kwargs = {}
    if real:
        kwargs = {
            "real_sample_ids": tuple(f"{prefix}-real-{index}" for index in range(len(values))),
            "real_latents": torch.arange(1, len(values) + 1, dtype=torch.float32).reshape(len(values), 1),
            "real_conditioning": {},
        }
    return SIDTrainingBatch(
        sample_ids=tuple(f"{prefix}-{index}" for index in range(len(values))),
        latent_template=template,
        conditioning={},
        unconditional_conditioning={},
        sample_weights=None if weights is None else torch.tensor(weights, dtype=torch.float32),
        **kwargs,
    )


def test_local_diffusers_sana_loader_uses_supported_dtype_keyword(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    class FakeTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.config = SimpleNamespace(in_channels=1, guidance_embeds=False)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["transformer"] = {"path": path, **kwargs}
            return cls()

    class FakePipeline:
        def __init__(self) -> None:
            self.text_encoder = torch.nn.Linear(1, 1)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["pipeline"] = {"path": path, **kwargs}
            return cls()

        def to(self, device):
            self.text_encoder.to(device)
            return self

        def encode_prompt(self, *args, **kwargs):
            del args, kwargs
            return torch.zeros(1, 1, 1), torch.ones(1, 1), None, None

    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        SimpleNamespace(
            SanaPipeline=FakePipeline,
            SanaTransformer2DModel=FakeTransformer,
        ),
    )
    _, prediction = build_local_diffusers_sana_sid_adapter(
        str(tmp_path),
        device="cpu",
        dtype=torch.bfloat16,
        checkpoint_identity="local-sana",
        load_conditioner=True,
    )

    assert prediction.checkpoint_identity == "local-sana"
    assert calls["transformer"]["torch_dtype"] is torch.bfloat16
    assert calls["pipeline"]["torch_dtype"] is torch.bfloat16
    assert "dtype" not in calls["transformer"]
    assert "dtype" not in calls["pipeline"]
    assert calls["transformer"]["local_files_only"] is True
    assert calls["pipeline"]["local_files_only"] is True
    assert calls["transformer"]["use_safetensors"] is True
    assert calls["pipeline"]["use_safetensors"] is True


def test_local_diffusers_sana_loader_separates_master_and_compute_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, dict[str, object]] = {}

    class FakeTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.config = SimpleNamespace(in_channels=1, guidance_embeds=False)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["transformer"] = {"path": path, **kwargs}
            return cls()

    class FakePipeline:
        def __init__(self) -> None:
            self.text_encoder = torch.nn.Linear(1, 1)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["pipeline"] = {"path": path, **kwargs}
            return cls()

        def to(self, device):
            self.text_encoder.to(device)
            return self

        def encode_prompt(self, *args, **kwargs):
            del args, kwargs
            return torch.zeros(1, 1, 1), torch.ones(1, 1), None, None

    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        SimpleNamespace(
            SanaPipeline=FakePipeline,
            SanaTransformer2DModel=FakeTransformer,
        ),
    )

    build_local_diffusers_sana_sid_adapter(
        str(tmp_path),
        device="cpu",
        dtype=torch.bfloat16,
        parameter_dtype=torch.float32,
        checkpoint_identity="local-sana",
        load_conditioner=True,
    )

    assert calls["transformer"]["torch_dtype"] is torch.float32
    assert calls["pipeline"]["torch_dtype"] is torch.bfloat16


def test_sana_sid_asset_identity_covers_sharded_diffusers_weights(tmp_path: Path) -> None:
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    first = "diffusion_pytorch_model-00001-of-00002.safetensors"
    second = "diffusion_pytorch_model-00002-of-00002.safetensors"
    (transformer / first).write_bytes(b"first")
    (transformer / second).write_bytes(b"second")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"block.0": first, "block.1": second}}),
        encoding="utf-8",
    )

    identity = sana_sid_engine._asset_identity(
        tmp_path,
        conditioner=False,
    )
    (transformer / second).write_bytes(b"changed")
    changed = sana_sid_engine._asset_identity(
        tmp_path,
        conditioner=False,
    )

    assert set(identity["files"]) == {
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        f"transformer/{first}",
        f"transformer/{second}",
    }
    assert changed != identity


def _config(**changes) -> SIDConfig:
    values = {
        "schedule": FewStepSchedule((999.0, 749.0, 499.0), (0.999, 0.749, 0.499)),
        "alpha": 1.0,
        "score_identity_weight": 1.0,
        "fake_score_flow_weight": 1.0,
    }
    values.update(changes)
    return SIDConfig(**values)


@pytest.mark.parametrize("policy", ["fresh", "fixed", "ddim"])
def test_sid_detaches_prefix_and_implements_all_noise_policies(policy: str) -> None:
    adapter = _Adapter(_Scale(0.2), "student")
    simulate_sid_student(
        adapter,
        _batch((0.0, 0.0)),
        _config().schedule,
        target_index=2,
        noise_policy=policy,
        generator=torch.Generator().manual_seed(7),
        training=True,
    )
    assert adapter.grad_enabled == [False, False, True]
    if policy == "fresh":
        assert not torch.equal(adapter.noises[0], adapter.noises[1])
        assert not torch.equal(adapter.noises[1], adapter.noises[2])
    elif policy == "fixed":
        torch.testing.assert_close(adapter.noises[0], adapter.noises[1], rtol=0, atol=1e-6)
        torch.testing.assert_close(adapter.noises[0], adapter.noises[2], rtol=0, atol=1e-6)
    if policy == "ddim":
        sigma = _config().schedule.sigmas[0]
        recovered = (adapter.inputs[0] - (1.0 - sigma) * adapter.cleans[0]) / sigma
        torch.testing.assert_close(adapter.noises[1], recovered, rtol=0, atol=1e-6)


def _native_losses(*, gan: bool = False):
    student = _Adapter(_Scale(0.2), "student")
    teacher = _Adapter(_Scale(0.6).requires_grad_(False), "teacher")
    fake = (
        _DiscriminatorAdapter(_FakeScale(0.4), "fake")
        if gan
        else _Adapter(_Scale(0.4), "fake")
    )
    config = _config(
        generator_adversarial_weight=0.2 if gan else 0.0,
        fake_score_adversarial_weight=0.3 if gan else 0.0,
    )
    return NativeSIDLossAdapter(student, teacher, fake, config), student, teacher, fake


def test_sid_objective_isolates_fake_score_and_generator_gradients() -> None:
    losses, student, teacher, fake = _native_losses()
    batch = _batch()
    fake_result = losses.fake_score_loss(
        batch,
        target_index=1,
        generator=torch.Generator().manual_seed(11),
    )
    fake_result.loss.backward()
    assert student.module.weight.grad is None
    assert fake.module.weight.grad is not None
    assert teacher.module.weight.grad is None
    fake.module.zero_grad(set_to_none=True)

    generator_result = losses.generator_loss(
        batch,
        target_index=1,
        generator=torch.Generator().manual_seed(13),
    )
    generator_result.loss.backward()
    assert student.module.weight.grad is not None
    assert fake.module.weight.grad is None
    assert teacher.module.weight.grad is None


def test_sid_diffusion_gan_requires_real_data_and_keeps_fake_params_isolated_from_g() -> None:
    losses, student, _, fake = _native_losses(gan=True)
    with pytest.raises(ValueError, match="real latent batch"):
        losses.fake_score_loss(
            _batch(),
            target_index=0,
            generator=torch.Generator().manual_seed(17),
        )
    batch = _batch(real=True)
    fake_result = losses.fake_score_loss(
        batch,
        target_index=0,
        generator=torch.Generator().manual_seed(17),
    )
    assert "fake_score_adversarial" in fake_result.metrics
    fake.module.zero_grad(set_to_none=True)
    generator_result = losses.generator_loss(
        batch,
        target_index=0,
        generator=torch.Generator().manual_seed(19),
    )
    generator_result.loss.backward()
    assert student.module.weight.grad is not None
    assert all(parameter.grad is None for parameter in fake.module.parameters())


def _recipe_mapping() -> dict[str, object]:
    return {
        "run": {"id": "sid-test", "output_dir": "runs/sid-test"},
        "model": {"recipe": "toy-flow", "checkpoint": "student"},
        "tuning": {"mode": "full"},
        "data": {"manifest": "prompts.jsonl"},
        "algorithm": {
            "type": "sid",
            "student_timesteps": [999.0, 749.0],
            "student_sigmas": [0.999, 0.749],
            "teacher_checkpoint": "teacher",
            "fake_score_checkpoint": "fake",
            "alpha": 1.0,
        },
        "optimizer": {"type": "adamw", "learning_rate": 5.0e-6},
        "fake_score_optimizer": {"type": "adamw", "learning_rate": 5.0e-6},
        "export": {"format": "safetensors"},
    }


def test_sid_recipe_is_strict_and_builder_checks_all_loaded_role_identities() -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())
    assert isinstance(recipe.algorithm, SIDAlgorithmSpec)
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
    unknown = deepcopy(_recipe_mapping())
    unknown["algorithm"]["unused"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        PostTrainingRecipe.from_mapping(unknown)
    student = _Adapter(_Scale(0.2), "student")
    teacher = _Adapter(_Scale(0.6).requires_grad_(False), "wrong")
    fake = _Adapter(_Scale(0.4), "fake")
    with pytest.raises(ValueError, match="loaded checkpoint identity"):
        build_native_sid_training_stack(
            recipe,
            student=student,
            teacher=teacher,
            fake_score=fake,
            fused_adamw=False,
        )
    teacher.checkpoint_identity = "teacher"
    stack = build_native_sid_training_stack(
        recipe,
        student=student,
        teacher=teacher,
        fake_score=fake,
        fused_adamw=False,
    )
    assert stack.student_optimizer.param_groups[0]["lr"] == 5.0e-6
    assert stack.fake_score_optimizer.param_groups[0]["lr"] == 5.0e-6


class _Counter:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state_dict) -> None:
        self.steps = int(state_dict["steps"])


class _EMA:
    def __init__(self, module: torch.nn.Linear) -> None:
        self.shadow = module.weight.detach().clone()

    def update(self, module: torch.nn.Linear) -> None:
        self.shadow.mul_(0.5).add_(module.weight.detach(), alpha=0.5)

    def state_dict(self):
        return {"shadow": self.shadow.clone()}

    def load_state_dict(self, state_dict) -> None:
        self.shadow.copy_(state_dict["shadow"])


class _EngineLosses:
    num_student_steps = 3

    def __init__(self, student, fake, *, fail_generator_call=None, use_generator=False) -> None:
        self.student = student
        self.fake = fake
        self.fail_generator_call = fail_generator_call
        self.use_generator = use_generator
        self.generator_calls = 0
        self.seen_fake_weights: list[float] = []
        self.target_indices: list[tuple[str, int]] = []
        self.seen_batches: list[tuple[str, tuple[str, ...]]] = []

    def loss_denominator(self, batch, *, role):
        del role
        if batch.sample_weights is None:
            return torch.tensor(float(batch.batch_size))
        return batch.sample_weights.sum()

    def _weights(self, batch):
        if batch.sample_weights is None:
            return torch.ones(batch.batch_size)
        return batch.sample_weights

    def _jitter(self, generator):
        if not self.use_generator:
            return torch.tensor(0.0)
        return torch.rand((), generator=generator) * 0.01

    def _result(self, per_sample, batch, **metrics):
        weights = self._weights(batch).to(per_sample)
        return SIDLossResult(
            loss=(per_sample * weights).sum() / weights.sum(),
            metrics={"loss_denominator": weights.sum(), **metrics},
        )

    def fake_score_loss(self, batch, *, target_index, generator=None):
        self.target_indices.append(("fake", target_index))
        self.seen_batches.append(("fake", batch.sample_ids))
        values = batch.latent_template.float()
        prediction = self.fake(values)
        target = values * (0.25 + self._jitter(generator))
        return self._result((prediction - target).square().reshape(values.shape[0], -1).mean(1), batch)

    def generator_loss(self, batch, *, target_index, generator=None):
        self.generator_calls += 1
        if self.generator_calls == self.fail_generator_call:
            raise RuntimeError("intentional generator failure")
        self.target_indices.append(("generator", target_index))
        self.seen_batches.append(("generator", batch.sample_ids))
        self.seen_fake_weights.append(float(self.fake.weight.detach().item()))
        values = batch.latent_template.float()
        target = self.fake(values).detach() + self._jitter(generator)
        prediction = self.student(values)
        return self._result((prediction - target).square().reshape(values.shape[0], -1).mean(1), batch)


def _engine(*, seed: int, accumulation_steps: int, fail_generator_call=None, use_generator=False):
    torch.manual_seed(seed)
    student = torch.nn.Linear(1, 1, bias=False)
    fake = torch.nn.Linear(1, 1, bias=False)
    teacher = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
    losses = _EngineLosses(
        student,
        fake,
        fail_generator_call=fail_generator_call,
        use_generator=use_generator,
    )
    student_scheduler = _Counter()
    fake_scheduler = _Counter()
    ema = _EMA(student)
    engine = NativeSIDTrainEngine(
        student_module=student,
        teacher_module=teacher,
        fake_score_module=fake,
        loss_adapter=losses,
        student_optimizer=torch.optim.SGD(student.parameters(), lr=0.05),
        fake_score_optimizer=torch.optim.SGD(fake.parameters(), lr=0.05),
        student_max_grad_norm=1000.0,
        fake_score_max_grad_norm=1000.0,
        gradient_accumulation_steps=accumulation_steps,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_scheduler,
        student_ema=ema,
    )
    return engine, losses, student_scheduler, fake_scheduler, ema


def test_sid_engine_uses_one_step_for_both_roles_and_fake_commit_precedes_generator() -> None:
    engine, losses, student_scheduler, fake_scheduler, _ = _engine(seed=23, accumulation_steps=1)
    initial_fake = float(engine.fake_score_module.weight.detach())
    result = engine.train_step(
        _batch((1.0, 2.0), prefix="fake"),
        _batch((3.0, 4.0), prefix="generator"),
        generator=torch.Generator().manual_seed(29),
    )
    assert losses.target_indices == [("fake", result.target_index), ("generator", result.target_index)]
    assert losses.seen_batches == [
        ("fake", ("fake-0", "fake-1")),
        ("generator", ("generator-0", "generator-1")),
    ]
    assert losses.seen_fake_weights[0] != initial_fake
    assert engine.student_optimizer_steps == engine.fake_score_optimizer_steps == 1
    assert student_scheduler.steps == fake_scheduler.steps == 1


def test_sid_poison_prevents_checkpoint_after_fake_score_commit() -> None:
    engine, _, _, _, _ = _engine(
        seed=31,
        accumulation_steps=1,
        fail_generator_call=1,
    )
    before = engine.fake_score_module.weight.detach().clone()
    with pytest.raises(RuntimeError, match="intentional generator failure"):
        engine.train_step(
            _batch((1.0, 2.0), prefix="fake"),
            _batch((3.0, 4.0), prefix="generator"),
        )
    assert not torch.equal(before, engine.fake_score_module.weight.detach())
    with pytest.raises(RuntimeError, match="partially committed"):
        engine.state_dict()


class _StatefulLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        return _batch((float(self.cursor),), prefix=f"cursor-{self.cursor}")

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_stack(seed: int):
    engine, _, student_scheduler, fake_scheduler, ema = _engine(
        seed=seed,
        accumulation_steps=2,
        use_generator=True,
    )
    loader = _StatefulLoader()
    progress = TrainingProgress()
    objective_generator = torch.Generator().manual_seed(101)
    model = torch.nn.ModuleDict(
        {
            "student": engine.student_module,
            "teacher": engine.teacher_module,
            "fake_score": engine.fake_score_module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=(engine.student_optimizer, engine.fake_score_optimizer),
        engine=engine,
        dataloader=loader,
        objective_generator=objective_generator,
        progress=progress,
            identity={
                "algorithm": "sid",
                "gradient_accumulation_steps": engine.gradient_accumulation_steps,
            },
        lr_scheduler=NamedStatefulCollection(
            {"student": student_scheduler, "fake_score": fake_scheduler}
        ),
        ema=NamedStatefulCollection({"student": ema}),
    )
    return (
        engine,
        loader,
        progress,
        objective_generator,
        model,
        state,
        student_scheduler,
        fake_scheduler,
        ema,
    )


def test_sid_dcp_split_resume_restores_rng_both_optimizers_schedulers_and_ema(tmp_path: Path) -> None:
    baseline = _checkpointable_stack(37)
    engine, loader, progress, generator, model, state, student_scheduler, fake_scheduler, ema = baseline
    session = NativeSIDTrainingSession(engine, loader, progress)
    session.run(max_steps=1, generator=generator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    artifact = manager.save(state)
    expected = session.run(max_steps=1, generator=generator)
    expected_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_ema = ema.shadow.clone()
    expected_schedulers = (student_scheduler.steps, fake_scheduler.steps)

    restored = _checkpointable_stack(41)
    (
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
        restored_student_scheduler,
        restored_fake_scheduler,
        restored_ema,
    ) = restored
    manager.load(restored_state, artifact.path)
    actual = NativeSIDTrainingSession(
        restored_engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=1, generator=restored_generator)
    assert restored_loader.cursor == 8
    assert restored_progress.optimizer_steps == 2
    assert actual.final_generator_loss == expected.final_generator_loss
    assert actual.final_fake_score_loss == expected.final_fake_score_loss
    assert (restored_student_scheduler.steps, restored_fake_scheduler.steps) == expected_schedulers
    torch.testing.assert_close(restored_ema.shadow, expected_ema, rtol=0, atol=0)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)


def test_sid_session_reports_crossed_export_boundaries() -> None:
    engine, _, _, _, _ = _engine(seed=43, accumulation_steps=1)
    loader = _StatefulLoader()
    progress = TrainingProgress()
    boundaries: list[tuple[int, int]] = []

    summary = NativeSIDTrainingSession(engine, loader, progress).run(
        max_steps=3,
        boundary_every_steps=2,
        boundary_sink=lambda previous, current: boundaries.append((previous, current)),
    )

    assert summary.final_step == 3
    assert boundaries == [(1, 2)]


def test_sana_sid_run_exposes_cli_state_and_writes_status(tmp_path: Path) -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping())

    class Session:
        def __init__(self) -> None:
            self.progress = TrainingProgress()
            self.engine = SimpleNamespace(global_step=0)

        def run(self, *, max_steps, generator, boundary_every_steps, boundary_sink):
            del generator, boundary_every_steps, boundary_sink
            for _ in range(max_steps):
                self.engine.global_step += 1
                self.progress.record_step(
                    microbatches=2,
                    samples=2,
                    latent_tokens=2,
                )
            return SIDRunSummary(
                initial_step=0,
                final_step=self.engine.global_step,
                iterations=max_steps,
                student_optimizer_steps=max_steps,
                fake_score_optimizer_steps=max_steps,
                final_generator_loss=1.0,
                final_fake_score_loss=0.5,
            )

    session = Session()
    run = SanaSIDTrainingRun(
        recipe=recipe,
        session=session,
        checkpoint_state=SimpleNamespace(objective_generator=None),
        checkpointer=SimpleNamespace(root=tmp_path / "checkpoints"),
        roles=SimpleNamespace(
            asset_identity={
                "student": {"path": "student", "files": {"model.safetensors": 10}},
                "teacher": {"path": "teacher", "files": {"model.safetensors": 11}},
                "fake_score": {"path": "fake-score", "files": {"model.safetensors": 12}},
            }
        ),
        output_dir=tmp_path,
        data_identity={"prompt_records": [{"prompt_id": "prompt-0"}]},
        resume_artifact=None,
        distributed_context=None,
    )

    summary = run.run(max_steps=2)

    assert summary.final_step == 2
    assert run.rank == 0
    assert run.world_size == 1
    assert run.is_coordinator is True
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == SANA_SID_RUN_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["role_assets"]["student"]["path"] == "student"
    assert manifest["data_identity"] == {"prompt_records": [{"prompt_id": "prompt-0"}]}
    assert manifest["resumed_from"] is None
    assert manifest["progress"]["optimizer_steps"] == 2


def test_sana_sid_materializer_builds_independent_roles_and_resume_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping = deepcopy(_recipe_mapping())
    mapping["run"] = {"id": "sana-sid-materializer", "output_dir": str(tmp_path / "run")}
    mapping["model"] = {
        "recipe": "sana-sprint-600m-1024px",
        "checkpoint": "student",
    }
    mapping["data"] = {
        "manifest": str(tmp_path / "prompts.jsonl"),
        "shuffle": False,
        "tail_policy": "drop",
        "options": {
            "height": 64,
            "width": 64,
            "microbatch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "snapshot_every_n_steps": 1,
        },
    }
    mapping["distributed"] = {"backend": "single"}
    recipe = PostTrainingRecipe.from_mapping(mapping)

    role_paths: dict[str, Path] = {}
    for role in ("student", "teacher", "fake_score"):
        root = tmp_path / role
        transformer = root / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}", encoding="utf-8")
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(
            role.encode("utf-8")
        )
        role_paths[role] = root
    (role_paths["student"] / "text_encoder").mkdir()
    (role_paths["student"] / "text_encoder/config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (role_paths["student"] / "text_encoder/model.safetensors").write_bytes(
        b"text-encoder"
    )
    (role_paths["student"] / "tokenizer").mkdir()
    (role_paths["student"] / "tokenizer/tokenizer.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (role_paths["student"] / "model_index.json").write_text(
        "{}",
        encoding="utf-8",
    )

    class Prediction:
        def __init__(self, identity: str) -> None:
            self.module = torch.nn.Linear(1, 1, bias=False)
            self.checkpoint_identity = identity

    predictions: list[Prediction] = []
    adapter_options: list[dict[str, object]] = []

    def build_adapter(path, *, checkpoint_identity, **kwargs):
        del path
        adapter_options.append(dict(kwargs))
        prediction = Prediction(checkpoint_identity)
        predictions.append(prediction)
        return SimpleNamespace(), prediction

    class PromptDataset:
        def __init__(self) -> None:
            self.records = (
                SimpleNamespace(
                    prompt_id="prompt-0",
                    generation={},
                    to_dict=lambda: {"prompt_id": "prompt-0", "generation": {}},
                ),
                SimpleNamespace(
                    prompt_id="prompt-1",
                    generation={},
                    to_dict=lambda: {"prompt_id": "prompt-1", "generation": {}},
                ),
            )

        def __len__(self):
            return len(self.records)

        def __iter__(self):
            return iter(self.records)

    class Sampler:
        def __init__(self, dataset, **kwargs) -> None:
            del kwargs
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

    class StatefulLoader:
        def state_dict(self):
            return {"cursor": 0}

        def load_state_dict(self, state_dict):
            assert "cursor" in state_dict

    class ConvertedLoader:
        def __init__(self, source, **kwargs) -> None:
            del kwargs
            self.source = source

        def state_dict(self):
            return self.source.state_dict()

        def load_state_dict(self, state_dict):
            self.source.load_state_dict(state_dict)

    class Engine:
        global_step = 0

        def state_dict(self):
            return {"global_step": self.global_step}

        def load_state_dict(self, state_dict):
            self.global_step = int(state_dict["global_step"])

    class Stack:
        def __init__(self, student, fake_score) -> None:
            self.engine = Engine()
            self.student_optimizer = torch.optim.AdamW(student.module.parameters())
            self.fake_score_optimizer = torch.optim.AdamW(fake_score.module.parameters())

        @staticmethod
        def checkpoint_state_kwargs():
            return {}

    class Session:
        def __init__(self, engine, dataloader, progress, **kwargs) -> None:
            del kwargs
            self.engine = engine
            self.dataloader = dataloader
            self.progress = progress

    monkeypatch.setattr(
        sana_sid_engine,
        "build_local_diffusers_sana_sid_adapter",
        build_adapter,
    )
    monkeypatch.setattr(
        sana_sid_engine.RolloutPromptDataset,
        "from_file",
        lambda *_args, **_kwargs: PromptDataset(),
    )
    monkeypatch.setattr(sana_sid_engine, "DeterministicDistributedSampler", Sampler)
    monkeypatch.setattr(
        sana_sid_engine,
        "build_stateful_dataloader",
        lambda *_args, **_kwargs: StatefulLoader(),
    )
    monkeypatch.setattr(sana_sid_engine, "SanaSIDDataLoader", ConvertedLoader)
    monkeypatch.setattr(
        sana_sid_engine,
        "build_native_sid_training_stack",
        lambda _recipe, *, student, fake_score, **_kwargs: Stack(student, fake_score),
    )
    monkeypatch.setattr(sana_sid_engine, "NativeSIDTrainingSession", Session)

    run = sana_sid_engine.materialize_sana_sid_training_run(
        recipe,
        base_dir=tmp_path,
        device="cpu",
        local_role_paths=role_paths,
        initialization_seed=17,
    )

    assert run.output_dir == (tmp_path / "run").resolve()
    assert run.world_size == 1
    assert len({id(value.module) for value in predictions}) == 3
    assert [value.get("parameter_dtype") for value in adapter_options] == [
        torch.float32,
        None,
        torch.float32,
    ]
    assert set(run.roles.asset_identity) == {"student", "teacher", "fake_score"}
    assert run.data_identity["kind"] == "prompt-only"
    assert run.checkpoint_state.identity["schema"] == (
        "worldfoundry-sana-sid-resume-identity"
    )
    assert run.checkpoint_state.identity["initialization_seed"] == 17


def test_sid_public_exports_are_lazy_and_resolvable() -> None:
    import worldfoundry.training.engine as engine_public
    import worldfoundry.training.post_training as public

    for name in (
        "SIDConfig",
        "SIDTrainingBatch",
        "NativeSIDTrainEngine",
        "NativeSIDTrainingSession",
        "build_native_sid_training_stack",
    ):
        assert name in public.__all__
        assert getattr(public, name) is not None
    for name in (
        "SANA_SID_RUN_SCHEMA",
        "SanaSIDTrainingRun",
        "materialize_sana_sid_training_run",
    ):
        assert name in engine_public.__all__
        assert getattr(engine_public, name) is not None
