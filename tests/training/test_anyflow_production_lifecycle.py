from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.cli.training import register_training_subparser
from worldfoundry.training.distributed.parallel import ParallelPlan
from worldfoundry.training.recipes.post_training.algorithms.anyflow import (
    AnyFlowBidirectionalOnPolicyAlgorithmSpec,
    AnyFlowBidirectionalPretrainAlgorithmSpec,
    AnyFlowFAROnPolicyAlgorithmSpec,
    AnyFlowFARPretrainAlgorithmSpec,
)
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

ROOT = Path(__file__).resolve().parents[2]
CONFIG_CASES = (
    (
        "anyflow_far_pretrain_1p3b.yaml",
        AnyFlowFARPretrainAlgorithmSpec,
    ),
    (
        "anyflow_bidirectional_pretrain_1p3b.yaml",
        AnyFlowBidirectionalPretrainAlgorithmSpec,
    ),
    (
        "anyflow_far_on_policy_1p3b.yaml",
        AnyFlowFAROnPolicyAlgorithmSpec,
    ),
    (
        "anyflow_bidirectional_on_policy_1p3b.yaml",
        AnyFlowBidirectionalOnPolicyAlgorithmSpec,
    ),
)


@pytest.mark.parametrize(("filename", "algorithm_type"), CONFIG_CASES)
def test_anyflow_production_configs_parse_strictly(
    filename: str,
    algorithm_type: type,
) -> None:
    recipe = PostTrainingRecipe.from_file(ROOT / "configs/post_training" / filename)

    assert isinstance(recipe.algorithm, algorithm_type)
    assert recipe.model.recipe == "wan2.1-t2v-1.3b"
    assert recipe.tuning.mode == "full"
    assert recipe.data.max_latent_tokens_per_microbatch == 2 * 21 * 60 * 104
    assert recipe.runtime.param_dtype == "bfloat16"
    assert recipe.runtime.activation_checkpoint == "full"
    assert recipe.distributed.backend == "single"
    assert recipe.export.format == "safetensors"


def test_anyflow_recipe_can_resolve_ddp_at_the_launched_world_size() -> None:
    recipe = PostTrainingRecipe.from_file(
        ROOT / "configs/post_training/anyflow_far_pretrain_1p3b.yaml"
    )
    distributed = replace(
        recipe.distributed,
        backend="ddp",
        dp_replicate=3,
    )

    plan = ParallelPlan.resolve(distributed, world_size=3)

    assert plan.backend == "ddp"
    assert plan.data_parallel_size == 3
    assert plan.mesh_shape == (3, 1, 1, 1)


def test_post_train_dispatches_anyflow_run_and_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import worldfoundry.training.engine.anyflow as anyflow_engine

    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    output_dir = tmp_path / "run"
    args = parser.parse_args(
        [
            "post-train",
            "--recipe",
            str(
                ROOT
                / "configs/post_training/anyflow_bidirectional_pretrain_1p3b.yaml"
            ),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--steps",
            "3",
            "--seed",
            "0",
        ]
    )
    called: dict[str, object] = {}

    @dataclass(frozen=True)
    class Summary:
        initial_step: int = 0
        final_step: int = 3
        optimizer_steps: int = 3
        final_loss: float = 0.25

    class Run:
        world_size = 1
        is_coordinator = True

        def __init__(self) -> None:
            self.output_dir = output_dir

        def run(self, *, max_steps: int) -> Summary:
            called["max_steps"] = max_steps
            return Summary()

        def export_student(self) -> SimpleNamespace:
            called["exported"] = True
            return SimpleNamespace(
                path=output_dir / "exports/student",
                file_size_bytes={"model.safetensors": 128},
            )

        def close(self) -> None:
            called["closed"] = True

    def materialize(recipe: PostTrainingRecipe, **kwargs: object) -> Run:
        called["recipe"] = recipe
        called.update(kwargs)
        return Run()

    monkeypatch.setattr(
        anyflow_engine,
        "materialize_anyflow_training_run",
        materialize,
    )

    assert args.func(args) == 0
    assert isinstance(
        called["recipe"].algorithm,
        AnyFlowBidirectionalPretrainAlgorithmSpec,
    )
    assert called["max_steps"] == 3
    assert called["device"] == "cpu"
    assert called["initialization_seed"] == 0
    assert called["exported"] is True
    assert called["closed"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm"] == "anyflow-bidirectional-pretrain"
    assert payload["summary"]["final_step"] == 3
    assert payload["trained_artifact"]["role"] == "student"


def test_anyflow_pretrain_materializer_wires_native_training_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import worldfoundry.training.engine.anyflow.materialize as materialize_module
    from worldfoundry.training.models.anyflow import (
        ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
    )
    from worldfoundry.training.post_training.distillation.anyflow.session import (
        NativeAnyFlowPretrainingSession,
    )
    from worldfoundry.training.post_training.shared.role_checkpoints import (
        resolve_role_checkpoint,
    )

    recipe = PostTrainingRecipe.from_file(
        ROOT / "configs/post_training/anyflow_bidirectional_pretrain_1p3b.yaml"
    )
    output_dir = tmp_path / "run"
    calls: dict[str, object] = {}

    class Manifest:
        sample_ids = ("sample-a", "sample-b")

        @classmethod
        def from_file(cls, path: Path, **kwargs: object) -> Manifest:
            calls["manifest"] = (path, kwargs)
            return cls()

    def descriptor(shape: tuple[int, ...]) -> SimpleNamespace:
        return SimpleNamespace(shape=shape)

    entry = SimpleNamespace(
        tensors={
            "clean_latents": descriptor((16, 21, 60, 104)),
            "condition.context": descriptor((512, 4096)),
        },
        provenance=SimpleNamespace(model_recipe="wan2.1-t2v-1.3b"),
    )

    class Cache:
        sample_ids = Manifest.sample_ids
        index = SimpleNamespace(entries=(entry,))

        def __init__(self, path: Path, **kwargs: object) -> None:
            calls["cache"] = (path, kwargs)

    class ConditioningIdentity:
        model_recipe = "wan2.1-t2v-1.3b"

        def to_dict(self) -> dict[str, object]:
            return {
                "branch": "unconditional",
                "model_recipe": self.model_recipe,
            }

    unconditional = SimpleNamespace(
        artifact=SimpleNamespace(identity=ConditioningIdentity()),
        tensors={"context": torch.zeros(1, 4096)},
    )

    class ConditioningStore:
        def __init__(self, path: Path) -> None:
            calls["conditioning_path"] = path

        def read(self, branch: str) -> object:
            calls["conditioning_branch"] = branch
            return unconditional

    class SourceLoader:
        def __iter__(self):
            return iter(())

        def state_dict(self) -> dict[str, object]:
            return {"cursor": 0}

        def load_state_dict(self, state: object) -> None:
            calls["source_state"] = state

    source_loader = SourceLoader()
    token_sampler = SimpleNamespace(max_latent_tokens=262080)

    def build_loader(**kwargs: object) -> tuple[SourceLoader, object]:
        calls["loader"] = kwargs
        return source_loader, token_sampler

    model = nn.Linear(2, 2)
    local_checkpoint = CheckpointSpec(
        source=tmp_path / "student",
        files=("transformer/diffusion_pytorch_model.safetensors",),
    )
    physical_checkpoint = resolve_role_checkpoint(
        role="student",
        reference="default",
        native_default=ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT,
        local_override=local_checkpoint,
    )
    student = SimpleNamespace(module=model, checkpoint_identity="default")

    class Roles:
        real_score = None
        fake_score = None

        def __init__(self) -> None:
            self.student = student

        def trainable_model(self) -> nn.Module:
            return model

        def checkpoint_identity(self) -> dict[str, object]:
            return {"student": physical_checkpoint.to_dict()}

    roles = Roles()

    def build_roles(*args: object, **kwargs: object) -> Roles:
        calls["roles"] = (args, kwargs)
        return roles

    class Engine:
        global_step = 0
        optimizer_steps = 0
        gradient_accumulation_steps = 1

        def train_step(self, batch: object) -> object:
            del batch
            return SimpleNamespace(loss=torch.zeros(()))

        def state_dict(self) -> dict[str, object]:
            return {"global_step": self.global_step}

        def load_state_dict(self, state: object) -> None:
            calls["engine_state"] = state

    engine = Engine()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

    class Stack:
        def __init__(self) -> None:
            self.engine = engine
            self.optimizer = optimizer
            from worldfoundry.training.post_training.distillation.anyflow.ema import (
                AnyFlowEMA,
            )

            self.ema = AnyFlowEMA(model, decay=0.999, warmup_steps=1000)

        def checkpoint_state_kwargs(self) -> dict[str, object | None]:
            return {
                "lr_scheduler": None,
                "ema": None,
                "algorithm_state": None,
            }

    stack = Stack()

    def build_stack(*args: object, **kwargs: object) -> Stack:
        calls["stack"] = (args, kwargs)
        return stack

    class Checkpointer:
        def __init__(self, root: Path) -> None:
            self.root = root
            calls["checkpointer_root"] = root

    def create_run_directory(path: Path, context: object) -> None:
        calls["run_directory"] = (path, context)
        path.mkdir(parents=True)

    monkeypatch.setattr(materialize_module, "TrainingManifestDataset", Manifest)
    monkeypatch.setattr(materialize_module, "VideoCachedDataset", Cache)
    monkeypatch.setattr(materialize_module, "SharedConditioningStore", ConditioningStore)
    monkeypatch.setattr(materialize_module, "build_wan_cache_loader", build_loader)
    monkeypatch.setattr(materialize_module, "materialize_anyflow_roles", build_roles)
    monkeypatch.setattr(
        materialize_module,
        "build_native_anyflow_pretraining_stack",
        build_stack,
    )
    monkeypatch.setattr(materialize_module, "TrainingCheckpointer", Checkpointer)
    monkeypatch.setattr(materialize_module, "create_run_directory", create_run_directory)

    run = materialize_module.materialize_anyflow_training_run(
        recipe,
        base_dir=tmp_path,
        device="cpu",
        output_dir=output_dir,
        checkpoint_overrides={"student": local_checkpoint},
        initialization_seed=0,
    )

    assert isinstance(run.session, NativeAnyFlowPretrainingSession)
    assert run.session.engine is engine
    assert run.session.dataloader is run.checkpoint_state.dataloader
    assert run.checkpoint_state.model is model
    assert run.checkpoint_state.optimizer is optimizer
    assert run.checkpoint_state.objective_generator.initial_seed() == 0
    assert run.checkpoint_state.identity["initialization_seed"] == 0
    assert run.checkpoint_state.identity["data"]["sample_ids"] == [
        "sample-a",
        "sample-b",
    ]
    role_identity = run.checkpoint_state.identity["roles"]["student"]
    assert student.checkpoint_identity == "default"
    assert role_identity["requested_reference"] == "local"
    assert role_identity["source_kind"] == "local"
    assert calls["checkpointer_root"] == output_dir / "checkpoints"
    assert calls["conditioning_branch"] == "unconditional"
    assert calls["loader"]["rank"] == 0
    assert calls["loader"]["world_size"] == 1
    assert calls["stack"][1]["student"] is student
    assert calls["roles"][1]["checkpoint_overrides"] == {
        "student": local_checkpoint
    }


def test_anyflow_close_releases_distributed_context_when_wait_fails(
    tmp_path: Path,
) -> None:
    from worldfoundry.training.engine.anyflow.run import AnyFlowTrainingRun

    recipe = PostTrainingRecipe.from_file(
        ROOT / "configs/post_training/anyflow_far_pretrain_1p3b.yaml"
    )
    closed: list[bool] = []

    class Session:
        def wait_for_checkpoints(self) -> None:
            raise RuntimeError("checkpoint failed")

    context = SimpleNamespace(
        rank=0,
        world_size=2,
        close=lambda: closed.append(True),
    )
    run = AnyFlowTrainingRun(
        recipe=recipe,
        session=Session(),
        checkpoint_state=SimpleNamespace(),
        checkpointer=SimpleNamespace(),
        roles=SimpleNamespace(),
        student_ema=SimpleNamespace(),
        output_dir=tmp_path,
        resume_artifact=None,
        distributed_context=context,
    )

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        run.close()

    assert closed == [True]


def test_anyflow_full_model_export_uses_ema_and_restores_live_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import worldfoundry.training.engine.anyflow.run as run_module
    from worldfoundry.training.engine.anyflow.run import AnyFlowTrainingRun
    from worldfoundry.training.post_training.distillation.anyflow.ema import (
        AnyFlowEMA,
    )

    recipe = PostTrainingRecipe.from_file(
        ROOT / "configs/post_training/anyflow_bidirectional_pretrain_1p3b.yaml"
    )
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    ema = AnyFlowEMA(model, decay=0.5, warmup_steps=0)
    model.weight.data.fill_(2.0)
    ema.update(model)
    model.weight.data.fill_(3.0)
    exported: list[float] = []

    def export(model: nn.Module, destination: Path, **kwargs: object) -> object:
        del kwargs
        exported.append(float(model.weight.item()))
        return SimpleNamespace(path=destination, file_size_bytes={})

    monkeypatch.setattr(run_module, "export_full_model", export)
    session = SimpleNamespace(
        engine=SimpleNamespace(global_step=1),
        wait_for_checkpoints=lambda: None,
    )
    run = AnyFlowTrainingRun(
        recipe=recipe,
        session=session,
        checkpoint_state=SimpleNamespace(),
        checkpointer=SimpleNamespace(),
        roles=SimpleNamespace(student=SimpleNamespace(module=model)),
        student_ema=ema,
        output_dir=tmp_path,
        resume_artifact=None,
        distributed_context=None,
    )
    run._summary = SimpleNamespace()

    run.export_student(tmp_path / "ema-student")

    assert exported == [1.5]
    assert model.weight.item() == pytest.approx(3.0)
