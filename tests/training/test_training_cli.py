from __future__ import annotations

import argparse
import builtins
import json
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.cli.training import register_training_subparser
from worldfoundry.cli.training_commands.common import (
    checkpoint_overrides,
    load_cache_recipe,
    record_elastic_worker_errors,
    training_base_dir,
    training_checkpoint_key,
    training_family,
)


def test_elastic_worker_error_recording_is_lazy_and_reraises(
    monkeypatch,
    tmp_path,
) -> None:
    error_file = tmp_path / "worker-error.json"
    calls: list[str] = []

    def record(function):
        def recorded(*args, **kwargs):
            calls.append(function.__name__)
            return function(*args, **kwargs)

        return recorded

    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.elastic.multiprocessing.errors",
        SimpleNamespace(record=record),
    )

    @record_elastic_worker_errors
    def worker() -> int:
        raise RuntimeError("worker failed")

    monkeypatch.delenv("TORCHELASTIC_ERROR_FILE", raising=False)
    with pytest.raises(RuntimeError, match="worker failed"):
        worker()
    assert calls == []

    monkeypatch.setenv("TORCHELASTIC_ERROR_FILE", str(error_file))
    with pytest.raises(RuntimeError, match="worker failed"):
        worker()
    assert calls == ["worker"]


def test_elastic_worker_error_recording_allows_missing_torch(monkeypatch, tmp_path) -> None:
    real_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name.startswith("torch"):
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(
        sys.modules,
        "torch.distributed.elastic.multiprocessing.errors",
        raising=False,
    )
    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    monkeypatch.setenv("TORCHELASTIC_ERROR_FILE", str(tmp_path / "worker-error.json"))

    @record_elastic_worker_errors
    def worker() -> int:
        return 7

    assert worker() == 7


def test_training_cli_is_split_by_command_and_imports_no_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    command_root = root / "worldfoundry/cli/training_commands"
    expected = {
        command_root / "common.py",
        command_root / "register.py",
        command_root / "handlers/audit.py",
        command_root / "handlers/cache.py",
        command_root / "handlers/post_train.py",
        command_root / "handlers/train.py",
    }

    assert all(path.is_file() for path in expected)
    assert all(len(path.read_text(encoding="utf-8").splitlines()) < 400 for path in expected)
    assert len((root / "worldfoundry/cli/training.py").read_text().splitlines()) < 100

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import worldfoundry.cli.training; "
                "print(json.dumps({"
                "'runtime': sorted(name for name in sys.modules "
                "if name.startswith('worldfoundry.training')), "
                "'handlers': sorted(name for name in sys.modules "
                "if name.startswith('worldfoundry.cli.training_commands.handlers'))}))"
            ),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(probe.stdout) == {"runtime": [], "handlers": []}

    registration_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import argparse, json, sys; "
                "from worldfoundry.cli.training import register_training_subparser; "
                "parser = argparse.ArgumentParser(); "
                "register_training_subparser(parser.add_subparsers()); "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name.startswith('worldfoundry.training'))))"
            ),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(registration_probe.stdout) == []


def test_exclusive_training_output_is_not_created_by_cli_logging(
    monkeypatch,
    tmp_path,
) -> None:
    import worldfoundry.core as core
    from worldfoundry.cli.main import _prepare_cli_run_observability

    for name in (
        "WORLDFOUNDRY_LOG_CONTEXT",
        "WORLDFOUNDRY_LOG_FILE",
        "WORLDFOUNDRY_LOG_JSON",
        "WORLDFOUNDRY_RUN_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(core, "bind_log_context", lambda **_: None)
    monkeypatch.setattr(core, "configure_logging", lambda **_: None)
    monkeypatch.setattr(core, "log_context_environment", lambda: {})
    output_dir = tmp_path / "training-run"
    args = Namespace(
        command="train",
        output_dir=output_dir,
        run_id=None,
        benchmark_id=None,
        model_id="sana-600m-512px",
        _requires_exclusive_output_dir=True,
    )

    state = _prepare_cli_run_observability(args, explicit_log_file=None)

    assert state is not None
    event_path, run_id, _ = state
    assert not output_dir.exists()
    assert event_path == (tmp_path / ".worldfoundry-cli-logs" / "training-run" / run_id / "events.jsonl")
    assert event_path.is_file()


def test_training_cli_exposes_wan_cache_and_attention_gate() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_training_subparser(subparsers)

    train = parser.parse_args(
        [
            "train",
            "--recipe",
            "wan.yaml",
            "--steps",
            "2",
            "--allow-unverified-attention-backend",
            "--allow-environment-drift",
        ]
    )
    cache = parser.parse_args(
        [
            "train-cache",
            "--recipe",
            "wan.yaml",
            "--safety-batch-size",
            "2",
            "--prompt-audits",
            "prompt-audits.json",
            "--checkpoint-override",
            "vae=/models/wan",
        ]
    )
    audit = parser.parse_args(
        [
            "train-audit-prompts",
            "--manifest",
            "raw.jsonl",
            "--output-manifest",
            "audited.jsonl",
            "--output-audits",
            "prompt-audits.json",
        ]
    )
    rollout_audit = parser.parse_args(
        [
            "train-audit-rollout-prompts",
            "--source",
            "raw-prompts.jsonl",
            "--output-manifest",
            "audited-prompts.jsonl",
            "--batch-size",
            "3",
        ]
    )
    post_train = parser.parse_args(
        [
            "post-train",
            "--recipe",
            "wan-dmd.yaml",
            "--steps",
            "3",
            "--no-export-student-adapter",
        ]
    )

    assert train.allow_unverified_attention_backend is True
    assert train.allow_environment_drift is True
    assert train.export_artifact is True
    assert train._requires_exclusive_output_dir is True
    assert train.func.__wrapped__.__name__ == "_handle_train"
    assert cache.safety_batch_size == 2
    assert cache.prompt_audits == Path("prompt-audits.json")
    assert cache.checkpoint_override == ["vae=/models/wan"]
    assert cache.func.__name__ == "_handle_train_cache"
    assert not hasattr(cache.func, "__wrapped__")
    assert audit.func.__name__ == "_handle_train_audit_prompts"
    assert rollout_audit.source == Path("raw-prompts.jsonl")
    assert rollout_audit.batch_size == 3
    assert rollout_audit.func.__name__ == "_handle_train_audit_rollout_prompts"
    assert post_train.func.__name__ == "_handle_post_train"
    assert post_train.func.__wrapped__.__name__ == "_handle_post_train"
    assert post_train.steps == 3
    assert post_train.export_artifact is False
    assert post_train.checkpoint_override is None
    assert post_train.reward_url is None
    assert post_train.reward_attention == "sdpa"
    assert post_train._requires_exclusive_output_dir is True
    assert training_family("wan2.1-t2v-1.3b") == "wan"
    assert training_family("sana-600m-512px") == "sana"
    assert training_family("ltx-2.3-i2v") == "ltx"
    assert training_family("cosmos-predict2.5-2b") == "cosmos"
    assert training_family("cosmos3-nano") == "cosmos"
    assert training_checkpoint_key("sana-600m-512px") == "dit"
    assert training_checkpoint_key("ltx-2-i2v") == "model"
    assert training_checkpoint_key("cosmos3-super") == "transformer"
    with pytest.raises(ValueError, match="does not support"):
        training_family("unsupported-model")


def test_train_exports_a_full_model_for_full_parameter_tuning(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "train",
            "--recipe",
            str(root / "configs/training/lvdm_short_unconditional.yaml"),
            "--base-dir",
            str(root),
            "--output-dir",
            str(tmp_path / "run"),
            "--steps",
            "1",
            "--resume-checkpoint",
            str(tmp_path / "checkpoint"),
            "--allow-environment-drift",
        ]
    )
    called: dict[str, object] = {}

    class Session:
        world_size = 1
        is_coordinator = True
        peft_application = None

        def __init__(self) -> None:
            self.output_dir = tmp_path / "run"

        def run(self, **kwargs):
            called["run"] = kwargs
            return SimpleNamespace(to_dict=lambda: {"optimizer_steps": 1})

        def export_full_model(self):
            called["exported"] = True
            return SimpleNamespace(
                path=self.output_dir / "model",
                file_size_bytes={"model.safetensors": 32},
            )

        def close(self):
            called["closed"] = True

    def materialize(recipe, **kwargs):
        called["recipe"] = recipe
        called["materialize"] = kwargs
        return Session()

    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.training.engine.lvdm.sft",
        SimpleNamespace(materialize_lvdm_short_training_session=materialize),
    )

    assert args.func(args) == 0
    assert called["exported"] is True
    assert called["closed"] is True
    assert called["run"]["allow_environment_drift"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["full_model"]["status"] == "exported"


def test_train_environment_drift_flag_requires_resume_checkpoint() -> None:
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "train",
            "--recipe",
            "missing.yaml",
            "--steps",
            "1",
            "--allow-environment-drift",
        ]
    )

    with pytest.raises(ValueError, match="requires --resume-checkpoint"):
        args.func(args)


def test_train_cache_loads_the_native_post_training_recipe() -> None:
    root = Path(__file__).resolve().parents[2]

    recipe = load_cache_recipe(root / "configs/post_training/wan_1p3b_dmd.yaml")

    assert type(recipe).__name__ == "PostTrainingRecipe"
    assert recipe.execution_owner == "worldfoundry-native"
    assert recipe.algorithm.type == "dmd"


def test_sana_cache_accepts_the_native_sid_post_training_recipe() -> None:
    from worldfoundry.training.data.sana_precompute import _validate_sana_cache_recipe

    root = Path(__file__).resolve().parents[2]
    recipe = load_cache_recipe(root / "configs/post_training/sana_sprint_600m_sid.yaml")

    assert _validate_sana_cache_recipe(recipe) is recipe
    assert recipe.algorithm.type == "sid"


def test_training_cli_checkpoint_overrides_are_named_and_unambiguous(tmp_path) -> None:
    overrides = checkpoint_overrides(
        ["dit=weights", "tokenizer=/models/tokenizer"],
        base_dir=tmp_path,
    )

    assert overrides == {
        "dit": str((tmp_path / "weights").resolve()),
        "tokenizer": "/models/tokenizer",
    }
    with pytest.raises(ValueError, match="NAME=PATH"):
        checkpoint_overrides(["weights"], base_dir=tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        checkpoint_overrides(["dit=a", "dit=b"], base_dir=tmp_path)


def test_training_base_dir_defaults_to_launch_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    assert training_base_dir(None) == tmp_path.resolve()
    assert training_base_dir(Path("assets")) == (tmp_path / "assets").resolve()


def test_post_train_materializes_sana_sid_and_exports_student(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    output_dir = tmp_path / "run"
    args = parser.parse_args(
        [
            "post-train",
            "--recipe",
            str(root / "configs/post_training/sana_sprint_600m_sid.yaml"),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--checkpoint-override",
            "student=models/student",
            "--checkpoint-override",
            "teacher=models/teacher",
            "--checkpoint-override",
            "fake_score=models/fake-score",
        ]
    )
    called: dict[str, object] = {}

    @dataclass(frozen=True)
    class Summary:
        initial_step: int = 0
        final_step: int = 2
        iterations: int = 2
        student_optimizer_steps: int = 2
        fake_score_optimizer_steps: int = 2
        final_generator_loss: float = 1.25
        final_fake_score_loss: float = 0.75

    class Run:
        world_size = 1
        is_coordinator = True

        def __init__(self) -> None:
            self.output_dir = output_dir

        def run(self, *, max_steps):
            called["max_steps"] = max_steps
            return Summary()

        def export_student(self):
            called["exported"] = True
            return SimpleNamespace(
                path=output_dir / "exports/student",
                file_size_bytes={"model.safetensors": 1024},
            )

        def close(self):
            called["closed"] = True

    def materialize(recipe, **kwargs):
        called["recipe"] = recipe
        called.update(kwargs)
        return Run()

    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.training.engine.sana.sid",
        SimpleNamespace(materialize_sana_sid_training_run=materialize),
    )

    assert args.func(args) == 0
    assert called["recipe"].algorithm.type == "sid"
    assert called["max_steps"] == 2
    assert called["local_role_paths"] == {
        "student": str((tmp_path / "models/student").resolve()),
        "teacher": str((tmp_path / "models/teacher").resolve()),
        "fake_score": str((tmp_path / "models/fake-score").resolve()),
    }
    assert called["exported"] is True
    assert called["closed"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm"] == "sid"
    assert payload["summary"]["final_step"] == 2
    assert payload["trained_artifact"]["role"] == "student"


def test_post_train_rejects_role_overrides_for_unsupported_algorithms(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "post-train",
            "--recipe",
            str(root / "configs/post_training/wan_1p3b_dmd.yaml"),
            "--base-dir",
            str(tmp_path),
            "--steps",
            "1",
            "--checkpoint-override",
            "student=models/student",
        ]
    )

    with pytest.raises(ValueError, match="require SANA SiD"):
        args.func(args)


@pytest.mark.parametrize(
    ("config_name", "model_recipe", "override", "override_key"),
    (
        ("ltx_2_video_flow_grpo.yaml", "ltx-2-i2v", "model=models/ltx", "model"),
        (
            "ltx_2p3_av_ray_flow_grpo.yaml",
            "ltx-2.3-i2v",
            "model=models/ltx",
            "model",
        ),
        (
            "wan22_t2v_a14b_flow_grpo.yaml",
            "wan2.2-t2v-a14b",
            "high-noise=models/wan-high",
            "high-noise",
        ),
        (
            "wan22_t2v_a14b_ray_flow_grpo.yaml",
            "wan2.2-t2v-a14b",
            "high-noise=models/wan-high",
            "high-noise",
        ),
        (
            "hunyuan_video_flow_grpo.yaml",
            "hunyuanvideo-t2v",
            "policy=models/hunyuan",
            "policy",
        ),
    ),
)
def test_post_train_routes_integrated_video_flow_policy_runs(
    monkeypatch,
    tmp_path,
    capsys,
    config_name,
    model_recipe,
    override,
    override_key,
) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    output_dir = tmp_path / "run"
    args = parser.parse_args(
        [
            "post-train",
            "--recipe",
            str(root / "configs/post_training" / config_name),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--reward-device",
            "cpu",
            "--reward-url",
            "http://reward.internal:9000",
            "--reward-attention",
            "flash_attention_2",
            "--resume-checkpoint",
            "checkpoints/step-00000010",
            "--checkpoint-override",
            override,
            "--seed",
            "73",
            "--steps",
            "2",
        ]
    )
    called: dict[str, object] = {}

    class Run:
        world_size = 3
        is_coordinator = True

        def __init__(self) -> None:
            self.output_dir = output_dir

        def run(self, *, max_iterations):
            called["max_iterations"] = max_iterations
            return SimpleNamespace(
                to_dict=lambda: {
                    "final_optimizer_step": 2,
                    "rollout_iterations": max_iterations,
                }
            )

        def export_policy(self):
            called["exported"] = True
            return SimpleNamespace(
                path=output_dir / "exports/policy",
                file_size_bytes={"adapter_model.safetensors": 64},
            )

        def close(self):
            called["closed"] = True

    def materialize(recipe, **kwargs):
        called["recipe"] = recipe
        called.update(kwargs)
        return Run()

    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.training.engine.video_policy",
        SimpleNamespace(materialize_video_flow_policy_training_run=materialize),
    )

    assert args.func(args) == 0
    assert called["recipe"].model.recipe == model_recipe
    assert called["base_dir"] == tmp_path.resolve()
    assert called["output_dir"] == output_dir
    assert called["device"] == "cpu"
    assert called["reward_device"] == "cpu"
    assert called["reward_url"] == "http://reward.internal:9000"
    assert called["reward_attention_implementation"] == "flash_attention_2"
    assert called["resume_checkpoint"] == (tmp_path / "checkpoints/step-00000010").resolve()
    assert called["checkpoint_overrides"] == {override_key: str((tmp_path / override.partition("=")[2]).resolve())}
    assert called["initialization_seed"] == 73
    assert called["max_iterations"] == 2
    assert called["exported"] is True
    assert called["closed"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm"] == "flow-grpo"
    assert payload["rank_count"] == 3
    assert payload["trained_artifact"]["role"] == "policy"


def test_post_train_keeps_wan21_flow_policy_on_its_native_materializer(
    monkeypatch,
    tmp_path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "post-train",
            "--recipe",
            str(root / "configs/post_training/wan_1p3b_flow_grpo.yaml"),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--no-export-trained-artifact",
        ]
    )
    called: dict[str, object] = {}

    class Run:
        world_size = 1
        is_coordinator = False
        output_dir = tmp_path / "run"

        def run(self, *, max_iterations):
            called["max_iterations"] = max_iterations
            return SimpleNamespace(to_dict=lambda: {"rollout_iterations": max_iterations})

        def close(self):
            called["closed"] = True

    def materialize(recipe, **kwargs):
        called["recipe"] = recipe
        called.update(kwargs)
        return Run()

    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.training.engine.wan.flow_policy",
        SimpleNamespace(materialize_wan_flow_policy_training_run=materialize),
    )

    assert args.func(args) == 0
    assert called["recipe"].model.recipe == "wan2.1-t2v-1.3b"
    assert called["max_iterations"] == 1
    assert called["force_torch_attention"] is True
    assert called["closed"] is True


def test_post_train_rejects_http_reward_for_wan21_flow_policy(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "post-train",
            "--recipe",
            str(root / "configs/post_training/wan_1p3b_flow_grpo.yaml"),
            "--base-dir",
            str(tmp_path),
            "--steps",
            "1",
            "--reward-url",
            "http://reward.internal:9000",
        ]
    )

    with pytest.raises(ValueError, match="LTX, Wan2.2, or HunyuanVideo"):
        args.func(args)


def test_train_cache_routes_diffusion_nft_to_rollout_conditioning(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    from worldfoundry.training.data.wan import rollout_cache

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "train-cache",
            "--recipe",
            str(root / "configs/post_training/wan_1p3b_diffusion_nft.yaml"),
            "--base-dir",
            str(root),
            "--manifest",
            str(tmp_path / "prompts.jsonl"),
            "--cache",
            str(tmp_path / "conditioning"),
            "--device",
            "cpu",
        ]
    )
    called: dict[str, object] = {}

    def materialize(recipe, **kwargs):
        called["algorithm"] = recipe.algorithm.type
        called.update(kwargs)
        return SimpleNamespace(
            index=SimpleNamespace(to_dict=lambda: {"entries": []}),
            entries=(),
            unconditional_conditioning=None,
        )

    monkeypatch.setattr(
        rollout_cache,
        "materialize_wan_rollout_conditioning_cache",
        materialize,
    )

    assert args.func(args) == 0
    assert called["algorithm"] == "diffusion-nft"
    assert called["manifest_path"] == (tmp_path / "prompts.jsonl").resolve()
    assert called["cache_dir"] == (tmp_path / "conditioning").resolve()
    assert json.loads(capsys.readouterr().out)["prompt_count"] == 0
