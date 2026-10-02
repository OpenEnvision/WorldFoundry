from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.cli.training import register_training_subparser
from worldfoundry.cli.training_commands.handlers import qwen as qwen_cli
from worldfoundry.training.post_training.agentic import (
    AgenticPrompt,
    AgentMessage,
    CausalLMAgenticPolicyAdapter,
    CausalLMGenerationConfig,
)
from worldfoundry.training.post_training.causal_lm.qwen3 import (
    Qwen3ActorCritic,
    Qwen3ChatCodec,
    materialize_qwen3_agentic_training_run,
    materialize_qwen3_post_training_run,
    materialize_qwen3_token_ppo_training_run,
    parse_qwen3_hermes_response,
)
from worldfoundry.training.post_training.causal_lm.qwen3 import materializer as qwen3_materializer
from worldfoundry.training.post_training.rl.algorithms.token_ppo import (
    NativeTokenPPODataLoader,
    TokenPPOSample,
)
from worldfoundry.training.recipes import PostTrainingRecipe
from worldfoundry.training.state_comparison import assert_state_equal, snapshot_state
from worldfoundry.training.tuning.full_model import FullModelArtifact, load_full_model

ROOT = Path(__file__).resolve().parents[2]
AGENTIC_CONFIG = ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo.yaml"
PPO_CONFIG = ROOT / "configs/post_training/qwen3_4b_token_ppo.yaml"
RAY_CONFIG = ROOT / "configs/post_training/qwen3_4b_agentic_token_grpo_ray.yaml"
RAY_CPU_FIXTURE = ROOT / "tests/training/fixtures/qwen3_agentic_ray_cpu.yaml"


class _TinyTokenizer:
    eos_token_id = 6
    eos_token = "<|endoftext|>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>"
        return 7

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append({"messages": messages, **kwargs})
        prompt_token = 2 if any(message["role"] == "tool" for message in messages) else 1
        return {
            "input_ids": torch.tensor([[prompt_token]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 1), dtype=torch.int64),
        }

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        values = list(token_ids)
        if values == [3, 4]:
            return '<tool_call>\n{"name":"calculator","arguments":{"expression":"2+3"}}\n</tool_call>'
        if values == [3, 4, 7]:
            return (
                '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'
                '<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>'
                "<|im_end|>"
            )
        if values == [5, 7]:
            return "<answer>5</answer>" if skip_special_tokens else "<answer>5</answer><|im_end|>"
        return ""


class _TinyQwenPolicy(nn.Module):
    def __init__(self, *, agentic: bool, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, use_cache=True)
        self.transitions = nn.Parameter(torch.full((8, 8), -30.0, dtype=dtype))
        self.embedding = nn.Embedding(8, 4, dtype=dtype)
        self.gradient_checkpointing = False
        self.forward_compute_dtypes: list[torch.dtype | None] = []
        with torch.no_grad():
            if agentic:
                self.transitions[1, 3] = 30.0
                self.transitions[3, 4] = 30.0
                self.transitions[4, 7] = 30.0
                self.transitions[2, 5] = 30.0
            else:
                self.transitions[1, 5] = 30.0
            self.transitions[5, 7] = 30.0

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        output_hidden_states=False,
    ):
        del attention_mask, use_cache
        device_type = input_ids.device.type
        self.forward_compute_dtypes.append(
            torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else None,
        )
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            logits=self.transitions[input_ids] + hidden.sum(dim=-1, keepdim=True) * 0.0,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


def _agentic_recipe(
    output_dir: Path,
    *,
    algorithm_type: str = "token-grpo",
) -> PostTrainingRecipe:
    algorithm: dict[str, object] = {
        "type": algorithm_type,
        "reward_weights": {"correctness": 1.0, "tool-success": 0.1},
        "group_size": 2,
        "sampling_temperature": 1.0,
    }
    if algorithm_type in {"token-grpo", "token-gspo"}:
        algorithm["clip_range"] = 0.2
    if algorithm_type != "token-gspo":
        algorithm["horizon"] = 4
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "qwen-agentic-test", "output_dir": str(output_dir)},
            "model": {
                "recipe": "qwen3-tiny-instruct",
                "checkpoint": "tiny-qwen",
                "options": {
                    "enable_thinking": False,
                    "max_new_tokens": 4,
                },
            },
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused.jsonl",
                "shuffle": False,
                "tail_policy": "uneven",
                "options": {"groups_per_batch": 1, "max_turns": 2},
            },
            "algorithm": algorithm,
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
            "runtime": {"param_dtype": "bfloat16", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "checkpoint": {"save_every_steps": 1, "async": False},
            "export": {"format": "safetensors"},
        }
    )


def _ppo_recipe(output_dir: Path) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": "qwen-ppo-test", "output_dir": str(output_dir)},
            "model": {
                "recipe": "qwen3-tiny-base",
                "checkpoint": "tiny-qwen",
                "options": {
                    "enable_thinking": True,
                    "max_new_tokens": 2,
                },
            },
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused.jsonl",
                "shuffle": False,
                "tail_policy": "pad",
                "options": {"batch_size": 4},
            },
            "algorithm": {
                "type": "token-ppo",
                "reward_weights": {"correctness": 1.0},
                "update_epochs": 1,
                "update_partitions": 4,
                "clip_range": 0.2,
                "value_clip_range": 0.2,
                "vf_coef": 0.5,
                "gamma": 1.0,
                "gae_lambda": 0.95,
                "reduction": "seq-mean-token-mean",
                "horizon": 2,
                "sampling_temperature": 1.0,
                "replay_microbatch_size": 1,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
            "runtime": {"param_dtype": "bfloat16", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "checkpoint": {"save_every_steps": 4, "async": False},
            "export": {"format": "safetensors"},
        }
    )


def _prompts() -> tuple[AgenticPrompt, ...]:
    return (
        AgenticPrompt(
            prompt_id="arithmetic",
            messages=(AgentMessage(role="user", content="Use the calculator for 2+3."),),
            conditioning={"answer": "5"},
        ),
    )


def _ppo_samples() -> tuple[TokenPPOSample, ...]:
    messages = (AgentMessage(role="user", content="What is 2+3?"),)
    return tuple(
        TokenPPOSample(
            sample_id=f"arithmetic-{index}",
            conditioning={"messages": messages, "answer": "5"},
        )
        for index in range(4)
    )


def _floating_state_dtypes(module: nn.Module) -> set[torch.dtype]:
    return {tensor.dtype for tensor in module.state_dict().values() if tensor.is_floating_point()}


def _optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> set[torch.dtype]:
    return {
        value.dtype
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }


def test_qwen3_hermes_parser_supports_parallel_object_arguments() -> None:
    message = parse_qwen3_hermes_response(
        "analysis\n"
        '<tool_call>{"name":"search","arguments":{"query":"qwen"}}</tool_call>'
        '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'
        "<|im_end|>"
    )
    assert message.content == "analysis"
    assert [call.name for call in message.tool_calls] == ["search", "calculator"]
    assert dict(message.tool_calls[1].arguments) == {"expression": "2+3"}


def test_qwen3_generation_reaches_im_end_after_parallel_tool_calls() -> None:
    tokenizer = _TinyTokenizer()
    codec = Qwen3ChatCodec(tokenizer, enable_thinking=False)
    policy = _TinyQwenPolicy(agentic=True)
    adapter = CausalLMAgenticPolicyAdapter(
        policy,
        codec,
        generation=CausalLMGenerationConfig(max_new_tokens=4),
        compute_dtype=torch.bfloat16,
    )

    turn = adapter.generate_turn(
        sample_id="parallel-tools",
        messages=(AgentMessage(role="user", content="Use both calculations."),),
        policy_revision="tiny-qwen",
        sampling_temperature=1.0,
        rollout_index=0,
        turn_index=0,
        conditioning={},
        generator=torch.Generator().manual_seed(7),
    )

    assert tokenizer.eos_token_id != tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert codec.eos_token_ids == (6, 7)
    assert turn.token_ids.tolist() == [3, 4, 7]
    assert turn.old_log_probs.dtype is torch.float32
    assert set(policy.forward_compute_dtypes) == {torch.bfloat16}
    assert [call.name for call in turn.message.tool_calls] == ["calculator", "calculator"]
    assert turn.finish_reason == "tool_calls"


def test_qwen3_agentic_materializer_runs_two_turns_resumes_and_exports(tmp_path: Path) -> None:
    output_dir = tmp_path / "agentic"
    recipe = _agentic_recipe(output_dir)
    tokenizer = _TinyTokenizer()
    policy = _TinyQwenPolicy(agentic=True, dtype=torch.bfloat16)
    first = materialize_qwen3_agentic_training_run(
        recipe,
        policy_module=policy,
        tokenizer=tokenizer,
        prompts=_prompts(),
        output_dir=output_dir,
        fused_adamw=False,
    )
    assert _floating_state_dtypes(policy) == {torch.float32}
    assert first.rollout_adapter.model_adapter.compute_dtype is torch.bfloat16
    summary = first.run(max_iterations=1)
    assert summary.final_optimizer_step == 1
    assert first.artifact_role == "policy"
    assert first.rollout_adapter.model_adapter.generation.max_new_tokens == 4
    assert all(call["enable_thinking"] is False for call in tokenizer.template_calls)
    assert all(call["tools"][0]["function"]["name"] == "calculator" for call in tokenizer.template_calls)
    assert set(policy.forward_compute_dtypes) == {torch.bfloat16}
    assert _optimizer_state_dtypes(first.checkpoint_state.optimizer) == {torch.float32}

    resumed = materialize_qwen3_agentic_training_run(
        recipe,
        policy_module=_TinyQwenPolicy(agentic=True),
        tokenizer=_TinyTokenizer(),
        prompts=_prompts(),
        output_dir=output_dir,
        resume_checkpoint="latest",
        fused_adamw=False,
    )
    resumed_summary = resumed.run(max_iterations=1)
    assert resumed_summary.initial_optimizer_step == 1
    assert resumed_summary.final_optimizer_step == 2
    artifact = resumed.export_policy()
    assert isinstance(artifact, FullModelArtifact)
    assert artifact.path.is_dir()


@pytest.mark.parametrize(
    "algorithm_type",
    ("token-grpo", "token-gspo", "token-dppo", "token-drpo", "token-cppo"),
)
def test_qwen3_grouped_token_algorithms_share_agentic_update_surface(
    tmp_path: Path,
    algorithm_type: str,
) -> None:
    output_dir = tmp_path / algorithm_type
    run = materialize_qwen3_post_training_run(
        _agentic_recipe(output_dir, algorithm_type=algorithm_type),
        policy_module=_TinyQwenPolicy(agentic=True),
        tokenizer=_TinyTokenizer(),
        prompts=_prompts(),
        output_dir=output_dir,
        fused_adamw=False,
    )
    summary = run.run(max_iterations=1)
    assert summary.final_optimizer_step == 1
    assert run.engine.algorithm.name == algorithm_type


def test_qwen3_token_ppo_materializer_updates_value_head_resumes_and_exports(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo"
    recipe = _ppo_recipe(output_dir)
    policy = _TinyQwenPolicy(agentic=False, dtype=torch.bfloat16)
    first_actor_critic = Qwen3ActorCritic(policy)
    first = materialize_qwen3_token_ppo_training_run(
        recipe,
        actor_critic=first_actor_critic,
        tokenizer=_TinyTokenizer(),
        samples=_ppo_samples(),
        output_dir=output_dir,
        fused_adamw=False,
    )
    assert _floating_state_dtypes(first_actor_critic) == {torch.float32}
    assert first.session.rollout_adapter.compute_dtype is torch.bfloat16
    assert torch.count_nonzero(first_actor_critic.value_head.weight).item() == 0
    summary = first.run(max_iterations=1)
    assert summary.final_optimizer_step == 4
    assert first.artifact_role == "actor-critic"
    assert first.engine.update_epochs == 1
    assert first.engine.update_partitions == 4
    assert first.checkpoint_state.progress.samples_seen == 4
    assert first.checkpoint_state.progress.latent_tokens_seen == 8
    assert torch.count_nonzero(first_actor_critic.value_head.weight).item() > 0
    assert set(policy.forward_compute_dtypes) == {torch.bfloat16}
    assert _optimizer_state_dtypes(first.checkpoint_state.optimizer) == {torch.float32}

    resumed_actor_critic = Qwen3ActorCritic(_TinyQwenPolicy(agentic=False))
    resumed = materialize_qwen3_token_ppo_training_run(
        recipe,
        actor_critic=resumed_actor_critic,
        tokenizer=_TinyTokenizer(),
        samples=_ppo_samples(),
        output_dir=output_dir,
        resume_checkpoint="latest",
        fused_adamw=False,
    )
    resumed_summary = resumed.run(max_iterations=1)
    assert resumed_summary.initial_optimizer_step == 4
    assert resumed_summary.final_optimizer_step == 8
    artifact = resumed.export_policy()
    assert isinstance(artifact, FullModelArtifact)
    assert artifact.path.is_dir()

    restored_actor_critic = Qwen3ActorCritic(_TinyQwenPolicy(agentic=False))
    loaded = load_full_model(restored_actor_critic, artifact.path)
    assert loaded == artifact
    assert (
        assert_state_equal(
            resumed_actor_critic.state_dict(),
            restored_actor_critic.state_dict(),
            path="exported_actor_critic",
        )
        > 0
    )

    with pytest.raises(ValueError, match="only supported by Qwen3 grouped Agentic"):
        materialize_qwen3_post_training_run(recipe, reward_url="http://judge")


def test_qwen3_token_ppo_split_resume_matches_uninterrupted_state(tmp_path: Path) -> None:
    torch.manual_seed(19)
    full_actor_critic = Qwen3ActorCritic(_TinyQwenPolicy(agentic=False))
    full = materialize_qwen3_token_ppo_training_run(
        _ppo_recipe(tmp_path / "full"),
        actor_critic=full_actor_critic,
        tokenizer=_TinyTokenizer(),
        samples=_ppo_samples(),
        initialization_seed=17,
        fused_adamw=False,
    )
    assert full.run(max_iterations=2).final_optimizer_step == 8
    expected = {
        "model": snapshot_state(full_actor_critic.state_dict()),
        "optimizer": snapshot_state(full.checkpoint_state.optimizer.state_dict()),
        "engine": snapshot_state(full.engine.state_dict()),
        "dataloader": snapshot_state(full.dataloader.state_dict()),
        "progress": snapshot_state(full.checkpoint_state.progress.state_dict()),
        "objective_generator": full.checkpoint_state.objective_generator.get_state().clone(),
    }

    torch.manual_seed(19)
    first = materialize_qwen3_token_ppo_training_run(
        _ppo_recipe(tmp_path / "split"),
        actor_critic=Qwen3ActorCritic(_TinyQwenPolicy(agentic=False)),
        tokenizer=_TinyTokenizer(),
        samples=_ppo_samples(),
        initialization_seed=17,
        fused_adamw=False,
    )
    assert first.run(max_iterations=1).final_optimizer_step == 4

    resumed_actor_critic = Qwen3ActorCritic(_TinyQwenPolicy(agentic=False))
    resumed = materialize_qwen3_token_ppo_training_run(
        _ppo_recipe(tmp_path / "split"),
        actor_critic=resumed_actor_critic,
        tokenizer=_TinyTokenizer(),
        samples=_ppo_samples(),
        resume_checkpoint="latest",
        initialization_seed=17,
        fused_adamw=False,
    )
    assert resumed.resume_artifact is not None
    assert resumed.run(max_iterations=1).final_optimizer_step == 8

    actual = {
        "model": resumed_actor_critic.state_dict(),
        "optimizer": resumed.checkpoint_state.optimizer.state_dict(),
        "engine": resumed.engine.state_dict(),
        "dataloader": resumed.dataloader.state_dict(),
        "progress": resumed.checkpoint_state.progress.state_dict(),
        "objective_generator": resumed.checkpoint_state.objective_generator.get_state(),
    }
    assert assert_state_equal(expected, actual, path="qwen3_ppo_resume") > 0
    assert any(name.startswith("policy.") for name in actual["model"])
    assert any(name.startswith("value_head.") for name in actual["model"])


def test_token_ppo_pad_crosses_as_many_epochs_as_needed() -> None:
    loader = NativeTokenPPODataLoader(
        _ppo_samples(),
        batch_size=5,
        policy_revision=lambda: "policy",
        sampling_temperature=1.0,
        shuffle=False,
        tail_policy="pad",
    )
    request = next(loader)
    assert len(request.sample_ids) == 5
    assert request.conditioning["base_sample_ids"] == (
        "arithmetic-0",
        "arithmetic-1",
        "arithmetic-2",
        "arithmetic-3",
        "arithmetic-0",
    )


@pytest.mark.parametrize(
    ("config", "artifact_role", "reward_url"),
    (
        (AGENTIC_CONFIG, "policy", "http://judge"),
        (RAY_CONFIG, "policy", "http://judge"),
        (PPO_CONFIG, "actor-critic", None),
    ),
)
def test_qwen3_post_train_cli_dispatches_native_runs(
    monkeypatch,
    tmp_path: Path,
    capsys,
    config: Path,
    artifact_role: str,
    reward_url: str | None,
) -> None:
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    output_dir = tmp_path / config.stem
    argv = [
        "post-train",
        "--recipe",
        str(config),
        "--base-dir",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--steps",
        "2",
        "--resume-checkpoint",
        "checkpoint-x",
    ]
    if reward_url is not None:
        argv.extend(("--reward-url", reward_url))
    args = parser.parse_args(argv)
    called: dict[str, object] = {}

    @dataclass(frozen=True)
    class _Summary:
        def to_dict(self) -> dict[str, object]:
            return {"final_optimizer_step": 2}

    class _Run:
        world_size = 1
        is_coordinator = True

        def __init__(self) -> None:
            self.output_dir = output_dir
            self.artifact_role = artifact_role

        def run(self, *, max_iterations):
            called["iterations"] = max_iterations
            return _Summary()

        def export_policy(self):
            return SimpleNamespace(
                path=tmp_path / "artifact",
                file_size_bytes={"model.safetensors": 1},
            )

        def close(self):
            called["closed"] = True

    def _materialize(recipe, **kwargs):
        called["algorithm"] = recipe.algorithm.type
        called["rollout"] = recipe.rollout
        called.update(kwargs)
        return _Run()

    monkeypatch.setattr(qwen_cli, "materialize_qwen3_cli_run", _materialize)
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert called["iterations"] == 2
    assert called["reward_url"] == reward_url
    assert called["device"] == "cpu"
    assert called["output_dir"] == output_dir
    assert called["initialization_seed"] == 42
    assert called["resume_checkpoint"] == (tmp_path / "checkpoint-x").resolve()
    if config == RAY_CONFIG:
        assert called["rollout"].trainer_binding == "actor"
        assert called["rollout"].placement == "separate"
    assert called["closed"] is True
    assert payload["trained_artifact"]["role"] == artifact_role
    assert payload["summary"]["final_optimizer_step"] == 2


def test_qwen3_ray_default_worker_device_follows_the_recipe_pool() -> None:
    gpu_recipe = PostTrainingRecipe.from_file(RAY_CONFIG)
    cpu_recipe = PostTrainingRecipe.from_file(RAY_CPU_FIXTURE)

    assert qwen3_materializer._ray_worker_device_type(gpu_recipe) == "cuda"
    assert qwen3_materializer._ray_worker_device_type(cpu_recipe) == "cpu"


def test_qwen3_configs_round_trip_and_public_import_is_transformers_lazy() -> None:
    for path in (AGENTIC_CONFIG, PPO_CONFIG):
        recipe = PostTrainingRecipe.from_file(path)
        assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe
    agentic = PostTrainingRecipe.from_file(AGENTIC_CONFIG)
    ppo = PostTrainingRecipe.from_file(PPO_CONFIG)
    assert agentic.algorithm.advantage_normalization == "group-population-std"
    assert ppo.algorithm.update_epochs == 1
    assert ppo.algorithm.update_partitions == 4

    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import worldfoundry.training.post_training.causal_lm.qwen3; "
                "print(json.dumps('transformers' in sys.modules))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(probe.stdout) is False


def test_qwen3_pretrained_loader_requests_fp32_master_weights(monkeypatch) -> None:
    import transformers

    captured: dict[str, object] = {}

    def _from_pretrained(source: str, **kwargs: object) -> nn.Module:
        captured["source"] = source
        captured.update(kwargs)
        return _TinyQwenPolicy(agentic=False)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _from_pretrained)
    loaded = qwen3_materializer._load_qwen3_policy(
        "Qwen/Qwen3-4B-Base",
        attention_implementation="sdpa",
    )
    assert isinstance(loaded, _TinyQwenPolicy)
    assert captured == {
        "source": "Qwen/Qwen3-4B-Base",
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
    }


def test_qwen3_rejects_legacy_data_generation_limit(tmp_path: Path) -> None:
    payload = _agentic_recipe(tmp_path / "unused").to_dict()
    payload["data"]["options"]["max_new_tokens"] = 16
    recipe = PostTrainingRecipe.from_mapping(payload)
    with pytest.raises(ValueError, match="belongs in model.options"):
        materialize_qwen3_agentic_training_run(recipe)


def test_qwen3_rejects_conflicting_generation_horizon(tmp_path: Path) -> None:
    payload = _ppo_recipe(tmp_path / "unused").to_dict()
    payload["algorithm"]["horizon"] = 3
    recipe = PostTrainingRecipe.from_mapping(payload)
    with pytest.raises(ValueError, match="horizon must equal"):
        materialize_qwen3_token_ppo_training_run(recipe)


def test_qwen3_token_ppo_requires_batch_divisible_by_update_partitions(tmp_path: Path) -> None:
    payload = _ppo_recipe(tmp_path / "unused").to_dict()
    payload["data"]["options"]["batch_size"] = 6
    recipe = PostTrainingRecipe.from_mapping(payload)
    with pytest.raises(ValueError, match="batch_size must be divisible"):
        materialize_qwen3_token_ppo_training_run(
            recipe,
            actor_critic=Qwen3ActorCritic(_TinyQwenPolicy(agentic=False)),
            tokenizer=_TinyTokenizer(),
            samples=_ppo_samples(),
            fused_adamw=False,
        )
