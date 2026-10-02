from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.core.io.serialization import write_jsonl
from worldfoundry.evaluation.api import ArtifactRef, GenerationResult
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import (
    get_benchmark_generation_adapter,
)
from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import (
    _materialize_generated_artifacts,
)
from worldfoundry.evaluation.tasks.execution.runners.t2v_safety_bench import (
    run_t2v_safety_bench_official_runner as runner,
)
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import (
    build_workspace_benchmark_command,
)


def _write_inputs(tmp_path: Path, count: int = 2) -> tuple[Path, Path]:
    generated = tmp_path / "generated"
    generated.mkdir()
    for index in range(1, count + 1):
        (generated / f"1-{index}.mp4").write_bytes(f"video-{index}".encode())
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(1, count + 1)) + "\n")
    return generated, prompts


def test_generation_adapter_uses_official_prompt_to_filename_mapping(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_T2V_SAFETY_BENCH_CLASS", "2")
    adapter = get_benchmark_generation_adapter("t2v-safety-bench")
    assert adapter is not None

    requests = adapter.materialize_requests(limit=1)

    assert len(requests) == 1
    assert requests[0].inputs["safety_class"] == 2
    assert requests[0].inputs["official_video_name"] == "2-1.mp4"
    assert requests[0].inputs["prompt"]


def test_generated_model_artifact_reaches_official_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_T2V_SAFETY_BENCH_CLASS", "1")
    adapter = get_benchmark_generation_adapter("t2v-safety-bench")
    assert adapter is not None
    request = adapter.materialize_requests(limit=1)[0]
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    source = generation_dir / "model-output.mp4"
    source.write_bytes(b"real-file-for-layout-test")
    result = GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id="model",
        artifacts={"generated_video": ArtifactRef(uri=str(source), kind="video")},
    )
    write_jsonl(generation_dir / "requests.jsonl", [request.to_dict()])
    write_jsonl(generation_dir / "results.jsonl", [result.to_dict()])
    generated = tmp_path / "generated"

    counts = _materialize_generated_artifacts(
        generation_output_dir=generation_dir,
        generated_artifact_dir=generated,
        artifact_manifest_path=tmp_path / "artifacts.jsonl",
        output_artifact="generated_video",
        allow_placeholders=False,
    )

    assert counts == (1, 0)
    assert (generated / "1-1.mp4").read_bytes() == source.read_bytes()
    staged, _, count = runner.stage_official_inputs(
        generated_video_dir=generated,
        prompt_path=Path("worldfoundry/data/benchmarks/assets/t2v-safety-bench/T2VSafetyBench/1.txt"),
        output_dir=tmp_path / "judge",
        class_id=1,
        limit=1,
    )
    assert count == 1
    assert (staged / "1-1.mp4").resolve() == (generated / "1-1.mp4").resolve()


def test_stage_official_inputs_is_bounded_and_exact(tmp_path):
    generated, prompts = _write_inputs(tmp_path)

    video_dir, staged_prompts, count = runner.stage_official_inputs(
        generated_video_dir=generated,
        prompt_path=prompts,
        output_dir=tmp_path / "out",
        class_id=1,
        limit=1,
    )

    assert count == 1
    assert staged_prompts.read_text() == "prompt 1\n"
    assert (video_dir / "1-1.mp4").resolve() == (generated / "1-1.mp4").resolve()
    assert not (video_dir / "1-2.mp4").exists()


def test_stage_official_inputs_rejects_missing_or_ambiguous_video(tmp_path):
    generated, prompts = _write_inputs(tmp_path, count=1)
    (generated / "nested").mkdir()
    (generated / "nested" / "1-1.mov").write_bytes(b"duplicate")

    with pytest.raises(ValueError, match="ambiguous generated videos"):
        runner.stage_official_inputs(
            generated_video_dir=generated,
            prompt_path=prompts,
            output_dir=tmp_path / "out",
            class_id=1,
            limit=None,
        )

    (generated / "nested" / "1-1.mov").unlink()
    prompts.write_text("prompt 1\nprompt 2\n")
    with pytest.raises(ValueError, match="missing generated video for prompt 2"):
        runner.stage_official_inputs(
            generated_video_dir=generated,
            prompt_path=prompts,
            output_dir=tmp_path / "out",
            class_id=1,
            limit=None,
        )


def test_official_command_passes_staged_video_dir_without_leaking_api_key(tmp_path, monkeypatch):
    generated, prompts = _write_inputs(tmp_path)
    secret = "not-a-real-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    runtime_root = Path(runner.CONFIG.default_repo_subdir).resolve()
    args = SimpleNamespace(
        prompt_path=prompts,
        classes=1,
        limit=1,
        model_name="custom-model",
        python=sys.executable,
        api_base=None,
        gpt_model="gpt-4o-2024-05-13",
        gpt_eval_prompts=None,
    )

    command = runner.build_official_command(
        config=runner.CONFIG,
        repo_root=runtime_root,
        generated_video_dir=generated,
        output_dir=tmp_path / "out",
        args=args,
    )

    assert command is not None
    assert command[command.index("--video-dir") + 1].endswith("/upstream_input/video")
    assert command[command.index("--prompt-path") + 1].endswith("/upstream_input/class1_prompts.txt")
    assert "--gpt-api" not in command
    assert secret not in command


def test_workspace_command_forwards_generated_dir_and_bound():
    command = build_workspace_benchmark_command(
        {
            "benchmark_id": "t2v-safety-bench",
            "params": {"generated_video_dir": "/generated", "limit": 3},
        },
        "/output",
    )

    assert command[command.index("--generated-video-dir") + 1] == "/generated"
    assert command[command.index("--limit") + 1] == "3"
    assert "--run-official" in command


def test_missing_api_key_blocks_before_upstream_execution(tmp_path, monkeypatch):
    generated, prompts = _write_inputs(tmp_path, count=1)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_T2V_SAFETY_BENCH_GPT_API_KEY", raising=False)
    output_dir = tmp_path / "out"

    exit_code = runner.main(
        [
            "--run-official",
            "--generated-video-dir",
            str(generated),
            "--prompt-path",
            str(prompts),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    scorecard = json.loads((output_dir / "scorecard.json").read_text())
    assert scorecard["validation"]["official_runtime_executed"] is False
    assert any("OPENAI_API_KEY" in reason for reason in scorecard["validation"]["blocked_reasons"])


def test_upstream_xlsx_rows_map_to_selected_class_metric(tmp_path):
    path = tmp_path / "nsfw_results_model_class5.xlsx"
    extracted = runner.extract_metrics([{"Prompt": "a", "Result": 1}, {"Prompt": "b", "Result": 0}], path)

    assert set(extracted) == {"disturbing_content_nsfw_rate"}
    assert extracted["disturbing_content_nsfw_rate"]["raw_score"] == 0.5
    assert extracted["disturbing_content_nsfw_rate"]["sample_count"] == 2


def test_gpt_client_uses_runtime_key_and_propagates_api_failure(tmp_path, monkeypatch):
    runtime_root = Path(runner.CONFIG.default_repo_subdir).resolve()
    monkeypatch.syspath_prepend(str(runtime_root))
    spec = importlib.util.spec_from_file_location("worldfoundry_t2v_safety_gpt4_test", runtime_root / "gpt4.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ext_frame", lambda *args, **kwargs: ["frame"])
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ANS: Yes, Yes: 80%, No: 20%"))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(module, "OpenAI", FakeClient)
    args = SimpleNamespace(
        gpt_api="runtime-key",
        api_base="https://example.invalid/v1",
        gpt_model="judge",
        n_frames=1,
        scale_percent=20,
        max_tokens=10,
        num_text=1,
        temperature=0.0,
    )
    assert module.gpt4_api(args, "evaluate", "prompt", "definition", video_path="video.mp4")
    assert captured["client"] == {
        "api_key": "runtime-key",
        "base_url": "https://example.invalid/v1",
        "max_retries": 0,
    }

    class BrokenCompletions:
        def create(self, **kwargs):
            raise PermissionError("invalid key")

    class BrokenClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=BrokenCompletions())

    monkeypatch.setattr(module, "OpenAI", BrokenClient)
    with pytest.raises(RuntimeError, match="GPT judge request failed"):
        module.gpt4_api(args, "evaluate", "prompt", "definition", video_path="video.mp4")
