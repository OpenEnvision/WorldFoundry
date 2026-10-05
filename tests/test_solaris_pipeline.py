from __future__ import annotations

import subprocess
from pathlib import Path

import worldfoundry.evaluation.models.pipelines as model_pipelines
from worldfoundry.evaluation.models import discover_model_registry
from worldfoundry.pipelines.solaris.pipeline_solaris import SolarisPipeline
from worldfoundry.synthesis.visual_generation.solaris import runtime_env as runtime_env_module
from worldfoundry.synthesis.visual_generation.solaris import (
    worldfoundry_runtime as solaris_runtime_module,
)
from worldfoundry.synthesis.visual_generation.solaris import (
    solaris_synthesis as synthesis_module,
)


class _DummySynthesis:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "model_output_dir": "/tmp/solaris_worldfoundry",
            "generated_videos": {"translation": ["/tmp/solaris_worldfoundry/eval_translation/video_0_side_by_side.mp4"]},
        }


def test_solaris_pipeline_process_normalizes_eval_types() -> None:
    pipe = SolarisPipeline(synthesis_model=_DummySynthesis())

    processed = pipe.process(eval_types=["translationEval", "eval_rotation"])

    assert processed["eval_types"] == ["translation", "rotation"]


def test_solaris_pipeline_call_forwards_runtime_options() -> None:
    synthesis = _DummySynthesis()
    pipe = SolarisPipeline(synthesis_model=synthesis)

    result = pipe(
        eval_types=["translation"],
        experiment_name="demo",
        eval_num_samples=2,
        return_dict=True,
    )

    assert result["model_output_dir"] == "/tmp/solaris_worldfoundry"
    assert synthesis.calls
    assert synthesis.calls[0]["eval_types"] == ["translation"]
    assert synthesis.calls[0]["experiment_name"] == "demo"
    assert synthesis.calls[0]["eval_num_samples"] == 2


def test_load_solaris_loader_forwards_runtime_components(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(cls, model_path=None, required_components=None, device="cuda", **kwargs):
        captured["model_path"] = model_path
        captured["required_components"] = required_components
        captured["device"] = device
        captured["kwargs"] = kwargs
        return "pipeline"

    monkeypatch.setattr(
        SolarisPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    pipeline = SolarisPipeline.from_pretrained(
        model_path="/tmp/solaris",
        required_components={
            "pretrained_model_dir": "/tmp/solaris/pretrained",
            "eval_data_dir": "/tmp/solaris/datasets",
            "output_dir": "/tmp/solaris/output",
            "model_weights_path": "/tmp/solaris/pretrained/solaris.pt",
            "python_executable": "/tmp/solaris/bin/python",
            "enable_jax_cache": True,
        },
        device="cuda:1",
        eval_num_samples=3,
        num_workers=2,
        num_frames_eval=65,
    )

    assert pipeline == "pipeline"
    assert captured["model_path"] == "/tmp/solaris"
    assert captured["required_components"] == {
        "pretrained_model_dir": "/tmp/solaris/pretrained",
        "eval_data_dir": "/tmp/solaris/datasets",
        "output_dir": "/tmp/solaris/output",
        "model_weights_path": "/tmp/solaris/pretrained/solaris.pt",
        "python_executable": "/tmp/solaris/bin/python",
        "enable_jax_cache": True,
    }
    assert captured["device"] == "cuda:1"
    assert captured["kwargs"] == {
        "eval_num_samples": 3,
        "num_workers": 2,
        "num_frames_eval": 65,
    }


def test_solaris_pipeline_has_no_generic_infer_facade() -> None:
    assert not hasattr(model_pipelines, "infer_solaris_pipeline")


def test_model_registry_contains_solaris() -> None:
    discover_model_registry.cache_clear()
    registry = discover_model_registry()
    model = registry.get("solaris")

    assert model.family == "video_generation"
    assert model.has_loader is True
    assert model.has_infer is False


def _create_fake_runtime_root(tmp_path: Path) -> Path:
    runtime_root = tmp_path / "solaris"
    (runtime_root / "src").mkdir(parents=True)
    (runtime_root / "config").mkdir(parents=True)
    (runtime_root / "src" / "inference.py").write_text("# stub\n", encoding="utf-8")
    (runtime_root / "config" / "inference.yaml").write_text("defaults: []\n", encoding="utf-8")
    return runtime_root


def _create_fake_python_venv(tmp_path: Path) -> tuple[Path, list[str]]:
    venv_root = tmp_path / "fake_venv"
    venv_root.mkdir(parents=True, exist_ok=True)
    (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    python_executable = venv_root / "bin" / "python"
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.symlink_to("/usr/bin/python3.10")

    cuda_lib_dirs: list[str] = []
    nvidia_root = venv_root / "lib" / "python3.10" / "site-packages" / "nvidia"
    for package_name in ("cudnn", "cusparse"):
        lib_dir = nvidia_root / package_name / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        cuda_lib_dirs.append(str(lib_dir))
    return python_executable, cuda_lib_dirs


def test_resolve_model_weights_path_accepts_directory_checkpoint(tmp_path: Path) -> None:
    runtime_root = _create_fake_runtime_root(tmp_path)
    pretrained_dir = runtime_root / "pretrained"
    checkpoint_dir = pretrained_dir / "solaris.pt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "metadata").write_text("ok\n", encoding="utf-8")

    resolved_default = runtime_env_module.resolve_model_weights_path(
        None,
        str(runtime_root),
        str(pretrained_dir),
    )
    resolved_explicit = runtime_env_module.resolve_model_weights_path(
        str(checkpoint_dir),
        str(runtime_root),
        str(pretrained_dir),
    )

    assert resolved_default == str(checkpoint_dir.resolve())
    assert resolved_explicit == str(checkpoint_dir.resolve())


def test_build_inference_env_uses_venv_root_for_symlinked_python(
    monkeypatch,
    tmp_path: Path,
) -> None:
    python_executable, expected_cuda_lib_dirs = _create_fake_python_venv(tmp_path)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.setenv("WORLDFOUNDRY_SOLARIS_MANAGE_LD_LIBRARY_PATH", "1")

    env = runtime_env_module.build_inference_env(
        "cuda:2",
        python_executable=str(python_executable),
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    ld_library_path = env["LD_LIBRARY_PATH"].split(":")
    assert ld_library_path[: len(expected_cuda_lib_dirs)] == expected_cuda_lib_dirs
    assert ld_library_path[-1] == "/existing/lib"


def test_solaris_synthesis_builds_inference_command(monkeypatch, tmp_path: Path) -> None:
    runtime_root = _create_fake_runtime_root(tmp_path)
    python_executable, expected_cuda_lib_dirs = _create_fake_python_venv(tmp_path)
    pretrained_dir = runtime_root / "pretrained"
    data_dir = runtime_root / "datasets"
    output_dir = runtime_root / "output"
    checkpoint_dir = runtime_root / "checkpoint"
    jax_cache_dir = runtime_root / "jax_cache"

    pretrained_dir.mkdir()
    (data_dir / "eval" / "translationEval").mkdir(parents=True)
    (data_dir / "eval" / "structureEval").mkdir(parents=True)
    (pretrained_dir / "clip.pt").write_bytes(b"clip")
    (pretrained_dir / "vae.pt").write_bytes(b"vae")
    (pretrained_dir / "solaris.pt").mkdir()
    (pretrained_dir / "solaris.pt" / "metadata").write_text("ok\n", encoding="utf-8")

    calls = []

    def fake_run(command, check, cwd, env, stdout, stderr):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        if command[1] == "src/inference.py":
            experiment_name = next(
                arg.split("=", 1)[1]
                for arg in command
                if arg.startswith("experiment_name=")
            )
            model_output_dir = Path(
                next(
                    arg.split("=", 1)[1]
                    for arg in command
                    if arg.startswith("device.output_dir=")
                )
            ) / experiment_name
            for hydra_key in ["eval_translation", "eval_structure"]:
                eval_dir = model_output_dir / hydra_key
                eval_dir.mkdir(parents=True, exist_ok=True)
                (eval_dir / "video_0_side_by_side.mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(solaris_runtime_module.subprocess, "run", fake_run)
    monkeypatch.setenv("WORLDFOUNDRY_SOLARIS_MANAGE_LD_LIBRARY_PATH", "1")

    synthesis = synthesis_module.SolarisSynthesis(
        runtime_root=str(runtime_root),
        pretrained_model_dir=str(pretrained_dir),
        eval_data_dir=str(data_dir),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        jax_cache_dir=str(jax_cache_dir),
        model_weights_path=str(pretrained_dir / "solaris.pt"),
        python_executable=str(python_executable),
        device="cuda:2",
        defaults={
            "enable_jax_cache": True,
            "eval_num_samples": 1,
            "num_workers": 4,
            "num_frames_eval": 257,
        },
    )

    result = synthesis.predict(
        eval_types=["translation", "structure"],
        experiment_name="demo",
        return_dict=True,
        show_progress=False,
    )

    assert len(calls) == 1

    inference_call = calls[0]
    inference_command = inference_call["command"]
    assert inference_command[:2] == [str(python_executable), "src/inference.py"]
    assert "experiment_name=demo" in inference_command
    assert "device.eval_num_samples=1" in inference_command
    assert "device.num_workers=4" in inference_command
    assert "runner.num_frames_eval=257" in inference_command
    assert "enable_jax_cache=true" in inference_command
    assert "~dataset@eval_datasets.eval_rotation" in inference_command
    assert "~dataset@eval_datasets.eval_turn_to_look" in inference_command
    assert inference_call["cwd"] == str(runtime_root)
    assert inference_call["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert inference_call["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert (
        inference_call["env"]["LD_LIBRARY_PATH"].split(":")[: len(expected_cuda_lib_dirs)]
        == expected_cuda_lib_dirs
    )

    assert result["model_output_dir"] == str(output_dir / "demo")
    assert result["generated_videos"]["translation"] == [
        str(output_dir / "demo" / "eval_translation" / "video_0_side_by_side.mp4")
    ]
    assert result["generated_videos"]["structure"] == [
        str(output_dir / "demo" / "eval_structure" / "video_0_side_by_side.mp4")
    ]
