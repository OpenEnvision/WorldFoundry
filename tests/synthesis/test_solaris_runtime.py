from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.core.io.paths import package_data_path
from worldfoundry.synthesis.visual_generation.solaris import runtime_env
from worldfoundry.synthesis.visual_generation.solaris import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


def test_solaris_inference_imports_sharding_helpers() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "worldfoundry/synthesis/visual_generation/solaris/solaris_runtime/src/runners/inference.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.utils.sharding"
            and alias.asname == "sharding_utils"
            for alias in node.names
        )
        for node in tree.body
    )


def test_solaris_runtime_root_is_a_data_directory_without_source_sentinels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit"
    preferred = tmp_path / "preferred"
    fallback = tmp_path / "fallback"
    for path in (explicit, preferred, fallback):
        path.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_SOLARIS_RUNTIME_ROOT", str(preferred))
    monkeypatch.setenv("SOLARIS_RUNTIME_ROOT", str(fallback))

    assert runtime_env.resolve_runtime_root(str(explicit)) == str(explicit)
    assert runtime_env.resolve_runtime_root() == str(preferred)
    monkeypatch.delenv("WORLDFOUNDRY_SOLARIS_RUNTIME_ROOT")
    assert runtime_env.resolve_runtime_root() == str(fallback)
    monkeypatch.delenv("SOLARIS_RUNTIME_ROOT")
    assert Path(runtime_env.resolve_runtime_root()) == Path(runtime_env.__file__).parent / "solaris_runtime"


def test_solaris_checkpoint_model_path_keeps_the_installed_default_work_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_SOLARIS_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("SOLARIS_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr(runtime, "_visible_cuda_device_count", lambda: 1)
    checkpoint = tmp_path / "checkpoint"
    weights = checkpoint / "solaris.pt"
    weights.mkdir(parents=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    model = runtime.SolarisRuntime.from_pretrained(
        pretrained_model_path=str(checkpoint),
        pretrained_model_dir=str(checkpoint),
        eval_data_dir=str(dataset),
    )

    expected = Path(runtime_env.__file__).parent / "solaris_runtime"
    assert model.runtime_root == str(expected)
    assert model.output_dir == str(expected / "output")
    assert model.checkpoint_dir == str(expected / "checkpoint")
    assert model.jax_cache_dir == str(expected / "jax_cache")
    assert model.model_weights_path == str(weights)


@pytest.mark.parametrize("pythonpath", [None, "/operator/pythonpath"])
@pytest.mark.parametrize("show_progress", [False, True])
def test_solaris_runs_the_package_module_from_a_data_only_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pythonpath: str | None, show_progress: bool
) -> None:
    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setattr(runtime, "_visible_cuda_device_count", lambda: 1)
    work = tmp_path / "work"
    weights = work / "pretrained" / "solaris.pt"
    weights.mkdir(parents=True)
    (work / "datasets").mkdir()
    model = runtime.SolarisRuntime.from_pretrained(runtime_root=str(work), python_executable="python", device="cuda:2")
    captured: dict[str, Any] = {}

    def execute(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", execute)
    result = model.predict(
        eval_types="translationEval,eval_structure",
        experiment_name="demo",
        show_progress=show_progress,
        return_dict=True,
    )

    assert captured["command"][:3] == [
        "python",
        "-m",
        "worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.inference",
    ]
    assert captured["cwd"] == str(work)
    assert captured["env"].get("PYTHONPATH") == pythonpath
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert captured["stdout"] == (None if show_progress else subprocess.DEVNULL)
    assert captured["stderr"] == (None if show_progress else subprocess.STDOUT)
    assert result["runtime_root"] == str(work)
    assert result["selected_eval_types"] == ["translation", "structure"]
    assert result["model_output_dir"] == str(work / "output" / "demo")
    assert not (work / "src").exists()


def test_solaris_hydra_targets_are_owned_package_definitions() -> None:
    config_root = package_data_path("models", "runtime", "configs", "solaris")
    package_root = Path(runtime_env.__file__).parents[3]
    targets: list[str] = []
    for path in config_root.rglob("*.yaml"):
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().partition(": ")
            if separator and key in {"target", "class"}:
                targets.append(value)
    assert len(targets) == 14
    for target in targets:
        if target == "jax.distributed.initialize":
            continue
        assert target.startswith("worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.")
        module, name = target.rsplit(".", 1)
        source = package_root.joinpath(*module.split(".")[1:]).with_suffix(".py")
        definitions = ast.parse(source.read_text()).body
        assert any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name for node in definitions)
