from __future__ import annotations

from pathlib import Path

from worldfoundry.synthesis.action_generation.dreamdojo.worldfoundry_runtime import DreamDojoRuntime


def test_dreamdojo_subprocess_imports_runtime_from_package_parent(tmp_path, monkeypatch):
    runtime = DreamDojoRuntime(checkpoints_dir=tmp_path)
    checkpoint = tmp_path / "model_ema_bf16.pt"
    checkpoint.touch()
    monkeypatch.setattr(runtime, "_checkpoint_path", lambda: checkpoint)

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.run", fake_run)

    runtime.predict(output_path=tmp_path / "run.json", output_dir=tmp_path / "output")

    code = captured["command"][2]
    assert "sys.path.insert(0, str(runtime_root.parent))" in code
    assert "sys.path.insert(0, str(runtime_root))" not in code
