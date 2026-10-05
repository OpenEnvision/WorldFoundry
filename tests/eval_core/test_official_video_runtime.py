from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from worldfoundry.pipelines.video_official.pipeline_official_video import Mochi1PreviewT2VPipeline
from worldfoundry.synthesis.visual_generation.official_video_runtime import (
    OfficialVideoRuntime,
    _normalize_cli_path_variables,
)


def test_official_video_variant_prefers_its_runtime_model_id(monkeypatch) -> None:
    captured = {}

    class FakeRuntime:
        def prepare(self) -> None:
            return None

    def from_model_id(model_id, **kwargs):
        captured.update(model_id=model_id, kwargs=kwargs)
        return FakeRuntime()

    monkeypatch.setattr(OfficialVideoRuntime, "from_model_id", from_model_id)

    pipeline = Mochi1PreviewT2VPipeline.from_pretrained(
        model_path={"checkpoint_path": "/tmp/mochi"},
        model_id="mochi-1",
        device="cpu",
    )

    assert captured["model_id"] == "mochi-1-preview-t2v"
    assert pipeline.model_id == "mochi-1-preview-t2v"


def test_official_video_runtime_checks_nested_required_paths(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "runtime"
    checkpoint_root = tmp_path / "ckpt"
    repo_root.mkdir()
    checkpoint_root.mkdir()

    config = {
        "runtime": {
            "kind": "official_cli",
            "repo_root_candidates": [str(repo_root)],
            "checkpoint_candidates": [str(checkpoint_root)],
            "required_paths": [
                {"id": "text_encoder", "path": "text_encoder/llm/config.json"},
                {"id": "entrypoint", "base": "repo", "path": "generate.py"},
            ],
            "command": ["{python}", "{repo_root}/generate.py", "--model_path", "{checkpoint_path}"],
        }
    }
    monkeypatch.setattr(OfficialVideoRuntime, "_load_config", staticmethod(lambda _: config))

    runtime = OfficialVideoRuntime(model_id="fixture", runtime_config_path="unused.yaml")
    missing_plan = runtime.runtime_plan(output_path=tmp_path / "out.mp4", prompt="demo")

    assert missing_plan["ready"] is False
    assert any("text_encoder" in item for item in missing_plan["missing"])
    assert any("entrypoint" in item for item in missing_plan["missing"])

    (checkpoint_root / "text_encoder" / "llm").mkdir(parents=True)
    (checkpoint_root / "text_encoder" / "llm" / "config.json").write_text("{}", encoding="utf-8")
    (repo_root / "generate.py").write_text("print('ok')\n", encoding="utf-8")

    ready_plan = runtime.runtime_plan(output_path=tmp_path / "out.mp4", prompt="demo")

    assert ready_plan["ready"] is True
    assert ready_plan["missing"] == []


def test_open_sora_runtime_requires_local_vae_and_text_encoder(tmp_path: Path, monkeypatch) -> None:
    hfd_root = tmp_path / "hfd"
    stdit = hfd_root / "hpcai-tech--OpenSora-STDiT-v3"
    vae = hfd_root / "hpcai-tech--OpenSora-VAE-v1.2"
    text_encoder = hfd_root / "DeepFloyd--t5-v1_1-xxl"
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(hfd_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))

    stdit.mkdir(parents=True)
    for filename in ("config.json", "model.safetensors"):
        (stdit / filename).touch()

    runtime = OfficialVideoRuntime.from_model_id("open-sora", device="cpu")
    missing_plan = runtime.runtime_plan(output_path=tmp_path / "missing.mp4", prompt="demo")

    assert missing_plan["ready"] is False
    assert any("video_vae" in item for item in missing_plan["missing"])
    assert any("text_encoder" in item for item in missing_plan["missing"])

    vae.mkdir()
    for filename in ("config.json", "model.safetensors"):
        (vae / filename).touch()
    text_encoder.mkdir()
    for filename in (
        "config.json",
        "pytorch_model.bin.index.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "spiece.model",
    ):
        (text_encoder / filename).touch()

    ready_runtime = OfficialVideoRuntime.from_model_id("open-sora", device="cpu")
    ready_plan = ready_runtime.runtime_plan(output_path=tmp_path / "ready.mp4", prompt="demo")

    assert ready_plan["ready"] is True
    assert ready_plan["missing"] == []


def test_materialize_cli_artifacts_preserves_media_suffixes(tmp_path: Path) -> None:
    produced_audio = tmp_path / "official.flac"
    produced_video = tmp_path / "official.mp4"
    produced_audio.write_bytes(b"fLaC-audio")
    produced_video.write_bytes(b"\x00\x00\x00\x18ftyp-video")

    output_path = tmp_path / "workspace.mp4"
    primary, artifacts = OfficialVideoRuntime._materialize_cli_artifacts(
        produced_audio,
        output_path,
        since=0.0,
    )

    assert primary == tmp_path / "workspace.flac"
    assert primary.read_bytes() == produced_audio.read_bytes()
    assert output_path.read_bytes() == produced_video.read_bytes()
    assert artifacts == (tmp_path / "workspace.flac", output_path)


def test_diffusers_pipeline_is_prepared_and_reused(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    calls: list[tuple[str, object]] = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_path: str, *, torch_dtype):
            calls.append((model_path, torch_dtype))
            return cls()

        def to(self, device: str):
            self.device = device
            return self

    config = {
        "runtime": {
            "kind": "diffusers_pipeline",
            "checkpoint_candidates": [str(checkpoint_root)],
            "pipeline_target": "fake_diffusers:FakePipeline",
            "torch_dtype": "float32",
        }
    }
    monkeypatch.setattr(OfficialVideoRuntime, "_load_config", staticmethod(lambda _: config))
    monkeypatch.setattr(
        "importlib.import_module",
        lambda module_name: SimpleNamespace(FakePipeline=FakePipeline)
        if module_name == "fake_diffusers"
        else __import__(module_name),
    )

    runtime = OfficialVideoRuntime(model_id="fixture", runtime_config_path="unused.yaml", device="cpu")
    runtime.prepare()
    first, _ = runtime._get_diffusers_pipeline(checkpoint_root)
    runtime.prepare()
    second, _ = runtime._get_diffusers_pipeline(checkpoint_root)

    assert first is second
    assert len(calls) == 1
    assert calls[0][0] == str(checkpoint_root.resolve())


def test_official_cli_resolves_prefixed_wan_flat_mirror(tmp_path: Path) -> None:
    expected = tmp_path / "Wan-AI--Wan2.1-T2V-14B"
    expected.mkdir()

    normalized = _normalize_cli_path_variables(
        {"wan_model_root": str(tmp_path / "Wan2.1-T2V-14B")}
    )

    assert normalized["wan_model_root"] == expected.resolve()


def test_official_cli_plan_exposes_selected_checkpoint_parent(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "runtime"
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "krea--krea-realtime-video"
    repo_root.mkdir()
    checkpoint.mkdir(parents=True)
    config = {
        "runtime": {
            "kind": "official_cli",
            "repo_root_candidates": [str(repo_root)],
            "checkpoint_candidates": [str(checkpoint)],
            "command": ["runner", "--model-folder", "{checkpoint_parent}"],
        }
    }
    monkeypatch.setattr(OfficialVideoRuntime, "_load_config", staticmethod(lambda _: config))

    runtime = OfficialVideoRuntime(model_id="fixture", runtime_config_path="unused.yaml")
    plan = runtime.runtime_plan(output_path=tmp_path / "out.mp4", prompt="demo")

    assert plan["command"][-1] == str(checkpoint_root.resolve())
