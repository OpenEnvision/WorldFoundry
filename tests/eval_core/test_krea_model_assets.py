from __future__ import annotations

import importlib.util
import types
from pathlib import Path


SETTINGS_PATH = (
    Path(__file__).resolve().parents[2]
    / "worldfoundry/synthesis/visual_generation/krea_realtime/krea_runtime/settings.py"
)


def _load_settings():
    spec = importlib.util.spec_from_file_location("worldfoundry_test_krea_settings", SETTINGS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_krea_resolves_hf_downloader_wan_directory(monkeypatch, tmp_path: Path) -> None:
    settings = _load_settings()
    model_root = tmp_path / "Wan-AI--Wan2.1-T2V-1.3B"
    model_root.mkdir()
    weight = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
    weight.touch()
    monkeypatch.setattr(settings, "MODEL_FOLDER", str(tmp_path))

    assert settings.resolve_model_path("Wan2.1-T2V-1.3B") == model_root.resolve()
    assert settings.resolve_model_asset(
        "Wan2.1-T2V-1.3B",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
    ) == weight


def test_krea_explicit_model_root_has_priority(monkeypatch, tmp_path: Path) -> None:
    settings = _load_settings()
    default_root = tmp_path / "Wan2.1-T2V-14B"
    explicit_root = tmp_path / "explicit-14b"
    default_root.mkdir()
    explicit_root.mkdir()
    monkeypatch.setattr(settings, "MODEL_FOLDER", str(tmp_path))
    monkeypatch.setenv("WORLDFOUNDRY_KREA_WAN_14B_ROOT", str(explicit_root))

    assert settings.resolve_model_path("Wan2.1-T2V-14B") == explicit_root.resolve()


def test_krea_compiler_stance_is_optional() -> None:
    settings = _load_settings()
    torch_stub = types.SimpleNamespace(compiler=types.SimpleNamespace())

    with settings.compiler_stance(torch_stub, "default"):
        pass


def test_krea_compiler_stance_uses_available_api() -> None:
    settings = _load_settings()
    seen: list[str] = []
    torch_stub = types.SimpleNamespace(
        compiler=types.SimpleNamespace(set_stance=lambda value: seen.append(value) or settings.nullcontext())
    )

    with settings.compiler_stance(torch_stub, "force_eager"):
        pass
    assert seen == ["force_eager"]
