from __future__ import annotations

from pathlib import Path

from worldfoundry.studio.inference import catalog as studio_catalog


def test_echo_memory_context_k1_workspace_defaults_use_public_checkpoint(monkeypatch, tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "Echo-Team--Echo-Memory" / "context_k1" / "epoch-0.safetensors"
    wan_root = checkpoint_root / "Wan-AI--Wan2.1-T2V-1.3B"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    wan_root.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(checkpoint_root))

    entry = studio_catalog.find_entry("echo-memory-context-k1")

    assert entry.default_model_ref == str(checkpoint)
    assert entry.default_load_kwargs == {"wan_base_dir": str(wan_root)}
    assert entry.default_input_path.endswith("worldfoundry/data/test_cases/studio_demo/00/image.jpg")
    assert entry.default_call_kwargs == {
        "num_frames": 81,
        "fps": 15,
        "width": 640,
        "height": 352,
        "num_chunks": 2,
        "steps": 50,
        "guidance_scale": 5.0,
        "seed": 42,
        "camera_trajectory": "z*80",
        "return_dict": True,
    }
