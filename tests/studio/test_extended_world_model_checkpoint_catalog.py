from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.studio.inference.catalog import find_entry


def test_moverse_catalog_declares_every_runtime_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "ckpts"
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    for directory in (
        "Orange-3DV-Team--MoVerse",
        "black-forest-labs--FLUX.1-Fill-dev",
        "depth-anything--DA3NESTED-GIANT-LARGE-1.1",
        "Wan-AI--Wan2.1-T2V-1.3B",
    ):
        (checkpoint_root / directory).mkdir(parents=True)

    entry = find_entry("moverse")

    assert entry.default_model_ref == str(checkpoint_root / "Orange-3DV-Team--MoVerse")
    assert set(entry.default_load_kwargs) == {"flux_path", "da3_path", "wan_path"}
    assert all(Path(value).is_dir() for value in entry.default_load_kwargs.values())
    assert Path(entry.default_input_path).is_file()


def test_dreamdojo_catalog_pins_gr1_dcp_shards_and_dataset(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "ckpts"
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    root = checkpoint_root / "nvidia--DreamDojo" / "2B_GR1_post-train"
    model_dir = root / "iter_000050000" / "model"
    model_dir.mkdir(parents=True)

    entry = find_entry("dreamdojo")

    assert entry.default_model_ref == str(root)
    assert entry.default_load_kwargs["dataset_path"] == str(
        checkpoint_root / "nvidia--PhysicalAI-Robotics-GR00T-Teleop-GR1" / "GR1_robot"
    )
    assert entry.default_load_kwargs["checkpoint_shards"] == [
        str(model_dir / f"__{index}_0.distcp") for index in range(8)
    ]


def test_sana_wm_streaming_catalog_binds_all_native_roles(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "ckpts"
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    streaming = checkpoint_root / "Efficient-Large-Model--SANA-WM_streaming"
    gemma = checkpoint_root / "Efficient-Large-Model--gemma-2-2b-it"
    streaming.mkdir(parents=True)
    gemma.mkdir(parents=True)

    entry = find_entry("sana-wm-streaming")
    overrides = entry.default_load_kwargs["checkpoint_overrides"]

    assert entry.default_model_ref == str(streaming)
    assert overrides["text-encoder"] == str(gemma)
    assert overrides["tokenizer"] == str(gemma)
    assert all(
        overrides[role] == str(streaming)
        for role in ("dit", "codec", "refiner", "refiner-connectors", "refiner-text-encoder")
    )
    assert Path(entry.default_input_path).is_file()


@pytest.mark.parametrize(
    ("model_id", "checkpoint_file"),
    (
        ("echo-memory-context-k20", "context_k20/epoch-0.safetensors"),
        ("echo-memory-spatial", "spatial_mem/epoch-0.safetensors"),
        ("echo-memory-block-ssm", "block_wise_ssm_two_chunk/epoch-0.safetensors"),
        ("echo-memory-videossm-hybrid", "videossm_hybrid/epoch-0.safetensors"),
        ("echo-memory-spatial-concat-text", "spatial_concat_text_two_chunk/epoch-0.safetensors"),
        ("echo-memory-spatial-no-injection", "spatial_inject_none_two_chunk/epoch-0.safetensors"),
        (
            "echo-memory-spatial-cross-attn-t32",
            "spatial_cross_attn_readout_t32_g4_two_chunk/epoch-0.safetensors",
        ),
        (
            "echo-memory-ssm-ctx1-every4-hint21",
            "ssm_ablation_ctx1_every4_hint21/epoch-0.safetensors",
        ),
        (
            "echo-memory-ssm-ctx5-every1-hint21",
            "ssm_ablation_ctx5_every1_hint21/epoch-0.safetensors",
        ),
        (
            "echo-memory-ssm-ctx5-every4-hint81",
            "ssm_ablation_ctx5_every4_hint81/epoch-0.safetensors",
        ),
    ),
)
def test_echo_memory_catalog_maps_each_model_to_its_immutable_checkpoint(
    model_id: str,
    checkpoint_file: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_root = tmp_path / "ckpts"
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    checkpoint = checkpoint_root / "Echo-Team--Echo-Memory" / checkpoint_file
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint_root / "Wan-AI--Wan2.1-T2V-1.3B").mkdir(parents=True)

    entry = find_entry(model_id)

    assert entry.default_model_ref == str(checkpoint)
    assert entry.default_load_kwargs["wan_base_dir"] == str(
        checkpoint_root / "Wan-AI--Wan2.1-T2V-1.3B"
    )
    assert entry.default_call_kwargs["num_frames"] == 81
    assert Path(entry.default_input_path).is_file()
