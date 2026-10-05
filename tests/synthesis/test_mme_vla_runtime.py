from __future__ import annotations

import numpy as np

from worldfoundry.synthesis.action_generation.mme_vla.runtime import (
    _buffer_observation,
    _policy_observation,
    _stage_paligemma_tokenizer,
)


def test_stage_paligemma_tokenizer_in_openpi_cache(tmp_path, monkeypatch):
    ckpt_root = tmp_path / "checkpoints"
    source = ckpt_root / "hfd_models" / "google--paligemma-3b-pt-224" / "tokenizer.model"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sentencepiece")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(ckpt_root))
    monkeypatch.delenv("OPENPI_DATA_HOME", raising=False)

    target = _stage_paligemma_tokenizer()

    assert not target.is_symlink()
    assert target.read_bytes() == b"sentencepiece"
    assert target.parent == ckpt_root / "runtime_assets" / "openpi" / "big_vision"


def test_buffer_observation_matches_official_single_view_contract():
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    payload = _buffer_observation(
        {"observation/image": image, "observation/state": [0.0] * 8},
        "move the object",
    )

    assert payload["images"].shape == (1, 1, 12, 18, 3)
    assert payload["images"].dtype == np.uint8
    assert payload["state"].shape == (1, 8)
    assert payload["exec_start_idx"] == 0


def test_policy_observation_normalizes_json_state_to_float32():
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    payload = _policy_observation(
        {
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/state": [0.0] * 8,
        },
        image,
        "move the object",
    )

    assert payload["observation/state"].shape == (8,)
    assert payload["observation/state"].dtype == np.float32
