from __future__ import annotations

import json

import pytest
import torch

from worldfoundry.core.io.integrity import canonical_json
from worldfoundry.training.data import (
    RolloutConditioningDataset,
    RolloutPromptDataset,
    RolloutPromptRecord,
    prepare_rollout_conditioning_cache,
)
from worldfoundry.training.safety import PromptSafetyAudit
from worldfoundry.training.safety.shieldgemma import SHIELDGEMMA_PROMPT_POLICIES


class _Encoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int, int]] = []

    def encode(
        self,
        *,
        sample_id: str,
        prompt: str,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.calls.append((sample_id, prompt, frames, height, width))
        offset = float(len(self.calls))
        return torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset


def _record(prompt_id: str, prompt: str) -> RolloutPromptRecord:
    audit = PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={key: 0.01 for key in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )
    return RolloutPromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        safety_audit=audit,
        generation={"height": 256, "width": 416, "num_frames": 17},
    )


def _manifest(tmp_path) -> RolloutPromptDataset:
    path = tmp_path / "prompts.jsonl"
    records = (_record("first", "first prompt"), _record("second", "second prompt"))
    path.write_text(
        "".join(canonical_json(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )
    return RolloutPromptDataset.from_file(path)


def test_rollout_conditioning_cache_round_trips_and_binds_every_identity(tmp_path) -> None:
    prompts = _manifest(tmp_path)
    encoder = _Encoder()
    cache = tmp_path / "cache"
    prepared = prepare_rollout_conditioning_cache(
        prompts,
        cache_root=cache,
        encoder=encoder,
        model_recipe="wan2.1-t2v-1.3b",
        conditioner={"repo_id": "encoder", "revision": "main"},
        tokenizer={"repo_id": "tokenizer", "revision": "main"},
    )
    dataset = RolloutConditioningDataset(prompts, cache)

    assert len(prepared.entries) == len(dataset) == 2
    assert encoder.calls[0] == ("first", "first prompt", 17, 256, 416)
    assert dataset.index == prepared.index
    assert dataset[0].record.prompt_id == "first"
    torch.testing.assert_close(
        dataset[0].conditioning["context"],
        torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1,
    )


def test_rollout_conditioning_cache_rejects_index_tampering(tmp_path) -> None:
    prompts = _manifest(tmp_path)
    cache = tmp_path / "cache"
    prepare_rollout_conditioning_cache(
        prompts,
        cache_root=cache,
        encoder=_Encoder(),
        model_recipe="wan2.1-t2v-1.3b",
        conditioner={"repo_id": "encoder", "revision": "main"},
        tokenizer={"repo_id": "tokenizer", "revision": "main"},
    )
    path = cache / "rollout-conditioning-index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["index"]["entries"][0]["record"]["prompt_id"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="another prompt dataset"):
        RolloutConditioningDataset(prompts, cache)
