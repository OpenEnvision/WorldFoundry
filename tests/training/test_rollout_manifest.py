from __future__ import annotations

import json

import pytest

from worldfoundry.core.io.integrity import canonical_json
from worldfoundry.training.data import RolloutPromptDataset, RolloutPromptRecord
from worldfoundry.training.safety import PromptSafetyAudit
from worldfoundry.training.safety.shieldgemma import SHIELDGEMMA_PROMPT_POLICIES


def _audit(prompt: str) -> PromptSafetyAudit:
    return PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={key: 0.01 for key in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )


def _record(prompt_id: str, prompt: str, *, split: str = "train") -> RolloutPromptRecord:
    return RolloutPromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        split=split,
        generation={"height": 256, "width": 416, "num_frames": 17},
        safety_audit=_audit(prompt),
    )


def test_rollout_prompt_manifest_is_safe_and_strict(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    records = (
        _record("first", "A red cube rotates slowly."),
        _record("second", "A blue sphere rolls forward."),
        _record("validation", "A green pyramid.", split="validation"),
    )
    path.write_text(
        "".join(canonical_json(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )

    dataset = RolloutPromptDataset.from_file(path, split="train")
    restored = RolloutPromptRecord.from_mapping(dataset[0].to_dict())

    assert restored == dataset[0]
    assert dataset.sample_ids == ("first", "second")
    assert set(restored.to_dict()) == {
        "schema",
        "prompt_id",
        "prompt",
        "split",
        "generation",
        "safety_audit",
    }


def test_rollout_prompt_manifest_rejects_tampering_and_duplicate_groups(tmp_path) -> None:
    record = _record("first", "A red cube rotates slowly.")
    payload = record.to_dict()
    payload["prompt"] = "different text"
    with pytest.raises(ValueError, match="differs from its safety audit"):
        RolloutPromptRecord.from_mapping(payload)

    for removed_field, value in (
        ("metadata", {"source": "unused"}),
        ("reward_suite", "unused"),
    ):
        stale_payload = record.to_dict()
        stale_payload[removed_field] = value
        with pytest.raises(ValueError, match="fields mismatch"):
            RolloutPromptRecord.from_mapping(stale_payload)

    path = tmp_path / "duplicates.jsonl"
    duplicate = _record("second", record.prompt)
    path.write_text(
        json.dumps(record.to_dict()) + "\n" + json.dumps(duplicate.to_dict()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prompt text"):
        RolloutPromptDataset.from_file(path)


def test_rollout_prompt_record_rejects_unsafe_audit() -> None:
    prompt = "A test prompt."
    unsafe = PromptSafetyAudit(
        prompt=prompt,
        unsafe_probabilities={key: (0.9 if key == "dangerous" else 0.01) for key in SHIELDGEMMA_PROMPT_POLICIES},
        threshold=0.5,
    )

    with pytest.raises(ValueError, match="unsafe rollout"):
        RolloutPromptRecord(
            prompt_id="unsafe",
            prompt=prompt,
            safety_audit=unsafe,
        )
