from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.training.data import (
    RolloutPromptAuditResult,
    RolloutPromptDataset,
    audit_rollout_prompts,
)
from worldfoundry.training.safety import PromptSafetyAudit, ShieldGemmaPromptFilter


class _StaticSafeFilter(ShieldGemmaPromptFilter):
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def require_safe(self, prompts):
        values = tuple(prompts)
        self.batches.append(values)
        return tuple(
            PromptSafetyAudit(
                prompt=prompt,
                unsafe_probabilities={
                    "dangerous": 0.01,
                    "harassment": 0.02,
                    "hate": 0.03,
                    "sexually-explicit": 0.04,
                },
                threshold=0.5,
            )
            for prompt in values
        )


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_rollout_prompt_audit_builds_the_cache_consumable_manifest(tmp_path) -> None:
    source = tmp_path / "prompts.jsonl"
    destination = tmp_path / "audited.jsonl"
    _write_source(
        source,
        [
            {"prompt_id": "red-cube", "prompt": "A red cube rolls slowly."},
            {
                "prompt_id": "blue-sphere",
                "prompt": "A blue sphere bounces gently.",
                "split": "train",
                "generation": {"height": 256, "width": 416, "num_frames": 17},
            },
        ],
    )
    prompt_filter = _StaticSafeFilter()

    result = audit_rollout_prompts(
        source_path=source,
        output_manifest_path=destination,
        prompt_filter=prompt_filter,
        batch_size=1,
    )

    assert isinstance(result, RolloutPromptAuditResult)
    assert result.manifest_path == destination.resolve()
    assert prompt_filter.batches == [
        ("A red cube rolls slowly.",),
        ("A blue sphere bounces gently.",),
    ]
    dataset = RolloutPromptDataset.from_file(destination)
    assert dataset.sample_ids == ("red-cube", "blue-sphere")
    assert dataset[0].generation == {}
    assert dataset[1].generation == {
        "height": 256,
        "width": 416,
        "num_frames": 17,
    }
    assert all(record.safety_audit.safe for record in dataset)


def test_rollout_prompt_audit_rejects_ambiguous_input_before_model_execution(tmp_path) -> None:
    source = tmp_path / "prompts.jsonl"
    destination = tmp_path / "audited.jsonl"
    _write_source(
        source,
        [
            {"prompt_id": "duplicate", "prompt": "first"},
            {"prompt_id": "duplicate", "prompt": "second"},
        ],
    )
    prompt_filter = _StaticSafeFilter()

    with pytest.raises(ValueError, match="prompt_id values must be unique"):
        audit_rollout_prompts(
            source_path=source,
            output_manifest_path=destination,
            prompt_filter=prompt_filter,
        )

    assert prompt_filter.batches == []
    assert not destination.exists()


def test_rollout_prompt_audit_never_overwrites_an_existing_manifest(tmp_path) -> None:
    source = tmp_path / "prompts.jsonl"
    destination = tmp_path / "audited.jsonl"
    _write_source(source, [{"prompt_id": "sample", "prompt": "a prompt"}])
    destination.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        audit_rollout_prompts(
            source_path=source,
            output_manifest_path=destination,
            prompt_filter=_StaticSafeFilter(),
        )

    assert destination.read_text(encoding="utf-8") == "existing\n"
