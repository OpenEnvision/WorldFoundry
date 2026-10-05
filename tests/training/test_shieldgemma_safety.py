from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.data import (  # noqa: E402
    PromptAuditSet,
    TrainingManifestDataset,
    audit_training_manifest_prompts,
)
from worldfoundry.training.safety import (  # noqa: E402
    SHIELDGEMMA_PROMPT_AUDIT_SCHEMA,
    PromptSafetyAudit,
    ShieldGemmaPromptFilter,
    UnsafeTrainingPromptError,
    shieldgemma_checkpoint_spec,
)


class _FakeTokenizer:
    padding_side = "right"

    def get_vocab(self) -> dict[str, int]:
        return {"Yes": 1, "No": 2}

    def __call__(self, texts, **kwargs):
        del kwargs
        count = len(texts)
        return {
            "input_ids": torch.ones(count, 3, dtype=torch.long),
            "attention_mask": torch.ones(count, 3, dtype=torch.long),
        }


class _FakeShieldModel(torch.nn.Module):
    def __init__(self, *, unsafe: bool) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.unsafe = unsafe

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        logits[:, -1, 1] = 8.0 if self.unsafe else -8.0
        logits[:, -1, 2] = -8.0 if self.unsafe else 8.0
        return SimpleNamespace(logits=logits)


def test_shieldgemma_safe_audit_records_prompt_and_freezes_model() -> None:
    model = _FakeShieldModel(unsafe=False)
    prompt_filter = ShieldGemmaPromptFilter(model, _FakeTokenizer())

    (audit,) = prompt_filter.require_safe(("a blue cup on a table",))

    assert audit.schema == SHIELDGEMMA_PROMPT_AUDIT_SCHEMA
    assert audit.safe is True
    assert audit.blocked_categories == ()
    assert audit.prompt == "a blue cup on a table"
    assert model.training is False
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert prompt_filter.tokenizer.padding_side == "left"


def test_shieldgemma_rejects_unsafe_prompts() -> None:
    prompt = "private unsafe fixture"
    prompt_filter = ShieldGemmaPromptFilter(_FakeShieldModel(unsafe=True), _FakeTokenizer())

    with pytest.raises(UnsafeTrainingPromptError) as captured:
        prompt_filter.require_safe((prompt,))

    assert prompt in str(captured.value)
    assert captured.value.audits[0].blocked_categories


def test_shieldgemma_refuses_to_truncate_unscored_content() -> None:
    prompt_filter = ShieldGemmaPromptFilter(
        _FakeShieldModel(unsafe=False),
        _FakeTokenizer(),
        max_input_tokens=2,
    )

    with pytest.raises(ValueError, match="refuses to truncate"):
        prompt_filter.audit(("content",))


def test_shieldgemma_checkpoint_is_pinned() -> None:
    checkpoint = shieldgemma_checkpoint_spec()

    assert checkpoint.revision == "d1dffc9c8c9237a90aab09c61383791e718ef9e8"
    assert set(checkpoint.files) == set(checkpoint.file_size_bytes)


def test_prompt_audit_sidecar_builds_and_binds_a_new_manifest(tmp_path) -> None:
    media = tmp_path / "sample.bin"
    media.write_bytes(b"safe training fixture")
    source = tmp_path / "manifest.jsonl"
    row = {
        "schema": "worldfoundry-training-sample",
        "sample_id": "sample",
        "task": "text_to_image",
        "prompt": "a blue cup on a table",
        "media": {
            "uri": media.name,
            "size_bytes": media.stat().st_size,
        },
        "width": 8,
        "height": 8,
        "num_frames": 1,
        "fps": 1,
        "conditions": {},
        "split": "train",
        "safety": {},
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "audited" / "manifest.jsonl"
    sidecar = tmp_path / "audited" / "prompt-audits.json"

    result = audit_training_manifest_prompts(
        manifest_path=source,
        output_manifest_path=output,
        output_audit_path=sidecar,
        prompt_filter=ShieldGemmaPromptFilter(
            _FakeShieldModel(unsafe=False),
            _FakeTokenizer(),
        ),
        verify_media_files=True,
    )
    manifest = TrainingManifestDataset.from_file(output, split="train", verify_files=True)
    loaded = PromptAuditSet.from_file(sidecar)
    (audit,) = loaded.select_for_manifest(manifest)

    assert result.audit_set == loaded
    assert audit == PromptSafetyAudit.from_mapping(audit.to_dict())
    assert manifest[0].safety["prompt_safe"] is True
    assert manifest[0].safety["model_revision"] == audit.model_revision
    assert manifest[0].media.uri == str(media.resolve())

    tampered = loaded.to_dict()
    tampered["records"][0]["audit"]["safe"] = False
    with pytest.raises(ValueError, match="derived decision"):
        PromptAuditSet.from_mapping(tampered)
