from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from worldfoundry.base_models.diffusion_model.contracts import Conditioning  # noqa: E402
from worldfoundry.training.data import (  # noqa: E402
    TRAINING_SAMPLE_SCHEMA,
    SanaCachedDataset,
    SanaCacheStore,
    SanaFeatureEncoder,
    TrainingManifestDataset,
    checkpoint_asset_identity,
    collate_sana_cached_samples,
    prepare_sana_training_cache,
    prompt_enhancement_config,
)
from worldfoundry.training.safety import ShieldGemmaPromptFilter  # noqa: E402


def test_checkpoint_asset_identity_records_repository_revision_files_and_sizes() -> None:
    identity = checkpoint_asset_identity(
        repo_id="owner/model",
        revision="pinned-revision",
        files=("weights", "config"),
        file_size_bytes={"weights": 10, "config": 20},
    )
    assert identity == {
        "repo_id": "owner/model",
        "revision": "pinned-revision",
        "files": ["weights", "config"],
        "file_size_bytes": {"weights": 10, "config": 20},
    }


class _Tokenizer:
    padding_side = "right"

    def get_vocab(self):
        return {"Yes": 1, "No": 2}

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.ones(len(texts), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }


class _SafeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        logits = torch.zeros(*input_ids.shape, 4)
        logits[:, -1, 1] = -8.0
        logits[:, -1, 2] = 8.0
        return SimpleNamespace(logits=logits)


class _Codec:
    scaling_factor = 0.41407

    def __init__(self) -> None:
        self.model = torch.nn.Conv2d(3, 32, kernel_size=32, stride=32, bias=False)

    def encode(self, images):
        return self.model(images) * self.scaling_factor


class _Conditioner:
    max_length = 3

    def __init__(self) -> None:
        self.encoder = torch.nn.Linear(1, 4, bias=False)

    def encode(self, request, *, device, dtype):
        values = torch.ones(request.batch_size, 3, 1, device=device, dtype=dtype)
        return Conditioning(
            positive={
                "context": self.encoder(values)[:, None],
                "context_mask": torch.tensor([[1, 1, 0]], device=device),
            },
            negative={
                "context": self.encoder(torch.zeros_like(values))[:, None],
                "context_mask": torch.tensor([[1, 0, 0]], device=device),
            },
        )


def _manifest(tmp_path: Path, prompt_filter: ShieldGemmaPromptFilter) -> TrainingManifestDataset:
    prompt = "a blue ceramic cup"
    (audit,) = prompt_filter.require_safe((prompt,))
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(32, 64, 128)).save(image_path)
    payload = image_path.read_bytes()
    row = {
        "schema": TRAINING_SAMPLE_SCHEMA,
        "sample_id": "cup",
        "task": "text_to_image",
        "prompt": prompt,
        "media": {
            "uri": image_path.name,
            "size_bytes": len(payload),
        },
        "width": 64,
        "height": 64,
        "num_frames": 1,
        "fps": 1,
        "conditions": {},
        "split": "train",
        "safety": {"prompt_safe": True, "model_revision": audit.model_revision},
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return TrainingManifestDataset.from_file(
        manifest_path,
        split="train",
        verify_files=True,
    )


def test_prepare_sana_cache_runs_safety_decode_and_frozen_feature_path(tmp_path: Path) -> None:
    prompt_filter = ShieldGemmaPromptFilter(_SafeModel(), _Tokenizer())
    manifest = _manifest(tmp_path, prompt_filter)
    codec = _Codec()
    conditioner = _Conditioner()
    encoder = SanaFeatureEncoder(codec, conditioner)
    store = SanaCacheStore(tmp_path / "cache")

    result = prepare_sana_training_cache(
        manifest=manifest,
        store=store,
        feature_encoder=encoder,
        prompt_filter=prompt_filter,
        model_recipe="sana-600m-512px",
        codec={"repo_id": "dcae", "revision": "main"},
        conditioner={"repo_id": "gemma", "revision": "main"},
        tokenizer={"repo_id": "tokenizer", "revision": "main"},
        prompt_enhancement=prompt_enhancement_config(
            enabled=True,
            max_text_length=3,
            prefix="pinned prefix",
        ),
        spatial_compression=32,
        safety_batch_size=1,
    )

    dataset = SanaCachedDataset(
        store.root,
        expected_sample_ids=manifest.sample_ids,
    )
    batch = collate_sana_cached_samples([dataset[0]])
    assert result.index == dataset.index
    assert result.safety_audits[0].safe is True
    assert result.unconditional_conditioning.identity.branch == "unconditional"
    assert batch.conditions["clean_latents"].shape == (1, 32, 2, 2)
    assert batch.conditions["context"].shape == (1, 1, 3, 4)
    assert not any(parameter.requires_grad for parameter in codec.model.parameters())
    assert not any(parameter.requires_grad for parameter in conditioner.encoder.parameters())
    assert "blue ceramic cup" in (store.root / "index.json").read_text(encoding="utf-8")


def test_prepare_sana_cache_refuses_to_overwrite_an_existing_index(tmp_path: Path) -> None:
    prompt_filter = ShieldGemmaPromptFilter(_SafeModel(), _Tokenizer())
    manifest = _manifest(tmp_path, prompt_filter)
    store = SanaCacheStore(tmp_path / "cache")
    (store.root / "index.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="will not overwrite"):
        prepare_sana_training_cache(
            manifest=manifest,
            store=store,
            feature_encoder=SanaFeatureEncoder(_Codec(), _Conditioner()),
            prompt_filter=prompt_filter,
            model_recipe="sana-600m-512px",
            codec={"repo_id": "dcae", "revision": "main"},
            conditioner={"repo_id": "gemma", "revision": "main"},
            tokenizer={"repo_id": "tokenizer", "revision": "main"},
            prompt_enhancement={"enabled": False, "max_text_length": 3, "prefix": ""},
        )
