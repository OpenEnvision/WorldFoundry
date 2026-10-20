# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU numerical contracts for shared quality components; no pretrained assets."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import (
    load_laion_aesthetic_linear_head,
    load_laion_aesthetic_mlp_head,
)
from worldfoundry.evaluation.tasks.metrics._shared.pyiqa import ImageQualityMetric, PyIQAMetric

pytestmark = pytest.mark.unit


@pytest.fixture
def torch_cpu() -> Any:
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(previous)


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_shared_laion_head_matches_reference_weights_and_predictions(tmp_path: Path, torch_cpu: Any, head: str) -> None:
    torch, nn = torch_cpu, torch_cpu.nn
    if head == "linear":
        reference = nn.Linear(768, 1).eval()
        state = reference.state_dict()
        load = load_laion_aesthetic_linear_head
    else:
        reference = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ).eval()
        state = {f"layers.{key}": value for key, value in reference.state_dict().items()}
        load = load_laion_aesthetic_mlp_head
    checkpoint = tmp_path / "head.pth"
    torch.save(state, checkpoint)
    actual = load(checkpoint)
    features = torch.arange(3 * 768, dtype=torch.float32).reshape(3, 768) / 768
    with torch.inference_mode():
        torch.testing.assert_close(actual(features), reference(features), rtol=0, atol=0)
    assert not actual.training


def test_iqa_preprocessing_matches_official_resize_for_paths_and_pil(tmp_path: Path, torch_cpu: Any) -> None:
    from torchvision import transforms

    class Metric(ImageQualityMetric):
        def _compute_scores(self, *args: Any) -> None:
            pass

    image = Image.fromarray(np.arange(19 * 31 * 3, dtype=np.uint8).reshape(19, 31, 3))
    path = tmp_path / "image.png"
    image.save(path)
    metric = Metric(device="cpu")
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    expected = torch_cpu.stack([transform(image), transform(image)])
    torch_cpu.testing.assert_close(metric._process_image([str(path), image]), expected, rtol=0, atol=0)
    for reference in (str(path), image):
        rendered, references = metric._process_images([str(path), image], reference)
        torch_cpu.testing.assert_close(rendered, expected, rtol=0, atol=0)
        torch_cpu.testing.assert_close(references, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["musiq", "musiq-spaq", "musiq-paq2piq"])
def test_shared_iqa_preserves_the_selected_checkpoint_variant(monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    class Metric(PyIQAMetric):
        def _compute_scores(self, *args: Any) -> None:
            pass

    calls: list[str] = []
    model = SimpleNamespace(to=lambda device: device)
    monkeypatch.setitem(sys.modules, "pyiqa", SimpleNamespace(create_metric=lambda name: calls.append(name) or model))
    metric = Metric(variant, device="cpu")
    assert calls == [variant]
    assert metric._metric == "cpu"


def test_dinov2_loader_preserves_explicit_cls_and_pooler_semantics(
    monkeypatch: pytest.MonkeyPatch, torch_cpu: Any
) -> None:
    from worldfoundry.base_models.perception_core.general_perception.dino_embeddings import load_dinov2_feature_model

    class Model(torch_cpu.nn.Module):
        def forward(self, pixel_values: Any) -> Any:
            return SimpleNamespace(last_hidden_state=pixel_values, pooler_output=pixel_values[:, 1])

    calls: list[tuple[str, bool]] = []

    def load(source: str, *, local_files_only: bool) -> Any:
        calls.append((source, local_files_only))
        return Model()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)))
    features = torch_cpu.arange(12, dtype=torch_cpu.float32).reshape(2, 3, 2)
    cls_model = load_dinov2_feature_model("local-dinov2-large", device="cpu")
    pooler_model = load_dinov2_feature_model("local-dinov2-base", use_pooler_output=True)
    torch_cpu.testing.assert_close(cls_model(features), features[:, 0], rtol=0, atol=0)
    torch_cpu.testing.assert_close(pooler_model(features), features[:, 1], rtol=0, atol=0)
    assert calls == [("local-dinov2-large", True), ("local-dinov2-base", True)]
    assert not cls_model.training


def test_masked_dino_batch_and_single_paths_keep_the_official_transform(torch_cpu: Any) -> None:
    from torchvision import transforms

    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.features import (
        MaskedDinoV2Features,
    )

    class Model(torch_cpu.nn.Module):
        def forward(self, tensor: Any) -> Any:
            assert tensor.shape[-2:] == (14, 14)
            return tensor.mean(dim=(-2, -1))

    model = Model().eval()
    extractor = MaskedDinoV2Features(model=model, device="cpu", input_size=14, batch_size=2)
    frame = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:6, 2:8] = True
    mask[2, 3] = False
    crop = frame[1:6, 2:8].copy()
    crop[1, 1] = 0
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((14, 14), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    reference = model(transform(crop).unsqueeze(0))[0].numpy()
    reference = reference / float(np.linalg.norm(reference))
    result = extractor.embed_batch([(frame, np.zeros_like(mask)), (frame, mask)])
    assert result[0] is None
    assert result[1] is not None
    single = extractor.embed(frame, mask)
    assert single is not None
    np.testing.assert_array_equal(result[1], reference)
    np.testing.assert_array_equal(single, reference)


def test_memory_protocols_share_backbone_but_not_batch_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_cpu: Any
) -> None:
    from worldfoundry.base_models.perception_core.general_perception import dino_embeddings
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.features import get_dino_model
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.human.evaluator import (
        _dino_extractor,
    )
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.object.evaluator import (
        _feature_extractor,
    )

    checkpoint = tmp_path / "dinov2-large"
    calls: list[Path] = []

    def load(path: Path, *, device: str) -> Any:
        assert device == "cpu"
        calls.append(path)
        return torch_cpu.nn.Identity()

    monkeypatch.setattr(dino_embeddings, "load_dinov2_feature_model", load)
    context = EvaluationContext(
        device="cpu",
        cache_dir=tmp_path / "assets",
        config={
            "metrics": {
                "human_appearance": {"dino_checkpoint": str(checkpoint)},
                "object_appearance": {"dino_checkpoint": str(checkpoint)},
            }
        },
    )
    try:
        human = _dino_extractor(context)
        obj = _feature_extractor(context)
        assert human is not obj
        assert human.model is obj.model is get_dino_model(context, checkpoint)
        assert human.effective_batch_size > obj.effective_batch_size == 1
        assert len(calls) == 1
        assert get_dino_model(context, tmp_path / "other-checkpoint") is not human.model
        assert len(calls) == 2
    finally:
        context.close()


def test_wbench_binds_the_official_koniq_asset_without_global_cache_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_cpu: Any
) -> None:
    from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES
    from worldfoundry.evaluation.tasks.execution.runners.wbench.runtime.wbench.src.metrics.video_quality import (
        imaging_quality,
    )

    checkpoint = tmp_path / "musiq_koniq_ckpt-e95806b9.pth"
    asset_ids: list[str] = []
    calls: list[tuple[str, dict[str, Any]]] = []
    model = object()

    def create(name: str, **kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "pyiqa", SimpleNamespace(create_metric=create))
    monkeypatch.setattr(imaging_quality, "wbench_asset_path", lambda name: asset_ids.append(name) or checkpoint)
    metric = imaging_quality.ImagingQualityMetric(device="cpu")
    assert metric.model is model
    assert calls == [("musiq", {"device": "cpu", "pretrained_model_path": str(checkpoint)})]
    assert asset_ids == ["wbench_musiq_koniq_checkpoint"]
    asset = next(
        asset for asset in BASE_MODEL_CAPABILITIES["wbench_quality_metric_assets"].assets if asset.id == asset_ids[0]
    )
    assert str(asset.local_path).endswith(checkpoint.name)


@pytest.mark.parametrize("height,width", [(180, 320), (720, 1280)])
def test_worldmembench_spaq_preserves_opencv_aspect_resize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_cpu: Any, height: int, width: int
) -> None:
    cv2 = pytest.importorskip("cv2")
    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.general_video_quality.musiq import (
        MusiqSpaqBackend,
    )

    captured: list[Any] = []

    class Model(torch_cpu.nn.Module):
        def forward(self, tensor: Any) -> Any:
            captured.append(tensor.clone())
            return torch_cpu.full((len(tensor),), 50.0)

    checkpoint = tmp_path / "musiq_spaq_ckpt-358bb6af.pth"
    checkpoint.touch()
    model = Model()

    def load(*, pretrained_model_path: str) -> Any:
        assert pretrained_model_path == str(checkpoint)
        return model

    monkeypatch.setitem(sys.modules, "pyiqa.archs.musiq_arch", SimpleNamespace(MUSIQ=load))
    backend = MusiqSpaqBackend(checkpoint=checkpoint, device="cpu")
    assert not backend.model.training
    frame = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    expected = frame
    if max(height, width) > 512:
        scale = 512 / max(height, width)
        expected = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
    tensor = torch_cpu.from_numpy(expected.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
    assert backend.score([frame]) == [50.0]
    torch_cpu.testing.assert_close(captured[0], tensor, rtol=0, atol=0)
