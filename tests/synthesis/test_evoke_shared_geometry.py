from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from worldfoundry.synthesis.visual_generation.evoke.evoke_runtime.evoke.modules.geometric_state import (
    vigeo_cloud,
)


def test_vigeo_shared_loader_and_stream_reset(monkeypatch, tmp_path):
    from worldfoundry.base_models.three_dimensions.depth.vigeo import ViGeo

    (tmp_path / "vigeo.pt").touch()
    model = Mock(mask_head=None)
    model.to.return_value = model
    model.eval.return_value = model
    load = Mock(return_value=model)
    monkeypatch.setattr(ViGeo, "from_pretrained", load)
    monkeypatch.setitem(sys.modules, "vigeo", ModuleType("vigeo"))
    paths = list(sys.path)

    estimator = vigeo_cloud.ViGeoDepthEstimator(device="cpu", weights=tmp_path)
    estimator._lazy()
    estimator._lazy()
    load.assert_called_once_with(str(tmp_path))
    assert estimator._model is model
    assert sys.path == paths

    estimator._kv = object()
    estimator._scale_locked = 2.0
    estimator._anchor_scales = [2.0]
    estimator.reset_stream()
    assert estimator._kv is None
    assert estimator._scale_locked is None
    assert estimator._anchor_scales == []
    model.reset_cache_state.assert_called_once_with()


def test_vigeo_preflight_checks_shared_source_and_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="ViGeo weights"):
        vigeo_cloud.check_assets(weights=tmp_path)
    (tmp_path / "vigeo.pt").touch()
    vigeo_cloud.check_assets(weights=tmp_path)
    with pytest.raises(ValueError, match="shared ViGeo"):
        vigeo_cloud.check_assets(src=tmp_path, weights=tmp_path)


def test_vigeo_local_checkpoint_loader(tmp_path):
    import torch
    from worldfoundry.base_models.three_dimensions.depth.vigeo import ViGeo, ViGeoModel

    assert ViGeoModel is ViGeo

    class TinyViGeo(ViGeo):
        def __init__(self, encoder, with_mask_head):
            torch.nn.Module.__init__(self)
            assert encoder == "vits"
            assert with_mask_head
            self.mask_head = torch.nn.Linear(1, 1, bias=False)

    value = torch.tensor([[2.0]])
    torch.save({"model": {"model.mask_head.weight": value}}, tmp_path / "vigeo.pt")
    model = TinyViGeo.from_pretrained(tmp_path, encoder="vits")
    torch.testing.assert_close(model.mask_head.weight, value)


def test_vigeo_capability_points_to_shared_package():
    from pathlib import Path
    from worldfoundry.base_models.capabilities import get_base_model_capability
    from worldfoundry.base_models.three_dimensions.depth import vigeo

    capability = get_base_model_capability("vigeo")
    assert capability.canonical_owner == vigeo.__name__
    assert Path(capability.canonical_path).resolve() == Path(vigeo.__file__).parent.resolve()
    assert capability.assets[0].required_files == ("vigeo.pt",)
