from __future__ import annotations

import pytest
from fastapi import HTTPException

from worldfoundry.studio.serving import workspace as workspace_app


def test_wan22_studio_exposes_audited_acceleration_controls() -> None:
    entry = workspace_app.find_entry("wan2.2-ti2v-5b")
    options = workspace_app._entry_runtime_options(entry)

    expected_boolean_defaults = {
        "fuse_qkv": False,
        "inplace_residual": False,
        "static_cross_kv": True,
        "fused_rope": False,
    }
    for key, default in expected_boolean_defaults.items():
        assert options[key]["supported"] is True
        assert options[key]["kind"] == "boolean"
        assert options[key]["default"] is default

    assert options["qkv_strategy"]["choices"] == (
        "auto",
        "packed",
        "split",
    )
    assert options["qkv_strategy"]["default"] == "auto"
    assert options["qkv_split_threshold"]["default"] == 8192
    assert options["rope_precision"]["choices"] == ("fp32", "fp64")
    assert options["rope_precision"]["default"] == "fp32"
    assert options["rms_norm_precision"]["choices"] == ("input", "fp32")
    assert options["rms_norm_precision"]["default"] == "input"


def test_wan22_studio_maps_acceleration_controls_to_loader_kwargs() -> None:
    entry = workspace_app.find_entry("wan2.2-ti2v-5b")
    _, load_kwargs = workspace_app._merge_common_params(
        entry,
        workspace_app.JobCreateRequest(
            model_id=entry.model_id,
            params={
                "fuse_qkv": True,
                "qkv_strategy": "split",
                "qkv_split_threshold": 4096,
                "inplace_residual": True,
                "static_cross_kv": True,
                "fused_rope": False,
                "rope_precision": "fp32",
                "rms_norm_precision": "input",
            },
        ),
    )

    assert load_kwargs == {
        "fuse_qkv": True,
        "qkv_strategy": "split",
        "qkv_split_threshold": 4096,
        "inplace_residual": True,
        "static_cross_kv": True,
        "fused_rope": False,
        "rope_precision": "fp32",
        "rms_norm_precision": "input",
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("qkv_strategy", "benchmark", "qkv_strategy must be one of"),
        ("qkv_split_threshold", 0, "qkv_split_threshold must be an integer"),
        ("rope_precision", "bf16", "rope_precision must be one of"),
        ("rms_norm_precision", "fp16", "rms_norm_precision must be one of"),
    ),
)
def test_wan22_studio_rejects_invalid_acceleration_values(
    key: str,
    value: object,
    message: str,
) -> None:
    entry = workspace_app.find_entry("wan2.2-ti2v-5b")

    with pytest.raises(HTTPException, match=message):
        workspace_app._validate_runtime_options(entry, {key: value})


def test_workspace_html_wires_wan22_acceleration_controls() -> None:
    for control_id in (
        "fuseQkv",
        "qkvStrategy",
        "qkvSplitThreshold",
        "inplaceResidual",
        "staticCrossKv",
        "fusedRope",
        "ropePrecision",
        "rmsNormPrecision",
    ):
        assert f'id="{control_id}"' in workspace_app.WORKSPACE_HTML
    assert 'fuseQkv: "fuse_qkv"' in workspace_app.WORKSPACE_HTML
    assert 'qkvStrategy: {key: "qkv_strategy"' in workspace_app.WORKSPACE_HTML
