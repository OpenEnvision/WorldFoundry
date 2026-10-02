from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    _WAN_DENOISER_OPTION_KEYS,
)
from worldfoundry.base_models.diffusion_model.optimizations.wan.feature_cache import (
    build_wan_feature_cache,
    resolve_wan_feature_cache,
)
from worldfoundry.core.acceleration.cache import (
    AdaCacheResidualCache,
    BlockTaylorSeerCache,
    CustomTaylorResidualCache,
    DualBlockFeatureCache,
    DynamicBlockFeatureCache,
    FirstBlockFeatureCache,
    MagCacheResidualCache,
    TaylorSeerResidualCache,
    TeaCacheResidualCache,
)


def _context(
    model_id: str = "wan2.2-ti2v-5b",
    *,
    component_options: dict[str, object] | None = None,
    **options: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        component_options=component_options or {},
        policy=SimpleNamespace(options=options),
    )


def test_wan22_ti2v_teacache_uses_published_embedding_calibration() -> None:
    config = resolve_wan_feature_cache(_context(teacache=True))
    assert config is not None
    assert config.algorithm == "teacache"
    assert config.options["signal"] == "time-embedding"
    assert len(config.options["coefficients"]) == 6
    cache = build_wan_feature_cache(config, branch="positive", total_steps=50)
    assert isinstance(cache, TeaCacheResidualCache)
    assert cache.signal_kind == "time-embedding"


def test_feature_cache_algorithms_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_wan_feature_cache(_context(teacache=True, adacache=True))


def test_unknown_teacache_model_requires_explicit_coefficients() -> None:
    with pytest.raises(ValueError, match="no published coefficients"):
        resolve_wan_feature_cache(_context("custom-wan", teacache=True))


def test_magcache_requires_calibration_and_selects_cfg_row() -> None:
    with pytest.raises(ValueError, match="calibrated"):
        resolve_wan_feature_cache(_context(magcache=True))
    config = resolve_wan_feature_cache(
        _context(
            magcache={
                "ratios": [[1.0] * 4, [0.9] * 4],
                "threshold": 0.2,
                "max_skip_steps": 2,
            }
        )
    )
    assert config is not None
    positive = build_wan_feature_cache(config, branch="positive", total_steps=4)
    negative = build_wan_feature_cache(config, branch="negative", total_steps=4)
    assert isinstance(positive, MagCacheResidualCache)
    assert positive.ratios[0] == 1.0
    assert negative.ratios[0] == 0.9


@pytest.mark.parametrize(
    ("option", "expected_type"),
    (("adacache", AdaCacheResidualCache), ("taylorseer", TaylorSeerResidualCache)),
)
def test_dynamic_cache_configs_construct_real_runtime_cache(option, expected_type) -> None:
    config = resolve_wan_feature_cache(_context(**{option: True}))
    assert config is not None
    cache = build_wan_feature_cache(config, branch="positive", total_steps=8)
    assert isinstance(cache, expected_type)


def test_lightx2v_block_taylor_has_explicit_identity_and_reference_schedule() -> None:
    assert "blocktaylorseer" in _WAN_DENOISER_OPTION_KEYS
    config = resolve_wan_feature_cache(
        _context(feature_cache="taylorseer-block")
    )
    assert config is not None
    assert config.algorithm == "blocktaylorseer"
    assert config.options["dense_pattern"] == (True, False, False, False)
    cache = build_wan_feature_cache(config, branch="negative", total_steps=8)
    assert isinstance(cache, BlockTaylorSeerCache)
    assert cache.branch == "negative"
    assert cache.algorithm != TaylorSeerResidualCache.algorithm


def test_lightx2v_custom_uses_calibrated_tea_gate_and_taylor_prediction() -> None:
    assert "custom" in _WAN_DENOISER_OPTION_KEYS
    config = resolve_wan_feature_cache(_context(custom=True))
    assert config is not None
    assert config.algorithm == "custom"
    assert config.options["signal"] == "time-embedding"
    assert config.options["warmup_steps"] == 1
    assert config.options["dense_last"] == 1
    cache = build_wan_feature_cache(config, branch="positive", total_steps=8)
    assert isinstance(cache, CustomTaylorResidualCache)
    assert cache.branch == "positive"
    assert cache.receipt()["decision"] == "teacache-polynomial"


@pytest.mark.parametrize(
    "bad_pattern",
    ((), (False, False), (True, 0), "dense,skip"),
)
def test_block_taylor_rejects_non_reference_safe_schedule_shapes(bad_pattern) -> None:
    with pytest.raises((TypeError, ValueError), match="dense_pattern"):
        resolve_wan_feature_cache(
            _context(
                feature_cache={
                    "algorithm": "blocktaylorseer",
                    "dense_pattern": bad_pattern,
                }
            )
        )


@pytest.mark.parametrize(
    ("algorithm", "expected_type"),
    (
        ("FirstBlock", FirstBlockFeatureCache),
        ("dual-block", DualBlockFeatureCache),
        ("dynamic_block", DynamicBlockFeatureCache),
    ),
)
def test_block_cache_configs_require_calibration_and_build_branch_local_instances(
    algorithm,
    expected_type,
) -> None:
    with pytest.raises(ValueError, match="residual_diff_threshold"):
        resolve_wan_feature_cache(_context(feature_cache=algorithm))

    config = resolve_wan_feature_cache(
        _context(
            feature_cache={
                "algorithm": algorithm,
                "residual_diff_threshold": 0.1,
                "downsample_factor": 2,
            }
        )
    )
    assert config is not None
    positive = build_wan_feature_cache(config, branch="positive", total_steps=8)
    negative = build_wan_feature_cache(config, branch="negative", total_steps=8)
    assert isinstance(positive, expected_type)
    assert isinstance(negative, expected_type)
    assert positive is not negative
    assert positive.branch == "positive"
    assert negative.branch == "negative"
    assert positive.receipt()["branch"] == "positive"


@pytest.mark.parametrize("algorithm", ("firstblock", "dualblock", "dynamicblock"))
def test_direct_block_cache_component_options_are_resolved(algorithm: str) -> None:
    assert algorithm in _WAN_DENOISER_OPTION_KEYS
    config = resolve_wan_feature_cache(
        _context(
            component_options={
                algorithm: {
                    "residual_diff_threshold": 0.2,
                    "downsample_factor": 2,
                    "dense_first": 2,
                    "dense_last": 1,
                }
            }
        )
    )
    assert config is not None
    assert config.algorithm == algorithm
    cache = build_wan_feature_cache(config, branch="positive", total_steps=8)
    assert cache.residual_diff_threshold == 0.2
    assert cache.downsample_factor == 2
    assert cache.dense_first == 2
    assert cache.dense_last == 1


@pytest.mark.parametrize("downsample_factor", (0, -1, 1.5, True))
def test_block_cache_rejects_invalid_downsample_factor(downsample_factor) -> None:
    error = (TypeError, ValueError)
    with pytest.raises(error, match="downsample_factor"):
        resolve_wan_feature_cache(
            _context(
                feature_cache={
                    "algorithm": "firstblock",
                    "residual_diff_threshold": 0.1,
                    "downsample_factor": downsample_factor,
                }
            )
        )


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("dense_first", 0),
        ("dense_first", True),
        ("dense_last", -1),
        ("dense_last", 1.5),
    ),
)
def test_block_cache_rejects_invalid_dense_boundaries(option: str, value) -> None:
    with pytest.raises((TypeError, ValueError), match=option):
        resolve_wan_feature_cache(
            _context(
                feature_cache={
                    "algorithm": "dynamicblock",
                    "residual_diff_threshold": 0.1,
                    option: value,
                }
            )
        )


def test_block_cache_rejects_unknown_options_instead_of_silently_ignoring_them() -> None:
    with pytest.raises(ValueError, match="unsupported firstblock options"):
        resolve_wan_feature_cache(
            _context(
                feature_cache={
                    "algorithm": "firstblock",
                    "residual_diff_threshold": 0.1,
                    "dense_lsat": 1,
                }
            )
        )


def test_dualblock_partition_is_validated_before_any_block_executes() -> None:
    config = resolve_wan_feature_cache(
        _context(
            feature_cache={
                "algorithm": "dualblock",
                "residual_diff_threshold": 0.1,
            }
        )
    )
    assert config is not None
    cache = build_wan_feature_cache(config, branch="positive", total_steps=8)
    calls = [0]

    def run_block(_block_id, hidden):
        calls[0] += 1
        return hidden

    with torch.no_grad(), pytest.raises(ValueError, match="at least 11 blocks"):
        cache.run_blocks(0, torch.zeros(1, 2, 4), run_block, block_count=10)
    assert calls[0] == 0
