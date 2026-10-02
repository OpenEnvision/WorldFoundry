import pytest
import torch

from worldfoundry.core.attention.kv_quantization import (
    KVQuantConfig,
    QuantizedKVStore,
    dequantize_kv,
    quantize_kv,
)

HEAD_DIM = 128


def _tensor(dtype: torch.dtype = torch.bfloat16, tokens: int = 37) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, tokens, 8, HEAD_DIM, dtype=dtype)


def _relative_error(restored: torch.Tensor, source: torch.Tensor) -> float:
    return ((restored.float() - source.float()).norm() / source.float().norm()).item()


# ── Config ───────────────────────────────────────────────────


def test_disabled_config_is_a_passthrough_marker() -> None:
    config = KVQuantConfig()

    assert not config.enabled
    assert config.compression_ratio() == 1.0
    with pytest.raises(ValueError, match="requires an enabled"):
        quantize_kv(_tensor(), config)


def test_config_rejects_unknown_dtypes_and_group_sizes() -> None:
    with pytest.raises(ValueError, match="Unsupported KV quantization dtype"):
        KVQuantConfig(dtype="int3")
    with pytest.raises(ValueError, match="group_size must be >= 1"):
        KVQuantConfig(dtype="int8", group_size=0)


def test_bit_widths_match_the_named_format() -> None:
    assert KVQuantConfig("int8").bits == 8
    assert KVQuantConfig("int4").bits == 4
    assert KVQuantConfig("int2").bits == 2


# ── Round trip ───────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ["int8", "int4", "int2"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_round_trip_preserves_shape_and_dtype(dtype: str, symmetric: bool) -> None:
    source = _tensor()

    restored = dequantize_kv(quantize_kv(source, KVQuantConfig(dtype, symmetric=symmetric)))

    assert restored.shape == source.shape
    assert restored.dtype == source.dtype


def test_accuracy_improves_monotonically_with_bit_width() -> None:
    source = _tensor()
    errors = {
        dtype: _relative_error(dequantize_kv(quantize_kv(source, KVQuantConfig(dtype))), source)
        for dtype in ("int2", "int4", "int8")
    }

    assert errors["int8"] < errors["int4"] < errors["int2"]
    # int8 is effectively lossless for attention purposes.
    assert errors["int8"] < 0.02


def test_accuracy_improves_with_smaller_groups() -> None:
    source = _tensor()

    coarse = _relative_error(dequantize_kv(quantize_kv(source, KVQuantConfig("int4", group_size=128))), source)
    fine = _relative_error(dequantize_kv(quantize_kv(source, KVQuantConfig("int4", group_size=16))), source)

    assert fine < coarse


def test_asymmetric_beats_symmetric_on_skewed_data() -> None:
    """Keys have offset per-channel distributions; a zero-centred range wastes codes."""
    source = _tensor() + 4.0

    asymmetric = _relative_error(dequantize_kv(quantize_kv(source, KVQuantConfig("int4"))), source)
    symmetric = _relative_error(dequantize_kv(quantize_kv(source, KVQuantConfig("int4", symmetric=True))), source)

    assert asymmetric < symmetric


def test_constant_groups_survive_a_zero_range() -> None:
    """A degenerate group must not divide by zero."""
    source = torch.full((1, 4, 2, HEAD_DIM), 0.5, dtype=torch.bfloat16)

    restored = dequantize_kv(quantize_kv(source, KVQuantConfig("int4")))

    assert torch.allclose(restored.float(), source.float(), atol=1e-2)


# ── Packing ──────────────────────────────────────────────────


@pytest.mark.parametrize(("dtype", "per_byte"), [("int8", 1), ("int4", 2), ("int2", 4)])
def test_sub_byte_codes_are_really_packed(dtype: str, per_byte: int) -> None:
    source = _tensor()

    quantized = quantize_kv(source, KVQuantConfig(dtype, group_size=128, symmetric=True))

    assert quantized.codes.dtype == torch.uint8
    assert quantized.codes.numel() * per_byte == source.numel()


def test_packing_rejects_unaligned_head_dims() -> None:
    source = torch.randn(1, 4, 2, 100, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="divisible by group_size"):
        quantize_kv(source, KVQuantConfig("int4", group_size=64))


def test_reported_compression_matches_measured_bytes() -> None:
    source = _tensor()
    source_bytes = source.numel() * source.element_size()

    for dtype in ("int8", "int4", "int2"):
        for group_size in (128, 64, 32):
            config = KVQuantConfig(dtype, group_size=group_size)
            measured = source_bytes / quantize_kv(source, config).nbytes()
            assert measured == pytest.approx(config.compression_ratio(), rel=0.02)


def test_nbytes_counts_metadata() -> None:
    quantized = quantize_kv(_tensor(), KVQuantConfig("int4", group_size=32))

    payload = quantized.codes.numel() * quantized.codes.element_size()
    assert quantized.nbytes() > payload


# ── Store ────────────────────────────────────────────────────


def test_store_concatenates_appended_chunks() -> None:
    store = QuantizedKVStore(KVQuantConfig("int8"), seq_dim=1)
    for _ in range(4):
        store.append(_tensor(tokens=10), _tensor(tokens=10))

    keys, values = store.materialize()

    assert store.length == 40
    assert keys.shape[1] == 40
    assert values.shape[1] == 40


def test_store_supports_dense_passthrough() -> None:
    store = QuantizedKVStore(KVQuantConfig(), seq_dim=1)
    keys_in, values_in = _tensor(tokens=6), _tensor(tokens=6)

    store.append(keys_in, values_in)
    keys, values = store.materialize()

    # Disabled quantization must be exact, not merely close.
    assert torch.equal(keys, keys_in)
    assert torch.equal(values, values_in)


def test_store_compacts_to_the_surviving_tokens() -> None:
    store = QuantizedKVStore(KVQuantConfig("int8"), seq_dim=1)
    for _ in range(3):
        store.append(_tensor(tokens=10), _tensor(tokens=10))

    store.compact(torch.arange(0, 30, 3))

    keys, _ = store.materialize()
    assert store.length == 10
    assert keys.shape[1] == 10


def test_store_rejects_materializing_when_empty() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        QuantizedKVStore(KVQuantConfig("int8")).materialize()


def test_store_reset_clears_everything() -> None:
    store = QuantizedKVStore(KVQuantConfig("int8"), seq_dim=1)
    store.append(_tensor(tokens=5), _tensor(tokens=5))

    store.reset()

    assert store.length == 0
    assert store.nbytes() == 0


def test_store_keeps_keys_and_values_at_independent_precision() -> None:
    store = QuantizedKVStore(KVQuantConfig("int8"), KVQuantConfig("int2"), seq_dim=1)
    store.append(_tensor(tokens=8), _tensor(tokens=8))

    keys, values = store.materialize()

    source = _tensor(tokens=8)
    # The coarser value config must be visibly lossier than the key config.
    assert _relative_error(keys, source) < _relative_error(values, source)
