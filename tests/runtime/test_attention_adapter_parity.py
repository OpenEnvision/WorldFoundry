from benchmarks.operators.attention_adapter_parity import backend_resolution_failures


def test_explicit_provider_fallback_fails_the_parity_gate() -> None:
    failures = backend_resolution_failures("flash_attention_3", ["torch"])
    assert failures and "resolved" in failures[0]


def test_flash_family_accepts_fa2_or_fa3() -> None:
    assert backend_resolution_failures("flash_attention", ["flash_attention_2"]) == []
    assert backend_resolution_failures("flash_attention", ["flash_attention_3"]) == []
