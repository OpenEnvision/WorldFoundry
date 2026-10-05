"""CPU-only regression tests for evaluation-framework review fixes.

Covers EF-35 (redact_secrets word-boundary matching) and EF-04
(missing/blank generation status fail-closed).
"""

from __future__ import annotations

from worldfoundry.evaluation.api import (
    GENERATION_UNKNOWN_STATUS,
    GenerationResult,
    is_generation_result_successful,
    is_generation_status_successful,
    normalize_generation_status,
)
from worldfoundry.evaluation.models.pipelines.results import pipeline_result_status
from worldfoundry.evaluation.reporting.run_manifest import redact_secrets


# ── EF-35: redact_secrets ───────────────────────────────────────────────


def test_redact_preserves_generation_parameters() -> None:
    payload = {
        "max_new_tokens": 512,
        "max_tokens": 128,
        "num_tokens": 8,
        "tokenizer": "gpt2",
        "tokenizer_path": "/models/gpt2/tokenizer.json",
        "top_k": 50,
        "keyword": "search",
        "cache_key": "abc",
        "monkey": "see",
    }

    assert redact_secrets(payload) == payload


def test_redact_still_hides_real_secrets() -> None:
    payload = {
        "api_key": "sk-123",
        "apiKey": "sk-camel",
        "authToken": "camel-token",
        "hf_token": "hf_abc",
        "token": "raw",
        "access_token": "at",
        "auth_token": "auth",
        "bearer_token": "bt",
        "client_secret": "cs",
        "password": "pw",
        "PASSWORD": "pw-upper",
        "private_key": "pk",
        "ssh-key": "sk",
        "credentials": "cred",
        "aws_secret_access_key": "aws",
    }

    redacted = redact_secrets(payload)

    assert all(value == "<redacted>" for value in redacted.values()), redacted


def test_redact_recurses_and_keeps_structure() -> None:
    redacted = redact_secrets(
        {
            "generation": {"max_new_tokens": 16, "api_key": "x"},
            "items": [{"hf_token": "t", "tokenizer": "llama"}],
        }
    )

    assert redacted == {
        "generation": {"max_new_tokens": 16, "api_key": "<redacted>"},
        "items": [{"hf_token": "<redacted>", "tokenizer": "llama"}],
    }


# ── EF-04: missing status fail-closed ──────────────────────────────────


def test_missing_status_normalizes_to_unknown() -> None:
    assert normalize_generation_status(None) == GENERATION_UNKNOWN_STATUS
    assert normalize_generation_status("") == GENERATION_UNKNOWN_STATUS
    assert normalize_generation_status("   ") == GENERATION_UNKNOWN_STATUS
    assert normalize_generation_status("SUCCEEDED") == "succeeded"


def test_missing_or_unknown_status_is_not_successful() -> None:
    assert not is_generation_status_successful(None)
    assert not is_generation_status_successful("")
    assert not is_generation_status_successful("unknown")
    assert is_generation_status_successful("ok")
    assert is_generation_status_successful("succeeded")


def test_result_row_without_status_is_not_scoreable() -> None:
    incomplete = GenerationResult.from_dict({"sample_id": "crashed-sample"})

    assert incomplete.status == GENERATION_UNKNOWN_STATUS
    assert not is_generation_result_successful(incomplete)


def test_serialized_result_round_trip_keeps_success() -> None:
    original = GenerationResult(sample_id="s1")
    restored = GenerationResult.from_dict(original.to_dict())

    assert original.status == "succeeded"
    assert restored.status == "succeeded"
    assert is_generation_result_successful(restored)


def test_in_process_pipeline_mapping_without_status_stays_successful() -> None:
    # Pipelines signal failure by raising; a normal return without an
    # explicit status is success (EF-04 must not break this path).
    assert pipeline_result_status({}) == "succeeded"
    assert pipeline_result_status({"status": "blocked"}) == "blocked"
    assert pipeline_result_status({"status": "OK"}) == "succeeded"
