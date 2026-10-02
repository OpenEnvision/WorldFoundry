from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.reporting import (
    ENVIRONMENT_SCHEMA_VERSION,
    ENV_REQUIREMENTS_SCHEMA_VERSION,
    REPRODUCIBILITY_PACKAGES,
    RUN_MANIFEST_SCHEMA_VERSION,
    is_sensitive_key,
    redact_secret_text,
    redact_secrets,
    validate_contract_file,
    write_run_manifest_artifacts,
)


def test_write_run_manifest_artifacts_redacts_secrets_and_validates(tmp_path: Path) -> None:
    paths = write_run_manifest_artifacts(
        output_dir=tmp_path,
        base_manifest={
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": "run-1",
            "runner": "unit",
            "status": "succeeded",
            "output_dir": str(tmp_path),
            "model": {"model_id": "model-a", "revision": "model-rev"},
            "dataset": {"dataset_id": "dataset-a", "revision": "dataset-rev"},
            "artifacts": {},
        },
        config={"api_key": "should-not-appear", "temperature": 0.0},
        required_env=("OPENAI_API_KEY",),
        required_paths=(tmp_path,),
        cache_paths={"hf_cache": tmp_path / "hf"},
        package_names=("worldfoundry",),
        environ={},
    )

    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    environment = json.loads(paths["environment"].read_text(encoding="utf-8"))
    env_requirements = json.loads(paths["env_requirements"].read_text(encoding="utf-8"))
    manifest_text = paths["run_manifest"].read_text(encoding="utf-8")

    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["environment"]["schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    assert manifest["env_requirements"]["schema_version"] == ENV_REQUIREMENTS_SCHEMA_VERSION
    assert "preflight" not in manifest
    assert manifest["env_requirements"]["missing_env"] == ["OPENAI_API_KEY"]
    assert manifest["config"]["api_key"] == "<redacted>"
    assert manifest["model_revision"] == "model-rev"
    assert manifest["dataset_revision"] == "dataset-rev"
    assert manifest["artifacts"]["environment"] == str(paths["environment"])
    assert manifest["artifacts"]["env_requirements"] == str(paths["env_requirements"])
    assert environment["python"]["version"]
    assert environment["torch"]["version"] is None or isinstance(environment["torch"]["version"], str)
    assert set(environment["cuda"]) == {
        "available",
        "build_version",
        "cudnn_version",
        "device_count",
    }
    assert "torch" in REPRODUCIBILITY_PACKAGES
    assert env_requirements["required_env"] == [{"name": "OPENAI_API_KEY", "present": False, "redacted": True}]
    assert "should-not-appear" not in manifest_text
    assert validate_contract_file(paths["run_manifest"], kind="run-manifest")["ok"] is True


def test_redact_secrets_recurses_nested_config() -> None:
    redacted = redact_secrets(
        {
            "safe": "visible",
            "nested": {
                "client_secret": "hidden",
                "items": [{"auth_token": "hidden-too"}],
            },
        }
    )

    assert redacted == {
        "safe": "visible",
        "nested": {
            "client_secret": "<redacted>",
            "items": [{"auth_token": "<redacted>"}],
        },
    }


def test_redact_secrets_covers_argv_values_and_inline_assignments() -> None:
    redacted = redact_secrets(
        {
            "command": [
                "python",
                "run.py",
                "--gemini-api-key",
                "gemini-secret-value",
                "--safe-option",
                "visible",
                "--token=inline-secret-value",
            ]
        }
    )

    assert redacted == {
        "command": [
            "python",
            "run.py",
            "--gemini-api-key",
            "<redacted>",
            "--safe-option",
            "visible",
            "--token=<redacted>",
        ]
    }


def test_redact_secrets_covers_nested_sequences_and_url_userinfo() -> None:
    redacted = redact_secrets(
        {
            "command": (
                "curl",
                "--provider-api-key",
                "provider-secret-value",
                "https://alice:password@example.test/private/model",
            ),
            "generation": [{"max_new_tokens": 64, "tokenizer": "gpt2"}],
        }
    )

    assert redacted == {
        "command": (
            "curl",
            "--provider-api-key",
            "<redacted>",
            "https://<redacted>@example.test/private/model",
        ),
        "generation": [{"max_new_tokens": 64, "tokenizer": "gpt2"}],
    }


def test_sensitive_key_matching_uses_exact_segments() -> None:
    assert is_sensitive_key("OPENAI_API_KEY") is True
    assert is_sensitive_key("headers.authorization") is True
    assert is_sensitive_key("authToken") is True
    assert is_sensitive_key("service-password") is True
    assert is_sensitive_key("TOKENIZERS_PARALLELISM") is False
    assert is_sensitive_key("MAX_NEW_TOKENS") is False


def test_redact_secret_text_handles_env_argv_assignments_and_url_userinfo() -> None:
    text = (
        "from-env --service-api-key argv-secret --token=inline-secret "
        'password="json-secret" https://alice:url-secret@example.test/path '
        "Authorization: Bearer bearer-secret sk-abcdefghijk "
        "MAX_NEW_TOKENS=64 TOKENIZERS_PARALLELISM=true"
    )

    redacted = redact_secret_text(
        text,
        {
            "PRIVATE_API_KEY": "from-env",
            "TOKENIZERS_PARALLELISM": "true",
        },
    )

    for secret in (
        "from-env",
        "argv-secret",
        "inline-secret",
        "json-secret",
        "url-secret",
        "bearer-secret",
        "sk-abcdefghijk",
    ):
        assert secret not in redacted
    assert "--service-api-key <redacted>" in redacted
    assert "--token=<redacted>" in redacted
    assert "https://<redacted>@example.test/path" in redacted
    assert "MAX_NEW_TOKENS=64" in redacted
    assert "TOKENIZERS_PARALLELISM=true" in redacted
