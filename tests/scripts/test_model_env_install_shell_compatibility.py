from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bash_function(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}$", source, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"missing shell function {name}"
    return match.group(0)


def test_model_env_installer_avoids_bash_namerefs() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")

    assert "local -n" not in source
    assert "WORLDFOUNDRY_PIP_INDEX_ARGS" in source


@pytest.mark.parametrize("has_cuda_headers", [False, True])
def test_unified_env_exposes_conda_cuda_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_cuda_headers: bool
) -> None:
    prefix = tmp_path / "selected env"
    target = subprocess.check_output(["uname", "-m"], text=True).strip()
    include = prefix / "targets" / f"{target}-linux" / "include"
    if has_cuda_headers:
        include.mkdir(parents=True)
        (include / "cuda_runtime.h").touch()
    existing = tmp_path / "existing include"
    exports = tmp_path / "exports.sh"
    fake_conda = tmp_path / "conda"
    fake_conda.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
    fake_conda.chmod(0o755)
    monkeypatch.setenv("CONDA_EXE", str(fake_conda))
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("WORLDFOUNDRY_DETECTED_DRIVER_CUDA", "12.8")
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_DATASET_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    monkeypatch.setenv("CPATH", str(existing))
    result = subprocess.run(
        [
            "bash",
            "scripts/setup/unified_install.sh",
            "--verify-only",
            "--cuda",
            "cu128",
            "--home",
            str(tmp_path),
            "--prefix",
            str(prefix),
            "--env-file",
            str(exports),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 17, result.stderr
    observed = subprocess.check_output(
        ["bash", "-c", 'source "$1"; printf "%s" "$CPATH"', "bash", str(exports)], text=True, timeout=10
    )
    assert observed == (f"{include}:{existing}" if has_cuda_headers else str(existing))


def test_solarwm_public_wan_ids_reach_flash_attn_post_install() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join(
        (
            "set -eu",
            _bash_function(source, "canonical_model_id"),
            _bash_function(source, "post_install_model_env"),
            "install_solarwm_flash_attn() { printf '%s\\n' solarwm-flash-attn; }",
            'post_install_model_env "$(canonical_model_id "$1")" /tmp/solarwm-test-env',
        )
    )

    for model_id in (
        "solarwm",
        "solarwm-wan22",
        "solarwm-wan22-5b",
        "solarwm-wan2.2-5b",
        "solarwm-wan2.2-14b",
    ):
        result = subprocess.run(
            ["bash", "-c", script, "bash", model_id],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (model_id, result.stderr)
        assert result.stdout.splitlines()[-1] == "solarwm-flash-attn", (model_id, result.stdout)


def test_solarwm_ltx_h3_ids_keep_explicit_interpreter_routes() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join((_bash_function(source, "canonical_model_id"), 'canonical_model_id "$1"'))

    for model_id in ("solarwm-ltx-22b", "solarwm-h3-33b"):
        result = subprocess.run(
            ["bash", "-c", script, "bash", model_id],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == model_id


def test_public_variant_ids_resolve_to_runtime_conda_env() -> None:
    expected_targets = {
        "dreamzero-droid": "dreamzero",
        "gamma-world-causal-few-step": "gamma-world",
        "hy-embodied-vla-umi": "hy-embodied-vla",
        "kandinsky6-lite": "kandinsky6",
        "kandinsky6-lite-distill": "kandinsky6",
        "kandinsky6-lite-pretrain": "kandinsky6",
        "kandinsky6-pro": "kandinsky6",
        "kandinsky6-pro-distill": "kandinsky6",
        "kandinsky6-pro-pretrain": "kandinsky6",
        "lapa-7b-openx": "lapa",
        "lingbot-va-base": "lingbot-va",
        "lingbot-vla-4b": "lingbot-vla",
        "lingbot-vla-v2-6b": "lingbot-vla-v2",
        "lingbot-world-act": "lingbot-world",
        "local-checkpoint": "mira",
        "matrix-game-3.5-first-person": "matrix-game-3.5",
        "spatial-forcing-libero-spatial": "spatial-forcing",
        "x-wam-robocasa-sft": "x-wam",
    }
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join(
        (
            _bash_function(source, "canonical_model_id"),
            'for model_id in "$@"; do printf "%s " "$model_id"; canonical_model_id "$model_id"; done',
        )
    )
    result = subprocess.run(
        ["bash", "-c", script, "bash", *expected_targets],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert dict(line.split(" ", 1) for line in result.stdout.splitlines()) == expected_targets

    from worldfoundry.cli.model_run import load_model_run_schema
    from worldfoundry.runtime.conda import load_runtime_conda_env_spec
    from worldfoundry.zoo.runtime.profiles import load_runtime_profile

    for public_id, canonical_id in expected_targets.items():
        schema = load_model_run_schema(public_id)
        profile = load_runtime_profile(schema.model_id, check_conda_env_exists=False)
        runtime_env = (profile.conda_env or {}).get("env_name") or (profile.conda_env or {}).get("resolved_env_name")
        install_spec = load_runtime_conda_env_spec(canonical_id)
        assert install_spec is not None, public_id
        assert install_spec.model_id == canonical_id
        assert install_spec.env_name == runtime_env, public_id


def test_unrelated_model_ids_keep_identity_canonicalization() -> None:
    model_ids = (
        "dreamzero",
        "gamma-world",
        "hy-embodied-vla",
        "kandinsky6",
        "lapa",
        "lingbot-va",
        "lingbot-vla",
        "lingbot-vla-v2",
        "lingbot-world",
        "matrix-game-3.5",
        "mira",
        "spatial-forcing",
        "wan2.1-t2v-1.3b",
        "x-wam",
    )
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join(
        (
            _bash_function(source, "canonical_model_id"),
            'for model_id in "$@"; do canonical_model_id "$model_id"; done',
        )
    )
    result = subprocess.run(
        ["bash", "-c", script, "bash", *model_ids],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(model_ids)


def test_setup_commands_follow_dependencies_and_precede_validation() -> None:
    script_source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    source = _bash_function(script_source, "install_dedicated_env")
    setup = source.index('install_env_setup_commands "$spec_json" "$env_prefix"')
    assert source.index('"${pip_args[@]}"') < setup
    assert source.index('"${req_pip_args[@]}"') < setup
    assert source.index('"${edit_pip_args[@]}"') < setup
    assert setup < source.index('verify_env_imports "$env_prefix"')
    unified = _bash_function(script_source, "install_model_env")
    assert unified.index('install_unified_env "$env_prefix"') < unified.index("install_env_setup_commands")
    assert 'if [[ "$VERIFY_ONLY" != "1" ]]; then\n      install_env_setup_commands' in unified


def test_setup_commands_execute_in_the_selected_conda_environment(tmp_path: Path) -> None:
    import json
    import shlex
    import sys

    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    commands = ["python -m pip install --no-build-isolation 'extension @ git+https://example.invalid/source'"]
    fake_conda = tmp_path / "conda"
    log = tmp_path / "commands.jsonl"
    fake_conda.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"with open({str(log)!r}, 'a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_conda.chmod(0o755)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"PYTHON_BIN={shlex.quote(sys.executable)}",
            f"CONDA_EXE_PATH={shlex.quote(str(fake_conda))}",
            _bash_function(source, "json_array_lines"),
            _bash_function(source, "install_env_setup_commands"),
            "install_env_setup_commands "
            + shlex.quote(json.dumps({"setup_commands": commands}))
            + " '/env with spaces'",
            "install_env_setup_commands '{}' '/unused-env'",
        ]
    )
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    assert [json.loads(line) for line in log.read_text().splitlines()] == [
        ["run", "-p", "/env with spaces", "bash", "-c", commands[0]]
    ]
