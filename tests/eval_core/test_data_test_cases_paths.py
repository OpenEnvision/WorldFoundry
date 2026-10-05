import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_TEST_CASE_COUNT_BASELINE = 500
TRACKED_TEST_CASE_BYTES_BASELINE = 352_822_015


def test_tracked_test_cases_do_not_grow_before_externalization() -> None:
    """Keep the existing fixture corpus stable until it moves to external storage."""

    result = subprocess.run(
        [
            "git",
            "ls-tree",
            "-rl",
            "HEAD",
            "--",
            "worldfoundry/data/test_cases",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("tracked test-case growth gate requires a Git checkout")

    rows = [line for line in result.stdout.splitlines() if line.strip()]
    tracked_bytes = sum(int(line.split(maxsplit=4)[3]) for line in rows)

    assert len(rows) <= TRACKED_TEST_CASE_COUNT_BASELINE
    assert tracked_bytes <= TRACKED_TEST_CASE_BYTES_BASELINE


def test_examples_reference_packaged_test_cases_root() -> None:
    legacy_root = "data" + "/test_cases"
    disallowed = (
        f"./{legacy_root}",
        f'"{legacy_root}/',
        f"'{legacy_root}/",
        f' / "{legacy_root}/',
        f" / '{legacy_root}/",
    )
    checked_roots = (
        REPO_ROOT / "test",
        REPO_ROOT / "worldfoundry/pipelines",
        REPO_ROOT / "worldfoundry/studio",
        REPO_ROOT / "worldfoundry/synthesis",
        REPO_ROOT / "scripts",
        REPO_ROOT / "docs",
    )
    checked_files = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/scope/scope_runtime/inference.py",
    )
    offenders: list[str] = []
    for root in checked_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path == Path(__file__).resolve():
                continue
            if path.is_dir() or not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".md", ".mdx", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in text for marker in disallowed):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    for path in checked_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in disallowed):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_test_cases_root_does_not_store_runtime_yaml_configs() -> None:
    test_cases_root = REPO_ROOT / "worldfoundry/data/test_cases"
    root_yaml_configs = sorted(
        path.name for path in test_cases_root.glob("*") if path.suffix in {".yaml", ".yml"}
    )

    assert root_yaml_configs == []


def test_wan_va_inference_configs_are_user_editable_yaml() -> None:
    runtime_config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/wan_va"
    test_case_root = REPO_ROOT / "worldfoundry/data/test_cases/wan_va"
    demo_yaml = runtime_config_root / "demo.yaml"
    i2va_yaml = runtime_config_root / "demo_i2va.yaml"
    inference_config_py = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/wan_va/wan_va/configs/va_inference_cfg.py"
    )
    inference_i2va_config_py = (
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/wan_va/wan_va/configs/va_inference_i2va.py"
    )
    retired_demo_modules = [
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/wan_va/wan_va/configs/va_demo_cfg.py",
        REPO_ROOT
        / "worldfoundry/synthesis/action_generation/wan_va/wan_va/configs/va_demo_i2va.py",
    ]

    demo_payload = yaml.safe_load(demo_yaml.read_text(encoding="utf-8"))
    i2va_payload = yaml.safe_load(i2va_yaml.read_text(encoding="utf-8"))
    inference_text = inference_config_py.read_text(encoding="utf-8")
    inference_i2va_text = inference_i2va_config_py.read_text(encoding="utf-8")

    assert not test_case_root.exists() or not any(test_case_root.glob("*.yaml"))
    assert [path for path in retired_demo_modules if path.exists()] == []
    assert demo_payload["wan22_pretrained_model_name_or_path"] is None
    assert demo_payload["infer_mode"] == "server"
    assert demo_payload["used_action_channel_ids"] == [0, 1, 2, 3, 4, 28]
    assert i2va_payload["infer_mode"] == "i2va"
    assert "demo.yaml" in inference_text
    assert "demo_i2va.yaml" in inference_i2va_text
    assert "Pick the green cube" not in inference_i2va_text

    from worldfoundry.synthesis.action_generation.wan_va.wan_va.configs.va_inference_cfg import (
        va_inference_cfg,
    )
    from worldfoundry.synthesis.action_generation.wan_va.wan_va.configs.va_inference_i2va import (
        va_inference_i2va_cfg,
    )

    assert va_inference_cfg.infer_mode == "server"
    assert va_inference_cfg.inverse_used_action_channel_ids[28] == 5
    assert va_inference_i2va_cfg.prompt == i2va_payload["prompt"]


def test_wan_va_runtime_configs_are_yaml_backed_and_training_code_is_removed() -> None:
    runtime_config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/wan_va"
    test_case_root = REPO_ROOT / "worldfoundry/data/test_cases/wan_va"
    config_root = REPO_ROOT / "worldfoundry/synthesis/action_generation/wan_va/wan_va/configs"
    removed_paths = [
        REPO_ROOT / "worldfoundry/synthesis/action_generation/wan_va/wan_va/train.py",
        REPO_ROOT / "worldfoundry/synthesis/action_generation/wan_va/wan_va/dataset",
        config_root / "va_demo_train_cfg.py",
        config_root / "va_libero_train_cfg.py",
        config_root / "va_robotwin_train_cfg.py",
    ]
    expected_yaml = {
        "shared.yaml",
        "demo.yaml",
        "demo_i2va.yaml",
        "franka.yaml",
        "franka_i2va.yaml",
        "libero.yaml",
        "libero_i2va.yaml",
        "robotwin.yaml",
        "robotwin_i2va.yaml",
    }

    assert [path for path in removed_paths if path.exists()] == []
    assert expected_yaml <= {path.name for path in runtime_config_root.glob("*.yaml")}
    assert not test_case_root.exists() or not any(test_case_root.glob("*.yaml"))

    payloads = {
        name: yaml.safe_load((runtime_config_root / name).read_text(encoding="utf-8"))
        for name in expected_yaml
    }
    assert payloads["shared.yaml"]["param_dtype"] == "bfloat16"
    for name in ("demo.yaml", "franka.yaml", "libero.yaml", "robotwin.yaml"):
        assert payloads[name]["wan22_pretrained_model_name_or_path"] is None
    assert payloads["libero.yaml"]["action_per_frame"] == 4
    assert payloads["robotwin.yaml"]["action_per_frame"] == 16
    assert payloads["franka.yaml"]["used_action_channel_ids"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        28,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        29,
    ]
    assert payloads["robotwin_i2va.yaml"]["infer_mode"] == "i2va"

    for path in config_root.glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "_load_yaml_config" in text
        assert "runtime", "configs" in text or path.name != "shared_config.py"
        assert "WAN_VA_TEST_CASE_ROOT" not in text
        assert "from easydict import EasyDict" not in text
        assert "/path/to/pretrained/model" not in text
        assert "dataset_path" not in text
        assert "enable_wandb" not in text

    from worldfoundry.synthesis.action_generation.wan_va.wan_va.configs import VA_CONFIGS

    assert set(VA_CONFIGS) == {
        "demo",
        "demo_i2av",
        "franka",
        "franka_i2av",
        "libero",
        "libero_i2av",
        "robotwin",
        "robotwin_i2av",
    }
    assert VA_CONFIGS["libero"].action_per_frame == payloads["libero.yaml"]["action_per_frame"]
    assert VA_CONFIGS["robotwin"].infer_mode == "server"
    assert VA_CONFIGS["robotwin"].inverse_used_action_channel_ids[29] == 15


def test_wan_va_server_requires_explicit_checkpoint_dir() -> None:
    server_path = (
        REPO_ROOT / "worldfoundry/synthesis/action_generation/wan_va/wan_va/wan_va_server.py"
    )
    text = server_path.read_text(encoding="utf-8")

    assert 'WAN_VA_CHECKPOINT_ENV = "WORLDFOUNDRY_WAN_VA_CHECKPOINT_DIR"' in text
    assert "def _resolve_checkpoint_dir(args, config):" in text
    assert "args.checkpoint_dir or os.getenv(WAN_VA_CHECKPOINT_ENV)" in text
    assert "Pass --checkpoint-dir" in text
    assert "os.path.join(None" not in text


def test_lingbot_world_runtime_config_is_yaml_backed() -> None:
    runtime_config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs/lingbot_world"
    test_case_root = REPO_ROOT / "worldfoundry/data/test_cases/lingbot_world"
    config_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/lingbot/lingbot_world_runtime/configs"
    expected_yaml = {
        "shared.yaml",
        "wan_i2v_A14B.yaml",
        "registry.yaml",
    }

    assert expected_yaml <= {path.name for path in runtime_config_root.glob("*.yaml")}
    assert not test_case_root.exists() or not any(test_case_root.glob("*.yaml"))
    payloads = {
        name: yaml.safe_load((runtime_config_root / name).read_text(encoding="utf-8"))
        for name in expected_yaml
    }
    assert payloads["shared.yaml"]["t5_dtype"] == "bfloat16"
    assert payloads["wan_i2v_A14B.yaml"]["fast_noise_checkpoint"] == "lingbot_world_fast"
    assert payloads["wan_i2v_A14B.yaml"]["sample_steps"] == 70
    assert payloads["registry.yaml"]["supported_sizes"]["i2v-A14B"] == [
        "720*1280",
        "1280*720",
        "480*832",
        "832*480",
    ]

    for path in config_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "_load_yaml_config" in text
        assert "runtime", "configs" in text or path.name != "shared_config.py"
        assert "LINGBOT_WORLD_TEST_CASE_ROOT" not in text
        assert "from easydict import EasyDict" not in text
        assert "lingbot_world_fast" not in text
        assert "sample_steps = 70" not in text

    from worldfoundry.synthesis.visual_generation.lingbot.lingbot_world_runtime.configs import (
        SIZE_CONFIGS,
        SUPPORTED_SIZES,
        WAN_CONFIGS,
    )

    cfg = WAN_CONFIGS["i2v-A14B"]
    assert cfg.fast_noise_checkpoint == "lingbot_world_fast"
    assert cfg.sample_steps == payloads["wan_i2v_A14B.yaml"]["sample_steps"]
    assert SIZE_CONFIGS["480*832"] == (480, 832)
    assert SUPPORTED_SIZES["i2v-A14B"] == tuple(payloads["registry.yaml"]["supported_sizes"]["i2v-A14B"])
