from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/embodied/resolve_official_image_digests.py"
_MIRROR_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/embodied/mirror_docker_images.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("resolve_official_image_digests", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_mirror_script():
    spec = importlib.util.spec_from_file_location("mirror_docker_images_digest_test", _MIRROR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:short",
        "sha256:" + ("z" * 64),
        "sha256:" + ("0" * 65),
        "garbage",
    ],
)
def test_resolver_rejects_malformed_registry_digest(digest: str) -> None:
    module = _load_script()
    resolved, errors = module.resolve_images(
        {"ghcr.io/example/bench:latest": ["bench"]},
        fetch_digest=lambda _image: digest,
    )
    assert not resolved
    assert "sha256" in errors["ghcr.io/example/bench:latest"]


def test_resolver_normalizes_valid_digest() -> None:
    module = _load_script()
    resolved, errors = module.resolve_images(
        {"ghcr.io/example/bench:latest": ["bench"]},
        fetch_digest=lambda _image: "A" * 64,
    )
    assert not errors
    assert resolved["ghcr.io/example/bench:latest"] == {
        "digest": "sha256:" + ("a" * 64),
        "platform": "linux/amd64",
        "profiles": ["bench"],
    }


def test_repository_parser_rejects_garbage_digest_suffix() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="digest"):
        module.repository_and_tag("ghcr.io/example/bench@garbage")


def test_collect_floating_images_rejects_malformed_pinned_ref(tmp_path: Path) -> None:
    module = _load_script()
    (tmp_path / "bad.yaml").write_text(
        "id: bad\ndocker:\n  image: ghcr.io/example/bench@garbage\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest"):
        module.collect_floating_images(tmp_path)


def test_apply_pins_records_platform_and_pins_source(tmp_path: Path) -> None:
    module = _load_script()
    image = "ghcr.io/example/bench:latest"
    digest = "sha256:" + ("b" * 64)
    profile = tmp_path / "bench.yaml"
    profile.write_text(
        f"id: bench\ndocker:\n  image: {image}\n  source_image: {image}\n",
        encoding="utf-8",
    )

    assert module.apply_pins_to_profile(profile, image_digests={image: digest})
    text = profile.read_text(encoding="utf-8")
    assert f"digest: {digest}" in text
    assert f"source_image: ghcr.io/example/bench@{digest}" in text
    assert "platform: linux/amd64" in text


def test_write_digest_map_adds_default_platform(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "digests.json"
    module.write_digest_map(
        output,
        {"ghcr.io/example/bench:latest": {"digest": "c" * 64, "profiles": ["bench"]}},
    )
    entry = json.loads(output.read_text(encoding="utf-8"))["images"]["ghcr.io/example/bench:latest"]
    assert entry["digest"] == "sha256:" + ("c" * 64)
    assert entry["platform"] == "linux/amd64"


def test_mirror_target_prefix_uses_latest_for_digest_only_source(tmp_path: Path) -> None:
    """A digest-only source has no tag; mirror destinations retain the prior latest default."""

    module = _load_mirror_script()
    digest = "sha256:" + ("d" * 64)
    (tmp_path / "bench.yaml").write_text(
        "id: bench\n"
        "docker:\n"
        f"  image: ghcr.io/example/bench@{digest}\n"
        f"  source_image: ghcr.io/example/bench@{digest}\n"
        "  platform: linux/amd64\n",
        encoding="utf-8",
    )

    mappings = module.load_image_mappings(
        tmp_path,
        ["all"],
        target_prefix="registry.example/worldfoundry",
    )
    assert len(mappings) == 1
    assert mappings[0].source_image == f"ghcr.io/example/bench@{digest}"
    assert mappings[0].target_image == "registry.example/worldfoundry/bench:latest"
    assert mappings[0].platform == "linux/amd64"


def test_mirror_identity_pin_only_pulls_for_requested_platform() -> None:
    module = _load_mirror_script()
    source = "ghcr.io/example/bench@sha256:" + ("e" * 64)
    commands: list[list[str]] = []
    module._run = lambda command, *, plan_only: commands.append(command)

    module.mirror_images(
        [module.ImageMapping("bench", source, source, "linux/amd64")],
        push=True,
        plan_only=True,
    )
    assert commands == [["docker", "pull", "--platform", "linux/amd64", source]]
