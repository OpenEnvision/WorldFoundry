from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.tasks.embodied import docker_runner
from worldfoundry.evaluation.tasks.embodied.docker_runner import build_docker_run_command
from worldfoundry.evaluation.tasks.embodied.image_refs import (
    AUTH_GATED_FLOATING_OFFICIAL_PROFILES,
    CROSS_REPO_FLOATING_MIRROR_PROFILES,
    KNOWN_FLOATING_OFFICIAL_PROFILES,
    image_ref_is_floating,
    normalize_digest,
    resolve_docker_image,
    resolve_docker_platform,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_ROOT = _REPO_ROOT / "worldfoundry/data/benchmarks/runtime_profiles/official"
_DIGEST_MAP = _PROFILE_ROOT / "docker_image_digests.json"


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:abc",
        "sha256:" + ("g" * 64),
        "sha256:" + ("a" * 63),
        "sha256:" + ("a" * 65),
        "sha512:" + ("a" * 64),
    ],
)
def test_normalize_digest_rejects_malformed_sha256(digest: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        normalize_digest(digest)


@pytest.mark.parametrize(
    "image",
    [
        "example/bench@garbage",
        "example/bench@sha256:abc",
        "example/bench@@sha256:" + ("a" * 64),
    ],
)
def test_malformed_embedded_digest_is_never_treated_as_floating(image: str) -> None:
    with pytest.raises(ValueError, match="digest"):
        image_ref_is_floating(image)
    with pytest.raises(ValueError, match="digest"):
        resolve_docker_image({"image": image}, require_pinned=False)


def test_resolver_rejects_conflicting_embedded_and_configured_digests() -> None:
    embedded = "sha256:" + ("a" * 64)
    configured = "sha256:" + ("b" * 64)
    with pytest.raises(ValueError, match="does not match"):
        resolve_docker_image(
            {"image": f"example/bench@{embedded}", "digest": configured},
            require_pinned=False,
        )


def test_resolver_rejects_conflicting_digest_alias_fields() -> None:
    with pytest.raises(ValueError, match="must match"):
        resolve_docker_image(
            {
                "image": "example/bench:latest",
                "digest": "sha256:" + ("a" * 64),
                "image_digest": "sha256:" + ("b" * 64),
            },
            require_pinned=False,
        )


def test_matching_embedded_and_configured_digest_is_idempotent() -> None:
    digest = "sha256:" + ("A" * 64)
    assert resolve_docker_image(
        {"image": f"example/bench@{digest}", "image_digest": digest.lower()},
        require_pinned=True,
    ) == f"example/bench@{digest.lower()}"


def test_docker_command_uses_resolved_digest_and_platform(tmp_path: Path) -> None:
    digest = "sha256:" + ("a" * 64)
    command = build_docker_run_command(
        {
            "docker": {
                "image": "example/bench:latest",
                "digest": digest,
                "platform": "linux/amd64",
            }
        },
        docker_config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "output",
    )

    platform_index = command.index("--platform")
    assert command[platform_index + 1] == "linux/amd64"
    assert f"example/bench@{digest}" in command


def test_docker_pull_passes_configured_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _Result:
        returncode = 1

    monkeypatch.setattr(docker_runner.subprocess, "run", lambda *_args, **_kwargs: _Result())
    monkeypatch.setattr(docker_runner.subprocess, "call", lambda command: calls.append(command) or 0)

    docker_runner._ensure_image(
        "docker",
        "example/bench@sha256:" + ("a" * 64),
        platform="linux/amd64",
        pull=True,
    )
    assert calls == [
        [
            "docker",
            "pull",
            "--platform",
            "linux/amd64",
            "example/bench@sha256:" + ("a" * 64),
        ]
    ]


def test_invalid_docker_platform_is_rejected() -> None:
    with pytest.raises(ValueError, match="platform"):
        resolve_docker_platform({"platform": "linux/amd64 --privileged"})


def test_verified_digest_map_matches_all_pinned_profiles() -> None:
    digest_map = json.loads(_DIGEST_MAP.read_text(encoding="utf-8"))["images"]
    assert len(digest_map) == 13
    for source_tag, metadata in digest_map.items():
        assert metadata["platform"] == "linux/amd64"
        digest = normalize_digest(metadata["digest"])
        for profile_id in metadata["profiles"]:
            docker = yaml.safe_load((_PROFILE_ROOT / f"{profile_id}.yaml").read_text(encoding="utf-8"))["docker"]
            assert docker["platform"] == metadata["platform"]
            assert docker["source_image"] == f"{source_tag.rsplit(':', 1)[0]}@{digest}"


def test_libero_pins_only_source_and_keeps_cross_repo_target_floating() -> None:
    docker = yaml.safe_load((_PROFILE_ROOT / "libero.yaml").read_text(encoding="utf-8"))["docker"]
    assert docker["image"] == "ghcr.io/openenvision/worldfoundry-embodied-libero:latest"
    assert image_ref_is_floating(docker["image"])
    assert not image_ref_is_floating(docker["source_image"])
    assert "digest" not in docker
    assert docker["platform"] == "linux/amd64"


def test_only_explicit_allowlist_profiles_remain_floating() -> None:
    floating_profiles: set[str] = set()
    for path in sorted(_PROFILE_ROOT.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        docker = payload.get("docker") or {}
        image = str(docker.get("image") or "").strip()
        source = str(docker.get("source_image") or "").strip()
        if image and (image_ref_is_floating(image) or (source and image_ref_is_floating(source))):
            floating_profiles.add(path.stem)

    assert floating_profiles == set(KNOWN_FLOATING_OFFICIAL_PROFILES)
    assert AUTH_GATED_FLOATING_OFFICIAL_PROFILES.isdisjoint(CROSS_REPO_FLOATING_MIRROR_PROFILES)
    assert CROSS_REPO_FLOATING_MIRROR_PROFILES == {"libero"}
