"""Validation and resolution for embodied Docker image references."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

_DIGEST_RE = re.compile(r"sha256:[0-9a-fA-F]{64}\Z")

AUTH_GATED_FLOATING_OFFICIAL_PROFILES = frozenset(
    {"behavior1k", "libero-plus", "molmospaces", "robomme"}
)
CROSS_REPO_FLOATING_MIRROR_PROFILES = frozenset({"libero"})
KNOWN_FLOATING_OFFICIAL_PROFILES = (
    AUTH_GATED_FLOATING_OFFICIAL_PROFILES | CROSS_REPO_FLOATING_MIRROR_PROFILES
)


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def require_pinned_images(*, override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    return _env_truthy("WORLDFOUNDRY_EMBODIED_REQUIRE_PINNED_IMAGES")


def _validate_embedded_digest(image: str) -> None:
    """Reject malformed digest syntax even when pin enforcement is disabled."""

    if "@" not in image:
        return
    if image.count("@") != 1:
        raise ValueError(f"invalid docker image digest reference: {image!r}")
    repository, digest = image.split("@", 1)
    if not repository or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError(
            f"invalid docker image digest reference {image!r}; expected name@sha256:<64 hex>"
        )


def image_ref_is_floating(image: str) -> bool:
    ref = str(image or "").strip()
    if not ref:
        return True
    _validate_embedded_digest(ref)
    if "@" in ref:
        return False
    leaf = ref.rsplit("/", 1)[-1]
    return ":" not in leaf or leaf.rsplit(":", 1)[-1] == "latest"


def assert_image_ref_pinned(
    image: str,
    *,
    require: bool | None = None,
    what: str = "image",
) -> None:
    ref = str(image or "").strip()
    _validate_embedded_digest(ref)
    if require_pinned_images(override=require) and image_ref_is_floating(ref):
        raise ValueError(
            f"refusing floating docker.{what}={image!r}; use an immutable tag or "
            "name@sha256:<64 hex>"
        )


def normalize_digest(digest: str) -> str:
    value = str(digest or "").strip()
    if not value:
        raise ValueError("digest must be non-empty")
    if not value.startswith("sha256:"):
        value = f"sha256:{value}"
    if _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"invalid docker digest {digest!r}; expected sha256:<64 hex>")
    return value.lower()


def repository_name(image: str) -> str:
    ref = str(image or "").strip()
    if not ref:
        raise ValueError("image must be non-empty")
    _validate_embedded_digest(ref)
    ref = ref.split("@", 1)[0]
    parent, slash, leaf = ref.rpartition("/")
    if ":" in leaf:
        leaf = leaf.rsplit(":", 1)[0]
    return f"{parent}{slash}{leaf}"


def apply_digest(image: str, digest: str) -> str:
    return f"{repository_name(image)}@{normalize_digest(digest)}"


def resolve_docker_image(
    docker_cfg: Mapping[str, Any],
    *,
    require_pinned: bool | None = None,
) -> str:
    image = str(docker_cfg.get("image") or "").strip()
    if not image:
        raise ValueError("docker.image is required")
    _validate_embedded_digest(image)
    digest = docker_cfg.get("digest")
    image_digest = docker_cfg.get("image_digest")
    normalized_digest = normalize_digest(str(digest)) if digest else None
    normalized_image_digest = normalize_digest(str(image_digest)) if image_digest else None
    if normalized_digest and normalized_image_digest and normalized_digest != normalized_image_digest:
        raise ValueError("docker.digest and docker.image_digest must match when both are configured")
    configured_digest = normalized_digest or normalized_image_digest
    if configured_digest and "@" in image:
        embedded_digest = normalize_digest(image.split("@", 1)[1])
        if embedded_digest != configured_digest:
            raise ValueError("docker.image embedded digest does not match docker.digest/image_digest")
    resolved = apply_digest(image, configured_digest) if configured_digest else image
    assert_image_ref_pinned(resolved, require=require_pinned, what="image")
    source = str(docker_cfg.get("source_image") or "").strip()
    if source:
        assert_image_ref_pinned(source, require=require_pinned, what="source_image")
    return resolved


def resolve_docker_platform(docker_cfg: Mapping[str, Any]) -> str | None:
    platform = str(docker_cfg.get("platform") or "").strip()
    if not platform:
        return None
    if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+(?:/[a-z0-9_.-]+)?", platform):
        raise ValueError(f"invalid docker platform: {platform!r}")
    return platform


__all__ = [
    "AUTH_GATED_FLOATING_OFFICIAL_PROFILES",
    "CROSS_REPO_FLOATING_MIRROR_PROFILES",
    "KNOWN_FLOATING_OFFICIAL_PROFILES",
    "apply_digest",
    "assert_image_ref_pinned",
    "image_ref_is_floating",
    "normalize_digest",
    "repository_name",
    "require_pinned_images",
    "resolve_docker_image",
    "resolve_docker_platform",
]
