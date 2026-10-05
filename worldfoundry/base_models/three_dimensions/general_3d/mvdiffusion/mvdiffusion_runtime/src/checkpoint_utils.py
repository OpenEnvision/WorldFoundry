"""Local checkpoint resolution for the vendored MVDiffusion runtime."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path


def resolve_mvdiffusion_checkpoint(*parts: str) -> Path:
    """Resolve an MVDiffusion file from the configured ``ckpt(s)`` roots."""

    primary = checkpoint_root_path()
    roots = [primary]
    alternate_name = {"ckpt": "ckpts", "ckpts": "ckpt"}.get(primary.name)
    if alternate_name:
        roots.append(primary.with_name(alternate_name))

    checked = []
    for root in dict.fromkeys(roots):
        candidate = root.joinpath(*parts)
        checked.append(candidate)
        if candidate.is_file():
            return candidate.resolve()

    rendered = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"MVDiffusion checkpoint is missing; checked:\n  - {rendered}"
    )
