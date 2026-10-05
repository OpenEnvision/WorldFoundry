"""Strict-local checkpoint resolution for the MMAudio runtime."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path


DFN5B_MODEL_ID = "apple/DFN5B-CLIP-ViT-H-14-384"
DFN5B_REQUIRED_FILES = (
    "open_clip_config.json",
    "open_clip_pytorch_model.bin",
)
BIGVGAN_MODEL_ID = "nvidia/bigvgan_v2_44khz_128band_512x"
BIGVGAN_REQUIRED_FILES = (
    "config.json",
    "bigvgan_generator.pt",
)


def _checkpoint_roots() -> tuple[Path, ...]:
    primary = checkpoint_root_path()
    roots = [primary, primary / "hfd", hfd_root_path()]
    alternate_name = {"ckpt": "ckpts", "ckpts": "ckpt"}.get(primary.name)
    if alternate_name:
        alternate = primary.with_name(alternate_name)
        roots.extend((alternate, alternate / "hfd"))
    return tuple(dict.fromkeys(roots))


def _resolve_checkpoint(
    model_id: str,
    required_files: tuple[str, ...],
    description: str,
) -> Path:
    names = (
        model_id.replace("/", "--"),
        model_id,
    )
    checked = []
    for root in _checkpoint_roots():
        for name in names:
            candidate = root / name
            checked.append(candidate)
            if all((candidate / filename).is_file() for filename in required_files):
                return candidate.resolve()

    rendered = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"MMAudio requires the official {description}; checked:\n  - {rendered}"
    )


def resolve_dfn5b_checkpoint() -> Path:
    """Resolve Apple's MMAudio OpenCLIP model without Hub access."""

    return _resolve_checkpoint(
        DFN5B_MODEL_ID,
        DFN5B_REQUIRED_FILES,
        "apple/DFN5B-CLIP-ViT-H-14-384 OpenCLIP export",
    )


def dfn5b_open_clip_ref() -> str:
    """Return the OpenCLIP local-directory model reference."""

    return f"local-dir:{resolve_dfn5b_checkpoint()}"


def resolve_bigvgan_v2_checkpoint() -> Path:
    """Resolve NVIDIA's 44 kHz BigVGAN v2 model without Hub access."""

    return _resolve_checkpoint(
        BIGVGAN_MODEL_ID,
        BIGVGAN_REQUIRED_FILES,
        "nvidia/bigvgan_v2_44khz_128band_512x export",
    )
