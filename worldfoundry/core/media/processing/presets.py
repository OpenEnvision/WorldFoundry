# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Named streaming post-processing presets.

Responsibility: map a preset name to a :class:`VideoPostProcessor`. The
registry is an in-repo static table of ``"module:attr"`` factory targets;
a target module is imported only when its preset is built, so optional
runtimes stay unimported until selected.

Every factory is called as ``factory(preset, device=device, **options)``
with the normalized preset name, so one factory can serve several presets.
The empty name and ``none`` / ``off`` / ``identity`` select
:class:`IdentityVideoPostProcessor`.

Public surface: :func:`available_postprocess_presets`,
:func:`build_postprocessor`, :func:`normalize_postprocess_preset_name`.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from worldfoundry.core.media.processing.postprocess import (
    IdentityVideoPostProcessor,
    VideoPostProcessor,
)

_IDENTITY_PRESET_ALIASES = frozenset({"", "none", "off", "identity"})
_POSTPROCESS_PRESET_FACTORIES: dict[str, str] = {
    "rtx-super-resolution": "worldfoundry.core.media.processing.rtx:rtx_postprocessor_from_preset",
    "rtx-super-resolution-ultra": "worldfoundry.core.media.processing.rtx:rtx_postprocessor_from_preset",
    "rtx-deblur-ultra": "worldfoundry.core.media.processing.rtx:rtx_postprocessor_from_preset",
}


def normalize_postprocess_preset_name(name: str) -> str:
    """Strip and lowercase ``name`` and spell separators with ``-``."""

    return name.strip().lower().replace("_", "-")


def available_postprocess_presets() -> tuple[str, ...]:
    """Return the selectable preset names, ``identity`` first."""

    return ("identity", *_POSTPROCESS_PRESET_FACTORIES)


def build_postprocessor(
    name: str,
    *,
    device: int | None = None,
    options: Mapping[str, object] | None = None,
) -> VideoPostProcessor:
    """Build the processor selected by preset ``name``.

    ``device`` is the CUDA device index for processors that run on a GPU;
    ``options`` are preset-specific keyword overrides forwarded to the
    factory. The identity preset accepts no options.
    """

    normalized = normalize_postprocess_preset_name(name)
    if normalized in _IDENTITY_PRESET_ALIASES:
        if options:
            raise ValueError(f"Postprocess preset 'identity' accepts no options; got {sorted(options)}.")
        return IdentityVideoPostProcessor()
    try:
        target = _POSTPROCESS_PRESET_FACTORIES[normalized]
    except KeyError as exc:
        available = ", ".join(available_postprocess_presets())
        raise ValueError(f"Unknown postprocess preset {name!r}. Available presets: {available}.") from exc
    module_name, _, attribute = target.partition(":")
    factory = getattr(import_module(module_name), attribute)
    return factory(normalized, device=device, **dict(options or {}))


__all__ = [
    "available_postprocess_presets",
    "build_postprocessor",
    "normalize_postprocess_preset_name",
]
