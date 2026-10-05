"""Lazy exports for Matrix-Game inference datasets and index metadata."""

from importlib import import_module

_EXPORTS = {
    "DA3MosaicVideoDataset": (".unified_dataset", "DA3MosaicVideoDataset"),
    "SubjectRefMemoryDA3MosaicVideoDataset": (
        ".subject_ref_memory_dataset",
        "SubjectRefMemoryDA3MosaicVideoDataset",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
