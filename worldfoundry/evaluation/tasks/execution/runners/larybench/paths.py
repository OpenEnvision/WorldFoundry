"""
Path resolution utilities for LARY.

Provides functions to resolve relative paths based on dataset and split.

    All data file paths in metadata CSVs are stored as relative paths.
They are resolved at runtime by prepending the appropriate data root:

  Data root  : DATA_DIR  (LARYBench root, e.g. /path/to/LARYBench)
               └── classification/{dataset}/...
               └── regression/{dataset}/...
  LA root    : LARY_LA_DIR

Required environment variable:
  DATA_DIR    — root of the LARYBench dataset tree (set in env.sh)
  LARY_LA_DIR — root of extracted latent-action .npz files (set in env.sh)

"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# LA subdirectory layout per dataset.
#
# _LA_STRIDE_DIR : datasets that have a fixed stride sub-layer between
#                  {dataset}/ and {split}/{model}/
#                  Layout: {LA_ROOT}/{dataset}/{stride_dir}/{split}/{model}/
#                  Datasets not listed use: {LA_ROOT}/{dataset}/{split}/{model}/
#
# _LA_NO_SPLIT   : datasets whose LA files are NOT split by train/val.
#                  Layout: {LA_ROOT}/{dataset}/{model}/
# ---------------------------------------------------------------------------
_LA_STRIDE_DIR = {
    "calvin": "stride_5",
    "vlabench": "stride_5",
    "vlabench_15": "stride_15",
    "vlabench_30": "stride_30",
    "agibotbeta": "stride_45",
    "robocoin": "stride_10",
}

_LA_NO_SPLIT = {"human_1st", "robot_1st", "libero"}


# ---------------------------------------------------------------------------
# Dataset data-root sub-path within DATA_DIR (LARYBench root).
#
# Maps dataset name → sub-path under DATA_DIR, with optional split override.
# - Video/classification datasets live under: DATA_DIR/classification/{sub}/
# - Image-pair/regression datasets live under: DATA_DIR/regression/{sub}/{split_dir}/
#
# For regression datasets with per-split directories (e.g. calvin), the
# split directory is handled in get_data_root() automatically.
# ---------------------------------------------------------------------------
_DATASET_SUBPATH = {
    # classification (video)
    "human_1st": ("classification", None),  # DATA_DIR/classification/  (mixed sub-dirs)
    "robot_1st": ("classification", None),  # DATA_DIR/classification/
    "libero": ("classification", "LIBERO"),
    "egodex": ("classification", "EgoDex"),
    "epic-kitchens": ("classification", "EPIC-KITCHENS"),
    "ego4d": ("classification", "Ego4D"),
    "holoassist": ("classification", "HoloAssist"),
    "ssv2": ("classification", "SSv2"),
    "taco": ("classification", "TACO"),
    "agibotworld-beta": ("classification", "AgiBotWorld-Beta"),
    # regression (image-pair)
    "calvin": ("regression", "calvin"),
    "vlabench": ("regression", "vlabench"),
    "vlabench_15": ("regression", "vlabench"),
    "vlabench_30": ("regression", "vlabench"),
    "agibotbeta": ("regression", "agibot_45"),
    "robocoin": ("regression", "robocoin_10"),
}

# Regression datasets where train/val live in split-named sub-dirs
# (e.g. calvin: train_stride5, val_stride5)
_REGRESSION_SPLIT_DIR = {
    ("calvin", "train"): "train_stride5",
    ("calvin", "val"): "val_stride5",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_la_root() -> Optional[str]:
    """Return the latent-action storage root from env var LARY_LA_DIR."""
    return os.environ.get("LARY_LA_DIR")


def get_data_root(dataset: str, split: str = "train") -> Optional[str]:
    """
    Return the data root directory for a given dataset/split.

    Resolution order:
      1. DATA_DIR + dataset sub-path  (primary, LARYBench layout)
      2. {DATASET}_DATA_DIR           (legacy explicit override)
    """
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        dataset_lower = dataset.lower()
        entry = _DATASET_SUBPATH.get(dataset_lower)
        if entry:
            category, sub = entry
            if sub:
                base = os.path.join(data_dir, category, sub)
            else:
                base = os.path.join(data_dir, category)

            # For regression datasets with per-split sub-dirs
            split_subdir = _REGRESSION_SPLIT_DIR.get((dataset_lower, split.lower()))
            if split_subdir:
                return os.path.join(base, split_subdir)
            return base

        # Generic fallback: DATA_DIR itself
        return data_dir

    # Legacy: explicit {DATASET}_DATA_DIR override
    path = os.environ.get(f"{dataset.upper().replace('-', '_')}_DATA_DIR")
    if path:
        return path

    return None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _la_prefix(dataset: str, split: str, model: str) -> Optional[str]:
    """
    Build the latent-action directory prefix (pure string ops, no stat calls).

    Layout:
      no-split datasets : {LA_ROOT}/{dataset}/{model}/
      with stride layer : {LA_ROOT}/{dataset}/{stride_dir}/{split}/{model}/
      standard          : {LA_ROOT}/{dataset}/{split}/{model}/
    """
    la_root = get_la_root()
    if not la_root:
        return None

    dataset_lower = dataset.lower()

    if dataset_lower in _LA_NO_SPLIT:
        return os.path.join(la_root, dataset_lower, model)

    stride_dir = _LA_STRIDE_DIR.get(dataset_lower)
    if stride_dir:
        return os.path.join(la_root, dataset_lower, stride_dir, split, model)
    return os.path.join(la_root, dataset_lower, split, model)


def resolve_la_path(la_path, dataset=None, split=None, model="dinov2"):
    """
    Resolve a latent-action path that may be relative.

    Args:
        la_path : Relative filename (e.g. latent_action_00000001.npz)
                  or an already-absolute path (returned as-is).
        dataset : Dataset name.
        split   : Data split (train / val / all / seen_train / ...).
        model   : Model name sub-directory (default dinov2).

    Returns:
        Absolute path string.
    """
    if not la_path or os.path.isabs(la_path):
        return la_path

    normalized = str(la_path).replace("\\", "/")
    parts = normalized.split("/")
    filename = parts[-1]

    # Newer extraction CSVs may already store a portable path such as
    # calvin/train/dinov2/latent_action_00000000.npz.  In that case, prepend
    # only the LA root; do not prepend the dataset/stride/split/model prefix
    # again.
    if len(parts) > 1:
        la_root = get_la_root()
        if la_root:
            return os.path.join(la_root, *parts)
        return os.path.join(*parts)

    if dataset and split:
        prefix = _la_prefix(dataset, split, model)
        if prefix:
            return os.path.join(prefix, filename)

    la_root = get_la_root()
    if la_root:
        return os.path.join(la_root, filename)
    return filename


def resolve_data_path(path, dataset=None, split=None):
    """
    Resolve a data file path that may be relative.

    Args:
        path    : Relative or absolute path.
        dataset : Dataset name (used to locate data root).
        split   : Data split.

    Returns:
        Absolute path string.
    """
    if not path or os.path.isabs(path):
        return path

    data_root = get_data_root(dataset, split) if dataset else None
    if data_root:
        return os.path.join(data_root, path)
    return path
