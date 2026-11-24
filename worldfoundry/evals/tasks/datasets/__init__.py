"""Public dataset helpers for manifests and access checks."""

from __future__ import annotations

from .access import (
    DatasetAccessIssue,
    DatasetAccessReport,
    build_hf_download_command,
    classify_dataset_access,
    find_hf_downloader,
)
from .local_status import (
    DatasetLocalStatus,
    check_local_dataset,
    direct_dataset_dir_candidates,
    direct_dataset_expected_files,
    hf_cache_dataset_dir,
)
from .location import (
    DatasetLocation,
    dataset_location_env_var,
    locate_local_dataset,
)
from .manager import (
    DatasetDownloadPlan,
    DatasetManager,
)
from .manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    build_dataset_manifest,
    file_sha256,
    load_dataset_manifest,
    read_dataset_samples,
    resolve_dataset_samples_path,
    validate_dataset_manifest,
    write_dataset_manifest,
)
from .refs import (
    DatasetRef,
    filter_dataset_refs,
    is_commit_like_revision,
    normalize_hf_dataset_id,
    parse_dataset_refs,
)

__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DatasetAccessIssue",
    "DatasetAccessReport",
    "DatasetDownloadPlan",
    "DatasetLocalStatus",
    "DatasetLocation",
    "DatasetManager",
    "DatasetManifest",
    "DatasetRef",
    "build_dataset_manifest",
    "build_hf_download_command",
    "check_local_dataset",
    "classify_dataset_access",
    "dataset_location_env_var",
    "direct_dataset_dir_candidates",
    "direct_dataset_expected_files",
    "file_sha256",
    "filter_dataset_refs",
    "find_hf_downloader",
    "hf_cache_dataset_dir",
    "is_commit_like_revision",
    "load_dataset_manifest",
    "locate_local_dataset",
    "normalize_hf_dataset_id",
    "parse_dataset_refs",
    "read_dataset_samples",
    "resolve_dataset_samples_path",
    "validate_dataset_manifest",
    "write_dataset_manifest",
]
