"""Local dataset checks: Hugging Face cache snapshots, direct dataset directories, and completeness scans."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path

from ..catalog.schema import BenchmarkDatasetRef
from .refs import (
    DatasetRef,
    JsonValue,
    _first_dataset_ref,
    _optional_str,
    _plain,
    is_commit_like_revision,
    normalize_hf_dataset_id,
)

_DIRECT_DATASET_FILE_COUNT_LIMIT = 10_000


_DIRECT_DATASET_INCOMPLETE_LIMIT = 20


_DIRECT_DATASET_METADATA_FILE_LIMIT = 200


_DIRECT_DATASET_METADATA_DIR_LIMIT = 256


_IGNORED_DIRECT_DATASET_SIBLING_NAMES = frozenset({".DS_Store"})


_IGNORED_DIRECT_DATASET_SIBLING_PARTS = frozenset({"__MACOSX"})


def hf_cache_dataset_dir(cache_dir: str | Path, dataset_id: str) -> Path:
    """Resolve the default Hugging Face cache folder name for a given dataset ID.

    Args:
        cache_dir: Root of HF cache.
        dataset_id: Target dataset repository ID.

    Returns:
        Path of the cache directory.
    """
    normalized = normalize_hf_dataset_id(dataset_id)
    if not normalized:
        raise ValueError("dataset_id must be a non-empty Hugging Face dataset id")
    return Path(cache_dir) / f"datasets--{normalized.replace('/', '--')}"


def direct_dataset_dir_candidates(cache_dir: str | Path, dataset_id: str) -> tuple[Path, ...]:
    """Retrieve all direct folder candidates where a dataset might reside in direct layout.

    Args:
        cache_dir: Root cache directory.
        dataset_id: Target dataset ID.

    Returns:
        Tuple of candidate directory Paths.
    """
    normalized = normalize_hf_dataset_id(dataset_id)
    if not normalized:
        raise ValueError("dataset_id must be a non-empty Hugging Face dataset id")
    root = Path(cache_dir)
    org, name = normalized.split("/", 1)
    candidate_roots = [root]
    nested_hfd_root = root / "hfd_datasets"
    if root.name != "hfd_datasets":
        candidate_roots.append(nested_hfd_root)

    candidates: list[Path] = []
    for candidate_root in candidate_roots:
        candidates.extend(
            (
                candidate_root / normalized.replace("/", "--"),
                candidate_root / normalized.replace("/", "__"),
                candidate_root / org / name,
            )
        )
    return tuple(dict.fromkeys(candidates))


def _dataset_ref_path_candidate(path: str | None) -> Path | None:
    """Resolve an explicit relative or absolute path from dataset reference.

    Args:
        path: Path string from reference.

    Returns:
        Resolved Path object or None.
    """
    if not path:
        return None
    resolved = resolve_worldfoundry_path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved


def direct_dataset_dir_candidates_for_ref(cache_dir: str | Path, ref: DatasetRef) -> tuple[Path, ...]:
    """Return direct-layout candidates, preferring an explicit manifest path."""

    if not ref.hf_dataset_id:
        return ()
    candidates: list[Path] = []
    explicit_path = _dataset_ref_path_candidate(ref.path)
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(direct_dataset_dir_candidates(cache_dir, ref.hf_dataset_id))
    return tuple(dict.fromkeys(candidates))


@dataclass(frozen=True)
class DatasetLocalStatus:
    """Represents the real-time presence and integrity status of a dataset on the local system.

    Attributes:
        hf_dataset_id: Hugging Face dataset repository ID.
        cache_dataset_dir: Directory path of the Hugging Face cache.
        direct_dataset_dir: Directory path under a direct file layout (non-cache).
        local_layout: Detected layout string ('hf_cache', 'direct_hfd', 'missing').
        expected_revision: Expected Git revision SHA/tag.
        ref_name: Checked reference label (e.g. 'main').
        referenced_snapshot: Active commit SHA resolved for the ref_name.
        revision_matches: Whether the local revision matches expectation.
        snapshot_dirs: Tuple of snapshot Paths checked under HF cache.
        file_count: Number of completed files discovered.
        expected_file_count: Total expected file count.
        missing_file_count: Count of missing files compared to expectations.
        direct_file_count: File count under direct layout directory.
        direct_file_count_capped: Whether the direct file count scan was capped.
        direct_revision: Direct layout revision string.
        direct_revision_matches: Whether direct layout revision matches expectation.
        direct_ready: Whether direct layout files are complete.
        incomplete_files: Tuple of incomplete/partial files under HF cache.
        direct_incomplete_files: Tuple of incomplete files under direct layout.
        broken_links: Tuple of broken symlinks paths.
        ready: Ultimate readiness boolean indicator.
        status: Detailed status label ('ready', 'not_found', etc.).
        reason: Optional descriptive reason for non-ready states.
    """

    hf_dataset_id: str | None
    cache_dataset_dir: Path | None
    direct_dataset_dir: Path | None = None
    local_layout: str = "missing"
    expected_revision: str | None = None
    ref_name: str | None = None
    referenced_snapshot: str | None = None
    revision_matches: bool = True
    snapshot_dirs: tuple[Path, ...] = ()
    file_count: int = 0
    expected_file_count: int | None = None
    missing_file_count: int = 0
    direct_file_count: int = 0
    direct_file_count_capped: bool = False
    direct_revision: str | None = None
    direct_revision_matches: bool = True
    direct_ready: bool = False
    incomplete_files: tuple[Mapping[str, JsonValue], ...] = ()
    direct_incomplete_files: tuple[Mapping[str, JsonValue], ...] = ()
    broken_links: tuple[str, ...] = ()
    ready: bool = False
    status: str = "not_ready"
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """Check if the local dataset is fully ready for evaluation.

        Returns:
            True if ready, False otherwise.
        """
        return self.ready

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this local status to a plain dictionary.

        Returns:
            Dictionary of status fields.
        """
        payload = _plain(asdict(self))
        payload["ok"] = self.ok
        return payload


def _read_ref(path: Path) -> str | None:
    """Read the content of a Git reference pointer file (e.g. refs/heads/main).

    Args:
        path: Path to the reference file.

    Returns:
        Content string, or None.
    """
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _status_from_local_check(
    *,
    dataset_dir: Path,
    snapshot_dirs: tuple[Path, ...],
    revision_matches: bool,
    file_count: int,
    expected_file_count: int | None = None,
    incomplete_files: tuple[Mapping[str, JsonValue], ...],
    broken_links: tuple[str, ...],
) -> str:
    """Determine the combined local readiness status based on individual validation results.

    Args:
        dataset_dir: Checked cache folder path.
        snapshot_dirs: Snapshot directories checked.
        revision_matches: Revision validation result.
        file_count: Discovered file count.
        expected_file_count: Total expected file count.
        incomplete_files: Incomplete files detected.
        broken_links: Broken links detected.

    Returns:
        Status string (e.g. 'ready', 'not_found', 'incomplete_files', etc.).
    """
    if not dataset_dir.exists():
        return "not_found"
    if incomplete_files:
        return "incomplete_files"
    if broken_links:
        return "broken_links"
    if not revision_matches:
        return "revision_mismatch"
    if not snapshot_dirs or not all(path.exists() for path in snapshot_dirs):
        return "missing_snapshot"
    if file_count <= 0:
        return "empty_snapshot"
    if expected_file_count is not None and file_count < expected_file_count:
        return "incomplete_snapshot"
    return "ready"


def _shallow_incomplete_files(
    root: Path, *, limit: int = _DIRECT_DATASET_INCOMPLETE_LIMIT
) -> tuple[Mapping[str, JsonValue], ...]:
    """Perform a shallow fast scan to detect any temporary download markers.

    Args:
        root: Root directory to search.
        limit: Capped size of result collection.

    Returns:
        Tuple of incomplete file info mappings.
    """
    incomplete_files: list[Mapping[str, JsonValue]] = []
    seen: set[Path] = set()
    pending = deque([root])
    while pending and len(incomplete_files) < limit:
        directory = pending.popleft()
        if len(incomplete_files) >= limit or not directory.is_dir() or directory in seen:
            continue
        seen.add(directory)
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                if len(incomplete_files) >= limit:
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        continue
                except OSError:
                    continue
                if not entry.name.endswith((".incomplete", ".aria2")):
                    continue
                path = Path(entry.path)
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                incomplete_files.append({"path": str(path), "size_bytes": path.stat().st_size})
    return tuple(incomplete_files)


def _snapshot_scan(snapshot_dir: Path) -> tuple[int, tuple[str, ...]]:
    """Scan snapshot folder recursively to count files and check for broken symlinks.

    Args:
        snapshot_dir: Snapshot directory Path.

    Returns:
        Tuple of (file_count_int, broken_links_tuple_of_strings).
    """
    if not snapshot_dir.is_dir():
        return 0, ()
    file_count = 0
    broken_links: list[str] = []
    pending = deque([snapshot_dir])
    seen: set[Path] = set()
    while pending:
        directory = pending.popleft()
        if directory in seen:
            continue
        seen.add(directory)
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                path = Path(entry.path)
                try:
                    if entry.is_symlink():
                        if not path.exists():
                            broken_links.append(str(path))
                        else:
                            file_count += 1
                        continue
                    if entry.is_file(follow_symlinks=False):
                        file_count += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                except OSError:
                    continue
    return file_count, tuple(broken_links)


def _expected_file_count_from_ref(ref: DatasetRef) -> int | None:
    """Retrieve expected dataset file count from metadata fields or siblings list.

    Args:
        ref: DatasetRef specification object.

    Returns:
        The expected file count, or None if unknown.
    """
    for source in (ref.metadata,):
        if not isinstance(source, Mapping):
            continue
        for key in ("expected_file_count", "expected_sibling_count", "sibling_count"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdigit() and int(value) > 0:
                return int(value)
        siblings = source.get("siblings")
        if isinstance(siblings, Sequence) and not isinstance(siblings, (str, bytes)):
            return len(siblings)
    return None


def _direct_dataset_scan(
    dataset_dir: Path,
    *,
    file_limit: int | None = None,
    incomplete_limit: int | None = None,
) -> tuple[int, bool, tuple[Mapping[str, JsonValue], ...]]:
    """Perform a comprehensive check on files under a direct dataset folder layout.

    Checks presence of expected siblings, identifies missing files, and detects incomplete downloads.

    Args:
        dataset_dir: Direct dataset layout directory Path.
        file_limit: Maximum files to scan.
        incomplete_limit: Maximum incomplete files to collect.

    Returns:
        Tuple of (file_count, file_count_capped, incomplete_files).
    """
    if not dataset_dir.is_dir():
        return 0, False, ()
    file_limit = _DIRECT_DATASET_FILE_COUNT_LIMIT if file_limit is None else file_limit
    incomplete_limit = _DIRECT_DATASET_INCOMPLETE_LIMIT if incomplete_limit is None else incomplete_limit

    def add_incomplete(path: Path, *, kind: str | None = None, relative_path: str | None = None) -> None:
        if len(incomplete_files) >= incomplete_limit:
            return
        item: dict[str, JsonValue] = {"path": str(path)}
        if path.exists():
            item["size_bytes"] = path.stat().st_size
        if kind:
            item["kind"] = kind
        if relative_path:
            item["relative_path"] = relative_path
        incomplete_files.append(item)

    file_count = 0
    file_count_capped = False
    incomplete_files: list[Mapping[str, JsonValue]] = []
    resume_expected_files = _direct_dataset_resume_expected_files(dataset_dir)

    if resume_expected_files:
        for relative_path in resume_expected_files:
            path = dataset_dir / relative_path
            if path.is_file():
                if file_count < file_limit:
                    file_count += 1
                else:
                    file_count_capped = True
            else:
                add_incomplete(path, kind="missing_expected_file", relative_path=relative_path)
                if len(incomplete_files) >= incomplete_limit:
                    break
        return file_count, file_count_capped, tuple(incomplete_files)

    expected_files = _direct_dataset_expected_files(dataset_dir)

    if expected_files:
        expected_to_check: list[str] = []
        seen_expected: set[str] = set()
        if expected_files and len(expected_files) > file_limit:
            file_count_capped = True
        for relative_path in expected_files[:file_limit]:
            expected_to_check.append(relative_path)
            seen_expected.add(relative_path)

        for relative_path in expected_to_check:
            path = dataset_dir / relative_path
            if path.is_file():
                if file_count < file_limit:
                    file_count += 1
                else:
                    file_count_capped = True
            else:
                add_incomplete(path, kind="missing_expected_file", relative_path=relative_path)
                if len(incomplete_files) >= incomplete_limit:
                    break

        return file_count, file_count_capped, tuple(incomplete_files)

    try:
        entries = os.scandir(dataset_dir)
    except OSError:
        return 0, False, ()
    with entries:
        for entry in entries:
            if entry.name in {".hfd", ".cache", ".git"}:
                continue
            path = Path(entry.path)
            if entry.name.endswith((".incomplete", ".aria2")):
                add_incomplete(path)
                continue
            try:
                if entry.is_file(follow_symlinks=False) or entry.is_dir(follow_symlinks=False):
                    file_count += 1
            except OSError:
                continue
            if file_count >= file_limit:
                file_count_capped = True
                add_incomplete(dataset_dir, kind="direct_hfd_scan_truncated")
                break
    return file_count, file_count_capped, tuple(incomplete_files)


def _safe_direct_dataset_relative_path(value: Any) -> str | None:
    """Validate and normalize a direct layout relative path string.

    Filters out absolute paths, parent directory traversals, and ignored names.

    Args:
        value: Input path value.

    Returns:
        Normalized relative path string or None.
    """
    text = _optional_str(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.name in _IGNORED_DIRECT_DATASET_SIBLING_NAMES or any(
        part in _IGNORED_DIRECT_DATASET_SIBLING_PARTS for part in path.parts
    ):
        return None
    return path.as_posix()


def _direct_dataset_expected_files(dataset_dir: Path) -> tuple[str, ...]:
    """Retrieve expected direct dataset file names listed in HuggingFace metadata.

    Args:
        dataset_dir: Root dataset directory Path.

    Returns:
        Tuple of expected relative file paths.
    """
    metadata_path = dataset_dir / ".hfd" / "repo_metadata.json"
    if not metadata_path.is_file():
        return ()
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        return ()

    expected: list[str] = []
    seen: set[str] = set()
    for item in siblings:
        value = item.get("rfilename") if isinstance(item, Mapping) else item
        normalized = _safe_direct_dataset_relative_path(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        expected.append(normalized)
    return tuple(expected)


def _direct_dataset_resume_expected_files(dataset_dir: Path) -> tuple[str, ...]:
    """Return relative paths from an HFD aria2 resume list."""

    urls_path = dataset_dir / ".hfd" / "aria2c_urls.txt"
    if not urls_path.is_file():
        return ()
    try:
        lines = urls_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    expected: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = line.strip()
        if not text.startswith(("http://", "https://")) or "/resolve/" not in text:
            continue
        suffix = text.split("#", 1)[0].split("?", 1)[0].split("/resolve/", 1)[1]
        parts = suffix.split("/", 1)
        if len(parts) != 2:
            continue
        normalized = _safe_direct_dataset_relative_path(unquote(parts[1]))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        expected.append(normalized)
    return tuple(expected)


def direct_dataset_expected_files(dataset_dir: str | Path) -> tuple[str, ...]:
    """Return expected relative files listed by an HFD direct-layout metadata file."""

    return _direct_dataset_expected_files(Path(dataset_dir))


def _direct_dataset_revision(dataset_dir: Path) -> str | None:
    local_dir_metadata = dataset_dir / ".cache" / "huggingface" / "download"
    local_dir_revision = _direct_dataset_download_metadata_revision(local_dir_metadata)
    if local_dir_revision:
        return local_dir_revision

    metadata_path = dataset_dir / ".hfd" / "repo_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        revision = payload.get("sha") or payload.get("revision") or payload.get("commit")
        if isinstance(revision, str) and revision.strip():
            return revision.strip()

    return None


def _direct_dataset_download_metadata_revision(local_dir_metadata: Path) -> str | None:
    """Read a bounded sample of Hugging Face download metadata revisions.

    Args:
        local_dir_metadata: Direct-layout ``.cache/huggingface/download`` directory.
    """

    if not local_dir_metadata.is_dir():
        return None
    revisions: set[str] = set()
    pending = deque([local_dir_metadata])
    seen_dirs = 0
    seen_files = 0
    while (
        pending and seen_dirs < _DIRECT_DATASET_METADATA_DIR_LIMIT and seen_files < _DIRECT_DATASET_METADATA_FILE_LIMIT
    ):
        directory = pending.popleft()
        seen_dirs += 1
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if seen_files >= _DIRECT_DATASET_METADATA_FILE_LIMIT:
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".metadata"):
                        continue
                except OSError:
                    continue
                path = Path(entry.path)
                seen_files += 1
                try:
                    first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
                except (OSError, IndexError, UnicodeDecodeError):
                    continue
                if first_line:
                    revisions.add(first_line)
                if len(revisions) > 1:
                    return None
    return next(iter(revisions)) if len(revisions) == 1 else None


def _direct_revision_matches(revision: str | None, direct_revision: str | None) -> bool:
    if revision and is_commit_like_revision(revision):
        return direct_revision == revision
    if revision:
        return bool(direct_revision)
    return True


def _direct_dataset_status(
    *,
    dataset_dir: Path,
    revision_matches: bool,
    file_count: int,
    incomplete_files: tuple[Mapping[str, JsonValue], ...],
) -> str:
    if not dataset_dir.exists():
        return "not_found"
    if incomplete_files:
        return "direct_hfd_incomplete_files"
    if file_count <= 0:
        return "direct_hfd_empty"
    if not revision_matches:
        return "direct_hfd_revision_mismatch"
    return "ready"


def check_local_dataset(
    ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
    cache_dir: str | Path,
    expected_revision: str | None = None,
) -> DatasetLocalStatus:
    """Verifies the presence and completeness of a dataset in the local Hugging Face cache.

    Analyzes whether files are complete, matches commit revision hashes, detects partial/corrupted
    downloads (e.g. unfinished `.hfd` tasks or empty cache locks), and computes statistics.

    Args:
        ref: The dataset reference to check (ID, ref object, or manifest dictionary).
        cache_dir: The root directory containing Hugging Face caches.
        expected_revision: An optional commit hash or branch name to enforce.

    Returns:
        DatasetLocalStatus: Object capturing readiness state, checked folder path, and error diagnoses.
    """
    parsed = _first_dataset_ref(ref)

    revision = expected_revision or parsed.revision
    if parsed.not_applicable:
        return DatasetLocalStatus(
            hf_dataset_id=parsed.hf_dataset_id,
            cache_dataset_dir=None,
            expected_revision=revision,
            ready=True,
            status="not_applicable",
            reason=parsed.reason or "dataset not applicable",
        )
    if not parsed.hf_dataset_id:
        return DatasetLocalStatus(
            hf_dataset_id=None,
            cache_dataset_dir=None,
            expected_revision=revision,
            revision_matches=False,
            ready=False,
            status="missing_dataset_id",
            reason="dataset ref does not include an hf_dataset_id",
        )

    dataset_dir = hf_cache_dataset_dir(cache_dir, parsed.hf_dataset_id)
    refs_dir = dataset_dir / "refs"
    snapshots_dir = dataset_dir / "snapshots"
    ref_name = revision if revision and not is_commit_like_revision(revision) else "main"
    referenced_snapshot = _read_ref(refs_dir / ref_name)

    snapshot_dirs: list[Path] = []
    if revision and is_commit_like_revision(revision):
        snapshot_dirs.append(snapshots_dir / revision)
    elif (revision and referenced_snapshot) or referenced_snapshot:
        snapshot_dirs.append(snapshots_dir / referenced_snapshot)
    elif snapshots_dir.is_dir():
        snapshot_dirs.extend(path for path in sorted(snapshots_dir.iterdir()) if path.is_dir())

    incomplete_files: tuple[Mapping[str, JsonValue], ...] = ()
    if dataset_dir.exists():
        incomplete_files = _shallow_incomplete_files(dataset_dir)

    broken_links: list[str] = []
    file_count = 0
    for snapshot_dir in snapshot_dirs:
        snapshot_file_count, snapshot_broken_links = _snapshot_scan(snapshot_dir)
        file_count += snapshot_file_count
        broken_links.extend(snapshot_broken_links)

    if revision and is_commit_like_revision(revision):
        revision_matches = (snapshots_dir / revision).is_dir() or referenced_snapshot == revision
    elif revision:
        revision_matches = bool(referenced_snapshot) and (snapshots_dir / referenced_snapshot).is_dir()
    else:
        revision_matches = True

    snapshot_tuple = tuple(snapshot_dirs)
    broken_link_tuple = tuple(broken_links)
    expected_file_count = _expected_file_count_from_ref(parsed)
    missing_file_count = max((expected_file_count or 0) - file_count, 0)
    status = _status_from_local_check(
        dataset_dir=dataset_dir,
        snapshot_dirs=snapshot_tuple,
        revision_matches=revision_matches,
        file_count=file_count,
        expected_file_count=expected_file_count,
        incomplete_files=incomplete_files,
        broken_links=broken_link_tuple,
    )
    hf_ready = status == "ready"
    if hf_ready:
        return DatasetLocalStatus(
            hf_dataset_id=parsed.hf_dataset_id,
            cache_dataset_dir=dataset_dir,
            direct_dataset_dir=None,
            local_layout="hf_cache",
            expected_revision=revision,
            ref_name=ref_name,
            referenced_snapshot=referenced_snapshot,
            revision_matches=revision_matches,
            snapshot_dirs=snapshot_tuple,
            file_count=file_count,
            expected_file_count=expected_file_count,
            missing_file_count=missing_file_count,
            direct_file_count=0,
            direct_file_count_capped=False,
            direct_revision=None,
            direct_revision_matches=False,
            direct_ready=False,
            incomplete_files=incomplete_files,
            direct_incomplete_files=(),
            broken_links=broken_link_tuple,
            ready=True,
            status=status,
        )
    direct_candidates = direct_dataset_dir_candidates_for_ref(cache_dir, parsed)
    direct_states: list[tuple[Path, int, bool, tuple[Mapping[str, JsonValue], ...], str | None, bool, str]] = []
    for candidate in direct_candidates:
        direct_file_count, direct_file_count_capped, direct_incomplete_files = _direct_dataset_scan(candidate)
        direct_revision = _direct_dataset_revision(candidate)
        direct_revision_match = _direct_revision_matches(revision, direct_revision)
        direct_status = _direct_dataset_status(
            dataset_dir=candidate,
            revision_matches=direct_revision_match,
            file_count=direct_file_count,
            incomplete_files=direct_incomplete_files,
        )
        direct_states.append(
            (
                candidate,
                direct_file_count,
                direct_file_count_capped,
                direct_incomplete_files,
                direct_revision,
                direct_revision_match,
                direct_status,
            )
        )
        if direct_status == "ready":
            break
    direct_state = next((item for item in direct_states if item[6] == "ready"), None)
    if direct_state is None:
        direct_state = next((item for item in direct_states if item[0].exists()), direct_states[0])
    (
        direct_dir,
        direct_file_count,
        direct_file_count_capped,
        direct_incomplete_files,
        direct_revision,
        direct_revision_match,
        direct_status,
    ) = direct_state
    direct_ready = direct_status == "ready"
    ready = hf_ready or direct_ready
    combined_status = "ready" if ready else (direct_status if status == "not_found" else status)
    if hf_ready:
        local_layout = "hf_cache"
    elif direct_dir.exists():
        local_layout = "direct_hfd"
    elif status != "not_found":
        local_layout = "hf_cache"
    else:
        local_layout = "missing"
    return DatasetLocalStatus(
        hf_dataset_id=parsed.hf_dataset_id,
        cache_dataset_dir=dataset_dir,
        direct_dataset_dir=direct_dir,
        local_layout=local_layout,
        expected_revision=revision,
        ref_name=ref_name,
        referenced_snapshot=referenced_snapshot,
        revision_matches=revision_matches or direct_revision_match,
        snapshot_dirs=snapshot_tuple,
        file_count=file_count or direct_file_count,
        expected_file_count=expected_file_count,
        missing_file_count=missing_file_count,
        direct_file_count=direct_file_count,
        direct_file_count_capped=direct_file_count_capped,
        direct_revision=direct_revision,
        direct_revision_matches=direct_revision_match,
        direct_ready=direct_ready,
        incomplete_files=incomplete_files,
        direct_incomplete_files=direct_incomplete_files,
        broken_links=broken_link_tuple,
        ready=ready,
        status=combined_status,
    )


__all__ = [
    "DatasetLocalStatus",
    "check_local_dataset",
    "direct_dataset_dir_candidates",
    "direct_dataset_dir_candidates_for_ref",
    "direct_dataset_expected_files",
    "hf_cache_dataset_dir",
]
