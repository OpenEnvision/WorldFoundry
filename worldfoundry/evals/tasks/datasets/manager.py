"""DatasetManager: check, locate, and plan downloads for the datasets a benchmark declares."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..catalog.schema import BenchmarkDatasetRef, BenchmarkZooEntry
from .access import DatasetAccessReport, build_hf_download_command, classify_dataset_access
from .local_status import DatasetLocalStatus, check_local_dataset, hf_cache_dataset_dir
from .location import DatasetLocation, locate_local_dataset
from .refs import (
    DatasetRef,
    JsonValue,
    _dedupe_download_refs,
    _plain,
    filter_dataset_refs,
    normalize_hf_dataset_id,
    parse_dataset_refs,
)


@dataclass(frozen=True)
class DatasetDownloadPlan:
    """Represents a cohesive downloading execution plan for one or more datasets.

    Attributes:
        refs: Tuple of mapped DatasetRef specs involved.
        commands: Tuple of tuple command arguments to run.
        cache_dir: Root cache directory where datasets should download.
        access_reports: Accessibility and compliance reports per dataset.
        local_checks: Local readiness status structures, if checked.
        metadata: Optional dictionary detailing overall plan statistics.
    """

    refs: tuple[DatasetRef, ...]
    commands: tuple[tuple[str, ...], ...]
    cache_dir: Path
    access_reports: tuple[DatasetAccessReport, ...] = ()
    local_checks: tuple[DatasetLocalStatus, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Get the unique Hugging Face dataset repository IDs involved in this plan.

        Returns:
            Tuple of dataset ID strings.
        """
        dataset_ids: list[str] = []
        for ref in self.refs:
            if ref.hf_dataset_id and ref.hf_dataset_id not in dataset_ids:
                dataset_ids.append(ref.hf_dataset_id)
        return tuple(dataset_ids)

    @property
    def requires_auth(self) -> bool:
        """Evaluate if any of the datasets in the plan requires credentials/authentication.

        Returns:
            True if any dataset is restricted/gated, False otherwise.
        """
        return any(report.requires_auth for report in self.access_reports)

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this download plan to a plain dictionary.

        Returns:
            Plain dictionary representation.
        """
        payload = _plain(asdict(self))
        payload["dataset_ids"] = list(self.dataset_ids)
        payload["requires_auth"] = self.requires_auth
        return payload


def _metadata_for_plan(
    *,
    source: Any,
    all_refs: tuple[DatasetRef, ...],
    selected_refs: tuple[DatasetRef, ...],
    requested_dataset_ids: tuple[str, ...],
    metadata_mode: str,
) -> dict[str, JsonValue]:
    """Generate plan statistics and diagnostics for a planned download execution.

    Args:
        source: BenchmarkZooEntry or mapping representing dataset source.
        all_refs: Parsed DatasetRefs.
        selected_refs: Selected DatasetRefs.
        requested_dataset_ids: Target dataset repo IDs requested.
        metadata_mode: 'filtered' or 'full'.

    Returns:
        Structured plan metadata dictionary.
    """
    if metadata_mode not in {"filtered", "full"}:
        raise ValueError("metadata_mode must be one of: filtered, full")

    benchmark_id = None
    if isinstance(source, BenchmarkZooEntry):
        benchmark_id = source.benchmark_id
    elif isinstance(source, Mapping):
        value = source.get("benchmark_id") or source.get("id")
        benchmark_id = str(value) if value is not None else None

    selected_dataset_ids = {ref.hf_dataset_id for ref in selected_refs}
    metadata: dict[str, JsonValue] = {
        "metadata_mode": metadata_mode,
        "benchmark_id": benchmark_id,
        "filtered": bool(requested_dataset_ids),
        "requested_dataset_ids": list(requested_dataset_ids),
        "hf_dataset_ids": [ref.hf_dataset_id for ref in selected_refs if ref.hf_dataset_id],
        "dataset_refs": [ref.to_dict() for ref in selected_refs],
        "missing_requested_dataset_ids": [
            dataset_id for dataset_id in requested_dataset_ids if dataset_id not in selected_dataset_ids
        ],
    }
    not_applicable = next((ref for ref in selected_refs or all_refs if ref.not_applicable), None)
    if not_applicable is not None:
        metadata["dataset_not_applicable"] = True
        metadata["reason"] = not_applicable.reason

    if metadata_mode == "full":
        metadata["available_hf_dataset_ids"] = [ref.hf_dataset_id for ref in all_refs if ref.hf_dataset_id]
        metadata["available_dataset_refs"] = [ref.to_dict() for ref in all_refs]

    return metadata


class DatasetManager:
    """Reusable management facade for benchmark datasets and Hugging Face cache repositories.

    This class provides unified access points to:
    - Parse and filter dataset reference structures.
    - Resolve cache directories.
    - Check local dataset completeness and file layouts.
    - Locate physical assets on disk (local roots, custom manifests, or Hugging Face cache).
    - Generate automated CLI download plans and commands.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        downloader: Sequence[str] = ("hf", "download"),
    ) -> None:
        """Initializes the manager with a target cache directory and a download utility prefix.

        Args:
            cache_dir: The cache directory Path.
            downloader: Executable name/args for downloader utility.
        """
        self.cache_dir = Path(cache_dir)
        self.downloader = tuple(downloader)

    def parse_refs(self, source: Any) -> tuple[DatasetRef, ...]:
        """Parses arbitrary dataset definitions (manifests, mappings, refs) into clean DatasetRef objects.

        Args:
            source: Source dataset spec or container.

        Returns:
            Tuple of DatasetRef objects.
        """
        return parse_dataset_refs(source)

    def filter_refs(
        self,
        refs: Iterable[DatasetRef],
        dataset_ids: Iterable[str] | None = None,
    ) -> tuple[DatasetRef, ...]:
        """Filters dataset references down to a specific set of active/requested dataset IDs.

        Args:
            refs: Dataset references collection.
            dataset_ids: Dataset repository IDs.

        Returns:
            Filtered tuple of DatasetRefs.
        """
        return filter_dataset_refs(refs, dataset_ids)

    def cache_dataset_dir(self, dataset_id: str) -> Path:
        """Resolves the expected local Hugging Face snapshot cache path for a given dataset ID.

        Args:
            dataset_id: Target dataset ID.

        Returns:
            HuggingFace snapshot cache path.
        """
        return hf_cache_dataset_dir(self.cache_dir, dataset_id)

    def build_download_command(
        self, ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str
    ) -> tuple[str, ...]:
        """Constructs an executable CLI command array to retrieve the target dataset from Hugging Face.

        Args:
            ref: DatasetRef target spec.

        Returns:
            Command arguments tuple.
        """
        return build_hf_download_command(ref, self.cache_dir, self.downloader)

    def classify_access(self, ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any]) -> DatasetAccessReport:
        """Classifies security and license accessibility for a dataset (e.g. public vs gated vs custom license).

        Args:
            ref: Target dataset reference.

        Returns:
            Accessibility report.
        """
        return classify_dataset_access(ref)

    def check_local(
        self,
        ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
        *,
        expected_revision: str | None = None,
    ) -> DatasetLocalStatus:
        """Scans local cache and verifies dataset completeness, files count, and revision hashes.

        Args:
            ref: Dataset target spec.
            expected_revision: Optional expected revision string.

        Returns:
            Readiness status structure.
        """
        return check_local_dataset(ref, self.cache_dir, expected_revision)

    def locate_local(
        self,
        ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
        *,
        data_root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> DatasetLocation:
        """Locate a previously downloaded dataset without network access.

        Priority matches :func:`locate_local_dataset`: per-dataset environment
        override, local manifests, a direct data root, then Hugging Face cache
        snapshots under this manager's ``cache_dir``.

        Args:
            ref: Dataset target spec.
            data_root: Optional data root.
            manifest_path: Optional manifest file path.
            env: Optional environment dictionary.

        Returns:
            The resolved location of the dataset.
        """
        return locate_local_dataset(
            ref,
            data_root=data_root,
            manifest_path=manifest_path,
            cache_dir=self.cache_dir,
            env=env,
        )

    def build_download_plan(
        self,
        source: Any,
        *,
        dataset_ids: Iterable[str] | None = None,
        check_local: bool = False,
        metadata_mode: str = "filtered",
    ) -> DatasetDownloadPlan:
        """Generates a complete, deduplicated download strategy for a collection of dataset sources.

        Args:
            source: Source dataset collection or manifest.
            dataset_ids: Subset of dataset repository IDs.
            check_local: Whether to run a local status scan on selected datasets.
            metadata_mode: Mode label.

        Returns:
            The cohesive download plan.
        """
        all_refs = self.parse_refs(source)
        requested_dataset_ids = tuple(
            dataset_id for dataset_id in (normalize_hf_dataset_id(item) for item in dataset_ids or ()) if dataset_id
        )
        selected_refs = filter_dataset_refs(all_refs, requested_dataset_ids)

        download_refs = _dedupe_download_refs(selected_refs)
        commands = tuple(self.build_download_command(ref) for ref in download_refs if ref.hf_dataset_id)
        local_checks = tuple(self.check_local(ref) for ref in download_refs) if check_local else ()
        access_reports = tuple(classify_dataset_access(ref) for ref in download_refs)
        metadata = _metadata_for_plan(
            source=source,
            all_refs=all_refs,
            selected_refs=download_refs,
            requested_dataset_ids=requested_dataset_ids,
            metadata_mode=metadata_mode,
        )
        return DatasetDownloadPlan(
            refs=download_refs,
            commands=commands,
            cache_dir=self.cache_dir,
            access_reports=access_reports,
            local_checks=local_checks,
            metadata=metadata,
        )


__all__ = [
    "DatasetDownloadPlan",
    "DatasetManager",
]
