"""Dataset access policy: license and gating classification and Hugging Face download commands."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..catalog.schema import BenchmarkDatasetRef
from .refs import DatasetRef, JsonValue, _access_value_is_truthy, _first_dataset_ref, _optional_str, _plain

_OPEN_LICENSES = frozenset(
    {
        "afl-3.0",
        "agpl-3.0",
        "apache-2.0",
        "artistic-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-3-clause-clear",
        "bsl-1.0",
        "cc-by-2.0",
        "cc-by-2.5",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
        "cc0-1.0",
        "cdla-permissive-1.0",
        "gpl-2.0",
        "gpl-3.0",
        "isc",
        "lgpl-2.1",
        "lgpl-3.0",
        "mit",
        "mpl-2.0",
        "odc-by",
        "postgresql",
        "unlicense",
        "wtfpl",
        "zlib",
    }
)


_LICENSE_REVIEW_MARKERS = (
    "custom",
    "non-commercial",
    "noncommercial",
    "proprietary",
    "research",
    "unknown",
)


@dataclass(frozen=True)
class DatasetAccessIssue:
    """Represents a specific compliance, licensing, or security issue found in a dataset reference.

    Attributes:
        code: Stable string identifier for the type of issue.
        severity: Severity indicator (e.g. 'error', 'restricted', 'warning').
        message: Explanatory diagnostic message.
        field: Reference field name implicated in this issue.
        value: Offending value, if any.
    """

    code: str
    severity: str
    message: str
    field: str | None = None
    value: JsonValue | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this issue to a plain dictionary.

        Returns:
            Dictionary of issue fields.
        """
        return _plain(asdict(self))


@dataclass(frozen=True)
class DatasetAccessReport:
    """Represents a security and accessibility audit report for a dataset reference.

    Attributes:
        hf_dataset_id: Normalised Hugging Face repository ID.
        access_status: Accessibility category ('public', 'restricted', etc.).
        license_status: License type ('open', 'review_required', etc.).
        requires_auth: Whether access needs Hugging Face authorization credentials.
        issues: Tuple of detected compliance/access issues.
    """

    hf_dataset_id: str | None
    access_status: str
    license_status: str
    requires_auth: bool = False
    issues: tuple[DatasetAccessIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Check if the dataset is freely accessible with an open license.

        Returns:
            True if fully public and openly licensed, False otherwise.
        """
        return self.access_status == "public" and self.license_status == "open"

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert this report to a plain dictionary.

        Returns:
            Dictionary of report fields.
        """
        payload = _plain(asdict(self))
        payload["ok"] = self.ok
        return payload


def _normalize_license(value: str | None) -> str | None:
    """Clean and normalize a raw license identifier string.

    Args:
        value: Raw license string.

    Returns:
        Normalized lowercase hyphenated license ID, or None.
    """
    text = _optional_str(value)
    if text is None:
        return None
    return text.casefold().replace("_", "-").replace(" ", "-")


def classify_dataset_access(ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any]) -> DatasetAccessReport:
    """Classify dataset access rules, privacy settings, and license requirements.

    Args:
        ref: Dataset reference source.

    Returns:
        The generated DatasetAccessReport.
    """
    parsed = _first_dataset_ref(ref)
    if parsed.not_applicable:
        return DatasetAccessReport(
            hf_dataset_id=parsed.hf_dataset_id,
            access_status="not_applicable",
            license_status="not_applicable",
            requires_auth=False,
            issues=(),
        )

    issues: list[DatasetAccessIssue] = []
    if not parsed.hf_dataset_id:
        issues.append(
            DatasetAccessIssue(
                code="missing_hf_dataset_id",
                severity="error",
                message="dataset ref does not include an hf_dataset_id",
                field="hf_dataset_id",
            )
        )

    if parsed.private is True:
        issues.append(
            DatasetAccessIssue(
                code="private_dataset",
                severity="restricted",
                message="dataset is marked private",
                field="private",
                value=parsed.private,
            )
        )
    if _access_value_is_truthy(parsed.gated):
        issues.append(
            DatasetAccessIssue(
                code="gated_dataset",
                severity="restricted",
                message="dataset is marked gated and may require access approval",
                field="gated",
                value=parsed.gated,
            )
        )
    if parsed.requires_auth and not any(issue.code in {"private_dataset", "gated_dataset"} for issue in issues):
        issues.append(
            DatasetAccessIssue(
                code="requires_auth",
                severity="restricted",
                message="dataset requires Hugging Face authentication",
                field="requires_auth",
                value=True,
            )
        )

    license_value = _normalize_license(parsed.license)
    if license_value is None:
        license_status = "missing"
        issues.append(
            DatasetAccessIssue(
                code="missing_license",
                severity="warning",
                message="dataset ref does not declare a license",
                field="license",
            )
        )
    elif license_value in _OPEN_LICENSES:
        license_status = "open"
    elif (
        license_value in {"other", "unknown"}
        or "cc-by-nc" in license_value
        or any(marker in license_value for marker in _LICENSE_REVIEW_MARKERS)
    ):
        license_status = "review_required"
        issues.append(
            DatasetAccessIssue(
                code="license_review_required",
                severity="warning",
                message="dataset license should be reviewed before redistribution or automated use",
                field="license",
                value=parsed.license,
            )
        )
    else:
        license_status = "review_required"
        issues.append(
            DatasetAccessIssue(
                code="unrecognized_license",
                severity="warning",
                message="dataset license is not in the known open-license allowlist",
                field="license",
                value=parsed.license,
            )
        )

    if not parsed.hf_dataset_id:
        access_status = "missing_dataset_id"
    elif parsed.requires_auth:
        access_status = "restricted"
    else:
        access_status = "public"

    return DatasetAccessReport(
        hf_dataset_id=parsed.hf_dataset_id,
        access_status=access_status,
        license_status=license_status,
        requires_auth=parsed.requires_auth,
        issues=tuple(issues),
    )


def find_hf_downloader() -> tuple[str, ...]:
    """Find the available Hugging Face command line downloader executable on the system PATH.

    Returns:
        Tuple containing the executable name/path and argument (e.g. ('hf', 'download')).

    Raises:
        FileNotFoundError: If neither 'hf' nor 'huggingface-cli' is present.
    """
    hf = shutil.which("hf")
    if hf:
        return (hf, "download")
    huggingface_cli = shutil.which("huggingface-cli")
    if huggingface_cli:
        return (huggingface_cli, "download")
    raise FileNotFoundError("neither 'hf' nor 'huggingface-cli' was found on PATH")


def build_hf_download_command(
    ref: DatasetRef | BenchmarkDatasetRef | Mapping[str, Any] | str,
    cache_dir: str | Path,
    downloader: Sequence[str] = ("hf", "download"),
) -> tuple[str, ...]:
    """Construct the command string to run the Hugging Face CLI downloader for a dataset ref.

    Args:
        ref: Dataset reference.
        cache_dir: Destination cache folder.
        downloader: Downloader executable sequence.

    Returns:
        Command arguments tuple.

    Raises:
        ValueError: If the dataset reference is missing its hf_dataset_id.
    """
    parsed = _first_dataset_ref(ref)
    if not parsed.hf_dataset_id:
        raise ValueError("cannot build a Hugging Face download command without hf_dataset_id")
    command = [*downloader, parsed.hf_dataset_id, "--repo-type", "dataset", "--cache-dir", str(Path(cache_dir))]
    if parsed.revision:
        command.extend(("--revision", parsed.revision))
    return tuple(command)


__all__ = [
    "DatasetAccessIssue",
    "DatasetAccessReport",
    "build_hf_download_command",
    "classify_dataset_access",
    "find_hf_downloader",
]
