#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldfoundry.evaluation.utils import load_manifest  # noqa: E402

DEFAULT_MANIFEST_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "hfd"
MANIFEST_EXTENSIONS = (".json", ".yaml", ".yml")


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    path: Path
    data: dict[str, Any]

    @property
    def hf_repo_id(self) -> str | None:
        repo_ids = self.hf_repo_ids
        return repo_ids[0] if repo_ids else None

    @property
    def hf_repo_ids(self) -> list[str]:
        repo_ids: list[str] = []

        def add(value: Any) -> None:
            repo_id = _hf_repo_id_from_value(value)
            if repo_id and repo_id not in repo_ids:
                repo_ids.append(repo_id)

        add(self.data.get("hf_repo_id"))
        source = self.data.get("source")
        if isinstance(source, dict):
            add(source.get("hf_repo_id"))
        checkpoint = self.data.get("checkpoint")
        if isinstance(checkpoint, dict):
            add(checkpoint.get("hf_repo_id"))
            for repo in checkpoint.get("repos") or ():
                add(repo.get("id") or repo.get("repo_id") if isinstance(repo, dict) else repo)
        for repo in self.data.get("checkpoint_refs") or ():
            add(repo.get("repo_id") or repo.get("id") if isinstance(repo, dict) else repo)
        for repo in self.data.get("checkpoints") or ():
            add(repo.get("repo_id") or repo.get("id") if isinstance(repo, dict) else repo)
        for variant in self.data.get("variants") or ():
            if isinstance(variant, dict):
                for repo in variant.get("checkpoint_refs") or ():
                    add(repo.get("repo_id") or repo.get("id") if isinstance(repo, dict) else repo)
                for repo in variant.get("checkpoints") or ():
                    add(repo.get("repo_id") or repo.get("id") if isinstance(repo, dict) else repo)
        official_sources = self.data.get("official_sources")
        if isinstance(official_sources, dict):
            for key in ("huggingface", "huggingface_models", "hf_models", "models"):
                huggingface = official_sources.get(key)
                items = huggingface if isinstance(huggingface, list) else [huggingface]
                for item in items:
                    add(item.get("repo_id") or item.get("id") or item.get("url") if isinstance(item, dict) else item)
        return repo_ids

    @property
    def hf_repo_revisions(self) -> dict[str, str]:
        revisions: dict[str, str] = {}

        def add(repo_value: Any, revision_value: Any) -> None:
            repo_id = _hf_repo_id_from_value(repo_value)
            if not repo_id or not isinstance(revision_value, str) or not revision_value.strip():
                return
            revisions.setdefault(repo_id, revision_value.strip())

        if "hf_repo_id" in self.data:
            add(self.data.get("hf_repo_id"), self.data.get("revision") or self.data.get("sha"))
        source = self.data.get("source")
        if isinstance(source, dict):
            add(source.get("hf_repo_id"), source.get("revision") or source.get("sha"))
        checkpoint = self.data.get("checkpoint")
        if isinstance(checkpoint, dict):
            add(checkpoint.get("hf_repo_id"), checkpoint.get("revision") or checkpoint.get("sha"))
            for repo in checkpoint.get("repos") or ():
                if isinstance(repo, dict):
                    add(repo.get("id") or repo.get("repo_id"), repo.get("revision") or repo.get("sha") or repo.get("commit"))
        for repo in self.data.get("checkpoint_refs") or ():
            if isinstance(repo, dict):
                add(repo.get("repo_id") or repo.get("id"), repo.get("revision") or repo.get("sha") or repo.get("commit"))
        for repo in self.data.get("checkpoints") or ():
            if isinstance(repo, dict):
                add(repo.get("repo_id") or repo.get("id"), repo.get("revision") or repo.get("sha") or repo.get("commit"))
        for variant in self.data.get("variants") or ():
            if isinstance(variant, dict):
                for repo in variant.get("checkpoint_refs") or ():
                    if isinstance(repo, dict):
                        add(repo.get("repo_id") or repo.get("id"), repo.get("revision") or repo.get("sha") or repo.get("commit"))
                for repo in variant.get("checkpoints") or ():
                    if isinstance(repo, dict):
                        add(repo.get("repo_id") or repo.get("id"), repo.get("revision") or repo.get("sha") or repo.get("commit"))
        parity = self.data.get("demo_parity")
        if isinstance(parity, dict):
            add(parity.get("checkpoint_repo_id"), parity.get("checkpoint_snapshot") or parity.get("checkpoint_revision"))
        official_sources = self.data.get("official_sources")
        if isinstance(official_sources, dict):
            for key in ("huggingface", "huggingface_models", "hf_models", "models"):
                huggingface = official_sources.get(key)
                items = huggingface if isinstance(huggingface, list) else [huggingface]
                for item in items:
                    if isinstance(item, dict):
                        add(
                            item.get("repo_id") or item.get("id") or item.get("url"),
                            item.get("revision") or item.get("sha") or item.get("commit"),
                        )
        return revisions

    @property
    def hf_repo_files(self) -> dict[str, list[str]]:
        """Return explicitly requested files grouped by Hugging Face repo id."""

        files: dict[str, list[str]] = {}

        def add(repo_value: Any, filename_value: Any) -> None:
            repo_id = _hf_repo_id_from_value(repo_value)
            filename = _normalize_hf_filename(filename_value)
            if not repo_id or not filename:
                return
            repo_files = files.setdefault(repo_id, [])
            if filename not in repo_files:
                repo_files.append(filename)

        def mappings(value: Any) -> tuple[dict[str, Any], ...]:
            if isinstance(value, dict):
                values = value.values()
            else:
                values = value or ()
            return tuple(item for item in values if isinstance(item, dict))

        checkpoint = self.data.get("checkpoint")
        if isinstance(checkpoint, dict):
            add(checkpoint.get("hf_repo_id"), checkpoint.get("filename"))
            for repo in mappings(checkpoint.get("repos")):
                add(repo.get("id") or repo.get("repo_id"), repo.get("filename"))
        for key in ("checkpoint_refs", "checkpoints"):
            for repo in mappings(self.data.get(key)):
                add(repo.get("repo_id") or repo.get("id"), repo.get("filename"))
        for variant in self.data.get("variants") or ():
            if not isinstance(variant, dict):
                continue
            for key in ("checkpoint_refs", "checkpoints"):
                for repo in mappings(variant.get(key)):
                    add(repo.get("repo_id") or repo.get("id"), repo.get("filename"))
        official_sources = self.data.get("official_sources")
        if isinstance(official_sources, dict):
            for key in ("huggingface", "huggingface_models", "hf_models", "models"):
                values = official_sources.get(key)
                items = values if isinstance(values, list) else [values]
                for item in items:
                    if isinstance(item, dict):
                        add(
                            item.get("repo_id") or item.get("id") or item.get("url"),
                            item.get("filename"),
                        )
        return files


def _hf_repo_id_from_value(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    marker = "huggingface.co/"
    if marker not in value:
        return value
    suffix = value.split(marker, 1)[1].strip("/")
    if suffix.startswith("spaces/"):
        return None
    parts = suffix.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def _normalize_hf_filename(value: Any) -> str | None:
    """Normalize one repository-relative filename without allowing traversal."""

    if not isinstance(value, str) or not value.strip():
        return None
    path = PurePosixPath(value.strip())
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _is_commit_like_revision(value: str | None) -> bool:
    if not value:
        return False
    return len(value) >= 7 and all(char in "0123456789abcdefABCDEF" for char in value)


def _iter_manifest_payloads(path: Path) -> list[dict[str, Any]]:
    payload = load_manifest(path)
    if isinstance(payload, dict):
        for key in ("models", "entries", "model_zoo", "manifests"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"{path} must contain a manifest object, a list of objects, or an object with a model list")


def load_manifests(manifest_dir: Path, model_id: str | None = None) -> list[ModelManifest]:
    if not manifest_dir.exists():
        raise FileNotFoundError(f"manifest directory does not exist: {manifest_dir}")
    if not manifest_dir.is_dir():
        raise NotADirectoryError(f"manifest path is not a directory: {manifest_dir}")

    manifests: list[ModelManifest] = []
    paths = [
        path
        for path in sorted(manifest_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in MANIFEST_EXTENSIONS
    ]
    for path in paths:
        payloads = _iter_manifest_payloads(path)
        for index, data in enumerate(payloads):
            raw_model_id = data.get("model_id") or data.get("id")
            if not isinstance(raw_model_id, str) or not raw_model_id.strip():
                if len(payloads) == 1:
                    raw_model_id = path.stem
                else:
                    raise ValueError(f"{path} entry {index} is missing model_id")
            manifest = ModelManifest(model_id=raw_model_id.strip(), path=path, data=data)
            if model_id is None or manifest.model_id == model_id:
                manifests.append(manifest)

    if model_id is not None and not manifests:
        raise ValueError(f"model_id not found in {manifest_dir}: {model_id}")
    return manifests


def find_hf_downloader() -> list[str]:
    hf = shutil.which("hf")
    if hf:
        return [hf, "download"]
    huggingface_cli = shutil.which("huggingface-cli")
    if huggingface_cli:
        return [huggingface_cli, "download"]
    raise FileNotFoundError("neither 'hf' nor 'huggingface-cli' was found on PATH")


def build_download_command(
    repo_id: str,
    cache_dir: Path,
    downloader: list[str] | None = None,
    revision: str | None = None,
    max_workers: int | None = None,
    filenames: list[str] | None = None,
) -> list[str]:
    prefix = downloader if downloader is not None else find_hf_downloader()
    command = [*prefix, repo_id, *(filenames or ()), "--cache-dir", str(cache_dir)]
    if revision:
        command.extend(["--revision", revision])
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    return command


def hf_cache_repo_dir(cache_dir: Path, repo_id: str) -> Path:
    namespace = repo_id.replace("/", "--")
    return cache_dir / f"models--{namespace}"


def direct_hfd_repo_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / repo_id.replace("/", "--")


def _direct_hfd_files(repo_dir: Path) -> list[Path]:
    if not repo_dir.is_dir():
        return []
    files: list[Path] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if ".hfd" in path.relative_to(repo_dir).parts:
            continue
        if any(part in {".git", ".cache"} for part in path.parts):
            continue
        files.append(path)
    return files


def _direct_hfd_revision(repo_dir: Path) -> str | None:
    metadata_path = repo_dir / ".hfd" / "repo_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        revision = payload.get("sha") or payload.get("revision") or payload.get("commit")
        if isinstance(revision, str) and revision.strip():
            return revision.strip()

    local_dir_metadata = repo_dir / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    if local_dir_metadata.is_dir():
        for path in local_dir_metadata.rglob("*.metadata"):
            try:
                first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
            except (OSError, IndexError, UnicodeDecodeError):
                continue
            if first_line:
                revisions.add(first_line)
    return next(iter(revisions)) if len(revisions) == 1 else None


def check_local_checkpoint(
    repo_id: str,
    cache_dir: Path,
    expected_revision: str | None = None,
    required_files: list[str] | None = None,
) -> dict[str, Any]:
    repo_dir = hf_cache_repo_dir(cache_dir, repo_id)
    direct_repo_dir = direct_hfd_repo_dir(cache_dir, repo_id)
    normalized_required_files = [
        filename
        for value in (required_files or ())
        if (filename := _normalize_hf_filename(value)) is not None
    ]
    refs_dir = repo_dir / "refs"
    snapshots_dir = repo_dir / "snapshots"
    ref_name = expected_revision if expected_revision and not _is_commit_like_revision(expected_revision) else "main"
    ref_path = refs_dir / ref_name
    referenced_snapshot = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None

    snapshot_dirs: list[Path] = []
    if expected_revision:
        snapshot_dirs.append(snapshots_dir / expected_revision)
    elif referenced_snapshot:
        snapshot_dirs.append(snapshots_dir / referenced_snapshot)
    elif snapshots_dir.is_dir():
        snapshot_dirs.extend(path for path in sorted(snapshots_dir.iterdir()) if path.is_dir())

    incomplete_files = [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in sorted(repo_dir.rglob("*.incomplete"))
        if path.is_file()
    ] if repo_dir.exists() else []
    broken_links: list[str] = []
    referenced_blob_names: set[str] = set()
    file_count = 0
    for snapshot_dir in snapshot_dirs:
        if not snapshot_dir.exists():
            continue
        for path in snapshot_dir.rglob("*"):
            if path.is_symlink():
                referenced_blob_names.add(path.readlink().name)
                if not path.exists():
                    broken_links.append(str(path))
            if path.is_file():
                file_count += 1

    if referenced_blob_names:
        blocking_incomplete_files = [
            item
            for item in incomplete_files
            if Path(str(item["path"])).name.removesuffix(".incomplete") in referenced_blob_names
        ]
    else:
        blocking_incomplete_files = incomplete_files
    orphan_incomplete_files = [
        item for item in incomplete_files if item not in blocking_incomplete_files
    ]

    revision_matches = True
    if expected_revision and _is_commit_like_revision(expected_revision):
        if referenced_snapshot:
            revision_matches = referenced_snapshot == expected_revision
        else:
            revision_matches = (snapshots_dir / expected_revision).is_dir()
    elif expected_revision and ref_path.is_file():
        revision_matches = bool(referenced_snapshot) and (snapshots_dir / referenced_snapshot).is_dir()
    elif expected_revision:
        revision_matches = False

    hf_required_files_ready = not normalized_required_files or any(
        all((snapshot_dir / filename).is_file() for filename in normalized_required_files)
        for snapshot_dir in snapshot_dirs
    )
    hf_missing_required_files = [
        filename
        for filename in normalized_required_files
        if not any((snapshot_dir / filename).is_file() for snapshot_dir in snapshot_dirs)
    ]
    hf_ready = (
        bool(snapshot_dirs)
        and all(path.exists() for path in snapshot_dirs)
        and file_count > 0
        and revision_matches
        and hf_required_files_ready
        and not blocking_incomplete_files
        and not broken_links
    )
    direct_files = _direct_hfd_files(direct_repo_dir)
    direct_incomplete_files = [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in sorted(direct_repo_dir.rglob("*.incomplete"))
        if path.is_file()
    ] if direct_repo_dir.exists() else []
    direct_revision = _direct_hfd_revision(direct_repo_dir)
    direct_revision_matches = True
    if expected_revision and _is_commit_like_revision(expected_revision):
        direct_revision_matches = direct_revision == expected_revision
    elif expected_revision:
        direct_revision_matches = bool(direct_revision)
    direct_missing_required_files = [
        filename
        for filename in normalized_required_files
        if not (direct_repo_dir / filename).is_file()
    ]
    direct_ready = (
        bool(direct_files)
        and direct_revision_matches
        and not direct_missing_required_files
        and not direct_incomplete_files
    )
    ready = hf_ready or direct_ready
    return {
        "repo_id": repo_id,
        "cache_repo_dir": str(repo_dir),
        "direct_hfd_repo_dir": str(direct_repo_dir),
        "local_layout": "hf_cache" if hf_ready else ("direct_hfd" if direct_ready else "missing"),
        "expected_revision": expected_revision,
        "referenced_snapshot": referenced_snapshot,
        "revision_matches": revision_matches or direct_revision_matches,
        "snapshot_dirs": [str(path) for path in snapshot_dirs],
        "file_count": file_count or len(direct_files),
        "direct_hfd_file_count": len(direct_files),
        "direct_hfd_metadata_path": str(direct_repo_dir / ".hfd" / "repo_metadata.json"),
        "direct_hfd_revision": direct_revision,
        "direct_hfd_revision_matches": direct_revision_matches,
        "direct_hfd_ready": direct_ready,
        "required_files": normalized_required_files,
        "hf_required_files_ready": hf_required_files_ready,
        "hf_missing_required_files": hf_missing_required_files,
        "direct_hfd_missing_required_files": direct_missing_required_files,
        "incomplete_files": incomplete_files,
        "blocking_incomplete_files": blocking_incomplete_files,
        "orphan_incomplete_files": orphan_incomplete_files,
        "direct_hfd_incomplete_files": direct_incomplete_files,
        "broken_links": broken_links,
        "ready": ready,
        "ok": ready,
    }


def _download_env_summary(env: dict[str, str]) -> dict[str, Any]:
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    return {
        "proxy_env_keys": [key for key in proxy_keys if env.get(key)],
        "hf_hub_disable_xet": env.get("HF_HUB_DISABLE_XET"),
        "hf_hub_enable_hf_transfer": env.get("HF_HUB_ENABLE_HF_TRANSFER"),
    }


def run_download_command(
    command: list[str],
    env: dict[str, str],
    *,
    timeout_seconds: int | None = None,
    retries: int = 0,
) -> dict[str, Any]:
    if retries < 0:
        raise ValueError("retries must be >= 0")
    attempts: list[dict[str, Any]] = []
    total_attempts = retries + 1
    for attempt_index in range(1, total_attempts + 1):
        start = time.monotonic()
        attempt: dict[str, Any] = {
            "attempt": attempt_index,
            "max_attempts": total_attempts,
            "command": command,
            "timeout_seconds": timeout_seconds,
        }
        try:
            completed = subprocess.run(command, env=env, text=True, check=False, timeout=timeout_seconds)
            attempt["returncode"] = completed.returncode
            if getattr(completed, "stdout", None):
                attempt["stdout"] = completed.stdout
            if getattr(completed, "stderr", None):
                attempt["stderr"] = completed.stderr
        except subprocess.TimeoutExpired as exc:
            attempt["returncode"] = None
            attempt["timed_out"] = True
            attempt["error"] = f"download command timed out after {timeout_seconds} seconds"
            if exc.stdout:
                attempt["stdout"] = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
            if exc.stderr:
                attempt["stderr"] = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        attempt["duration_seconds"] = round(time.monotonic() - start, 3)
        attempts.append(attempt)
        if attempt.get("returncode") == 0:
            break

    return {
        "command": command,
        "ok": attempts[-1].get("returncode") == 0,
        "returncode": attempts[-1].get("returncode"),
        "attempt_count": len(attempts),
        "retries": retries,
        "timeout_seconds": timeout_seconds,
        "attempts": attempts,
    }


def download_manifest(
    manifest: ModelManifest,
    cache_dir: Path,
    execute: bool,
    repo_id_filter: list[str] | None = None,
    check_local: bool = False,
    env_overrides: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    retries: int = 0,
    max_workers: int | None = 1,
) -> dict[str, Any]:
    repo_ids = [
        repo_id
        for repo_id in manifest.hf_repo_ids
        if not repo_id_filter or repo_id in set(repo_id_filter)
    ]
    if repo_id_filter and not repo_ids:
        return {
            "model_id": manifest.model_id,
            "ok": False,
            "error": f"none of the requested repo ids are listed for this model: {', '.join(repo_id_filter)}",
            "requested_repo_ids": repo_id_filter,
            "available_repo_ids": manifest.hf_repo_ids,
            "executed": False,
        }
    if not repo_ids:
        return {
            "model_id": manifest.model_id,
            "ok": False,
            "error": "missing hf_repo_id",
            "executed": False,
        }

    if execute:
        revisions = manifest.hf_repo_revisions
        repo_files = manifest.hf_repo_files
        commands = [
            build_download_command(
                repo_id,
                cache_dir,
                revision=revisions.get(repo_id),
                max_workers=max_workers,
                filenames=repo_files.get(repo_id),
            )
            for repo_id in repo_ids
        ]
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        download_runs = [
            run_download_command(command, env, timeout_seconds=timeout_seconds, retries=retries)
            for command in commands
        ]
        local_checks = [
            check_local_checkpoint(
                repo_id,
                cache_dir,
                manifest.hf_repo_revisions.get(repo_id),
                manifest.hf_repo_files.get(repo_id),
            )
            for repo_id in repo_ids
        ] if check_local else []
        return {
            "model_id": manifest.model_id,
            "hf_repo_ids": repo_ids,
            "commands": commands,
            "executed": True,
            "returncodes": [run.get("returncode") for run in download_runs],
            "download_runs": download_runs,
            "download_options": {
                "timeout_seconds": timeout_seconds,
                "retries": retries,
                "max_workers": max_workers,
                "env": _download_env_summary(env),
            },
            "local_checks": local_checks,
            "ok": all(run.get("ok") is True for run in download_runs)
            and all(check.get("ready") is True for check in local_checks),
        }

    revisions = manifest.hf_repo_revisions
    repo_files = manifest.hf_repo_files
    commands = [
        build_download_command(
            repo_id,
            cache_dir,
            downloader=["hf", "download"],
            revision=revisions.get(repo_id),
            max_workers=max_workers,
            filenames=repo_files.get(repo_id),
        )
        for repo_id in repo_ids
    ]
    local_checks = [
        check_local_checkpoint(
            repo_id,
            cache_dir,
            manifest.hf_repo_revisions.get(repo_id),
            manifest.hf_repo_files.get(repo_id),
        )
        for repo_id in repo_ids
    ] if check_local else []
    return {
        "model_id": manifest.model_id,
        "hf_repo_ids": repo_ids,
        "commands": commands,
        "command": commands[0],
        "executed": False,
        "download_options": {
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "max_workers": max_workers,
        },
        "local_checks": local_checks,
        "ok": all(check.get("ready") is True for check in local_checks) if check_local else True,
    }


def build_report(results: list[dict[str, Any]], *, mode: str, cache_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "worldfoundry-model-zoo-checkpoint-download-report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "check_only": mode == "check",
        "cache_dir": str(cache_dir),
        "results": results,
        "ok": all(result.get("ok") is True for result in results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or execute Hugging Face checkpoint downloads from model-zoo manifests."
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--model-id", help="Download only this model_id.")
    parser.add_argument("--repo-id", action="append", default=None, help="Download/check only this Hugging Face repo id. May repeat.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the Hugging Face CLI. Without this flag, check local cache and print the required commands.",
    )
    parser.add_argument("--disable-xet", action="store_true", help="Set HF_HUB_DISABLE_XET=1 for Hugging Face downloads.")
    parser.add_argument("--disable-hf-transfer", action="store_true", help="Set HF_HUB_ENABLE_HF_TRANSFER=0 for Hugging Face downloads.")
    parser.add_argument("--timeout", type=int, default=None, help="Per-attempt Hugging Face CLI timeout in seconds.")
    parser.add_argument("--retries", type=int, default=0, help="Retry failed or timed-out Hugging Face downloads this many times.")
    parser.add_argument("--max-workers", type=int, default=1, help="Pass --max-workers to hf download. Defaults to 1.")
    parser.add_argument("--check-local", action="store_true", help="Check local HF cache snapshots and incomplete files.")
    parser.add_argument("--allow-all-execute", action="store_true", help="Allow --execute without --model-id or --repo-id.")
    parser.add_argument("--report-path", type=Path, default=None, help="Write a structured JSON download report to this path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.execute and args.model_id is None and args.repo_id is None and not args.allow_all_execute:
            raise ValueError("--execute requires --model-id or --repo-id unless --allow-all-execute is set")
        if args.timeout is not None and args.timeout <= 0:
            raise ValueError("--timeout must be > 0")
        if args.retries < 0:
            raise ValueError("--retries must be >= 0")
        if args.max_workers is not None and args.max_workers <= 0:
            raise ValueError("--max-workers must be > 0")
        manifests = load_manifests(args.manifest_dir, args.model_id)
        env_overrides = {}
        if args.disable_xet:
            env_overrides["HF_HUB_DISABLE_XET"] = "1"
        if args.disable_hf_transfer:
            env_overrides["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        results = [
            download_manifest(
                manifest,
                args.cache_dir,
                args.execute,
                repo_id_filter=args.repo_id,
                check_local=args.check_local or args.execute,
                env_overrides=env_overrides or None,
                timeout_seconds=args.timeout,
                retries=args.retries,
                max_workers=args.max_workers,
            )
            for manifest in manifests
        ]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mode = "execute" if args.execute else "check"
    report = build_report(results, mode=mode, cache_dir=args.cache_dir)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"mode: {mode}")
        for result in results:
            status = "ok" if result["ok"] else "failed"
            print(f"{result['model_id']}: {status}")
            if "commands" in result:
                for command in result["commands"]:
                    print("  " + " ".join(command))
            elif "command" in result:
                print("  " + " ".join(result["command"]))
            if "error" in result:
                print(f"  error: {result['error']}")
            for check in result.get("local_checks", []):
                local_status = "ready" if check.get("ready") else "not-ready"
                print(f"  local {check['repo_id']}: {local_status} {check['cache_repo_dir']}")

    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
