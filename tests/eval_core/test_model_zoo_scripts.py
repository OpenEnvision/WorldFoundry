from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / "model_zoo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(manifest_dir: Path, payload: object) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "models.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_download_checkpoints_plan_only_filters_model_and_never_calls_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir,
        {
            "models": [
                {"model_id": "alpha", "checkpoint_refs": [{"repo_id": "org/alpha"}]},
                {"model_id": "beta", "checkpoint_refs": [{"repo_id": "org/beta"}]},
            ]
        },
    )

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("plan-only must not execute a download command")

    monkeypatch.setattr(download_checkpoints.subprocess, "run", fail_run)

    manifests = download_checkpoints.load_manifests(manifest_dir, model_id="beta")
    results = [
        download_checkpoints.download_manifest(manifest, tmp_path / "cache" / "hfd", execute=False)
        for manifest in manifests
    ]

    assert [result["model_id"] for result in results] == ["beta"]
    assert results[0]["executed"] is False
    assert results[0]["command"] == [
        "hf",
        "download",
        "org/beta",
        "--cache-dir",
        str(tmp_path / "cache" / "hfd"),
        "--max-workers",
        "1",
    ]


def test_download_checkpoints_loads_nested_catalog_manifests(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest_dir = tmp_path / "model_zoo"
    _write_manifest(
        manifest_dir / "world_models",
        {
            "model_id": "matrix-game-2",
            "checkpoint": {
                "repos": [
                    {
                        "id": "Skywork/Matrix-Game-2.0",
                        "sha": "f1729d99a80e0f07993a77d7dad4a3190e23c2c8",
                    }
                ]
            },
        },
    )

    manifests = download_checkpoints.load_manifests(manifest_dir, model_id="matrix-game-2")

    assert len(manifests) == 1
    assert manifests[0].hf_repo_ids == ["Skywork/Matrix-Game-2.0"]
    assert manifests[0].hf_repo_revisions == {
        "Skywork/Matrix-Game-2.0": "f1729d99a80e0f07993a77d7dad4a3190e23c2c8"
    }


def test_download_checkpoints_execute_calls_selected_hf_downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="alpha",
        path=tmp_path / "alpha.yaml",
        data={"checkpoint_refs": [{"repo_id": "org/alpha"}]},
    )
    calls: list[list[str]] = []

    def fake_build_download_command(
        repo_id: str,
        cache_dir: Path,
        revision: str | None = None,
        max_workers: int | None = None,
        filenames: list[str] | None = None,
    ) -> list[str]:
        assert revision is None
        assert max_workers == 1
        assert filenames is None
        return ["huggingface-cli", "download", repo_id, "--cache-dir", str(cache_dir)]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(download_checkpoints, "build_download_command", fake_build_download_command)
    monkeypatch.setattr(download_checkpoints.subprocess, "run", fake_run)

    result = download_checkpoints.download_manifest(manifest, tmp_path / "cache" / "hfd", execute=True)

    assert result["ok"] is True
    assert calls == [["huggingface-cli", "download", "org/alpha", "--cache-dir", str(tmp_path / "cache" / "hfd")]]


def test_download_checkpoints_repo_id_filter_limits_multi_checkpoint_manifest(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="multi",
        path=tmp_path / "multi.yaml",
        data={"checkpoint_refs": [{"repo_id": "org/a"}, {"repo_id": "org/b"}]},
    )

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
        repo_id_filter=["org/b"],
    )

    assert result["ok"] is True
    assert result["hf_repo_ids"] == ["org/b"]
    assert result["commands"] == [
        ["hf", "download", "org/b", "--cache-dir", str(tmp_path / "cache" / "hfd"), "--max-workers", "1"]
    ]


def test_download_checkpoints_plan_only_includes_manifest_revision(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="revisioned",
        path=tmp_path / "revisioned.yaml",
        data={"checkpoint_refs": [{"repo_id": "org/model", "sha": "abc1234"}]},
    )

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["commands"] == [
        [
            "hf",
            "download",
            "org/model",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "abc1234",
            "--max-workers",
            "1",
        ]
    ]


def test_download_checkpoints_reads_variant_checkpoint_refs(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="variant-model",
        path=tmp_path / "variant-model.yaml",
        data={
            "variants": [
                {
                    "id": "small",
                    "checkpoint_refs": [{"repo_id": "org/model-small", "revision": "abc1234"}],
                },
                {
                    "id": "large",
                    "checkpoint_refs": [{"repo_id": "org/model-large", "sha": "def5678"}],
                },
            ]
        },
    )

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=False,
    )

    assert result["hf_repo_ids"] == ["org/model-small", "org/model-large"]
    assert result["commands"] == [
        [
            "hf",
            "download",
            "org/model-small",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "abc1234",
            "--max-workers",
            "1",
        ],
        [
            "hf",
            "download",
            "org/model-large",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--revision",
            "def5678",
            "--max-workers",
            "1",
        ],
    ]


def test_download_checkpoints_execute_retries_and_preserves_proxy_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest = download_checkpoints.ModelManifest(
        model_id="alpha",
        path=tmp_path / "alpha.json",
        data={"hf_repo_id": "org/alpha"},
    )
    calls: list[tuple[list[str], dict[str, str], int | None]] = []

    def fake_find_hf_downloader() -> list[str]:
        return ["hf", "download"]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        calls.append((command, env, kwargs.get("timeout")))
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(download_checkpoints, "find_hf_downloader", fake_find_hf_downloader)
    monkeypatch.setattr(download_checkpoints.subprocess, "run", fake_run)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")

    result = download_checkpoints.download_manifest(
        manifest,
        tmp_path / "cache" / "hfd",
        execute=True,
        check_local=False,
        env_overrides={"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"},
        timeout_seconds=123,
        retries=1,
        max_workers=1,
    )

    assert result["ok"] is True
    assert len(calls) == 2
    assert calls[0][0] == [
        "hf",
        "download",
        "org/alpha",
        "--cache-dir",
        str(tmp_path / "cache" / "hfd"),
        "--max-workers",
        "1",
    ]
    assert calls[0][1]["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert calls[0][1]["HF_HUB_DISABLE_XET"] == "1"
    assert calls[0][1]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert calls[0][2] == 123
    assert result["download_runs"][0]["attempt_count"] == 2
    assert result["download_options"]["env"]["proxy_env_keys"]


def test_download_checkpoints_execute_records_timeout_without_deleting_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    incomplete = cache_dir / "models--org--alpha" / "blobs" / "part.incomplete"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text("partial", encoding="utf-8")
    manifest = download_checkpoints.ModelManifest(
        model_id="alpha",
        path=tmp_path / "alpha.json",
        data={"hf_repo_id": "org/alpha"},
    )

    def fake_find_hf_downloader() -> list[str]:
        return ["hf", "download"]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(download_checkpoints, "find_hf_downloader", fake_find_hf_downloader)
    monkeypatch.setattr(download_checkpoints.subprocess, "run", fake_run)

    result = download_checkpoints.download_manifest(
        manifest,
        cache_dir,
        execute=True,
        check_local=True,
        timeout_seconds=5,
        retries=0,
    )

    assert result["ok"] is False
    assert result["download_runs"][0]["attempts"][0]["timed_out"] is True
    assert result["local_checks"][0]["incomplete_files"][0]["path"].endswith("part.incomplete")
    assert incomplete.exists()


def test_download_checkpoints_main_writes_structured_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    manifest_dir = tmp_path / "model_zoo"
    report_path = tmp_path / "report.json"
    _write_manifest(manifest_dir, {"model_id": "alpha", "hf_repo_id": "org/alpha"})

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("check mode must not execute a download command")

    monkeypatch.setattr(download_checkpoints.subprocess, "run", fail_run)

    exit_code = download_checkpoints.main(
        [
            "--manifest-dir",
            str(manifest_dir),
            "--model-id",
            "alpha",
            "--cache-dir",
            str(tmp_path / "cache" / "hfd"),
            "--report-path",
            str(report_path),
            "--json",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "worldfoundry-model-zoo-checkpoint-download-report"
    assert report["mode"] == "check"
    assert report["check_only"] is True
    assert report["ok"] is True
    assert report["results"][0]["download_options"]["max_workers"] == 1


def test_download_checkpoints_local_check_detects_ready_snapshot(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    blob = repo_dir / "blobs" / "blob-a"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    blob.write_text("weights", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/blob-a")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir)

    assert result["ready"] is True
    assert result["file_count"] == 1
    assert result["incomplete_files"] == []


def test_download_checkpoints_local_check_accepts_symlinked_hf_repo_root(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    shared_repo_dir = tmp_path / "shared" / "models--org--model"
    revision = "abc123def456"
    snapshot = shared_repo_dir / "snapshots" / revision
    blob = shared_repo_dir / "blobs" / "blob-a"
    (shared_repo_dir / "refs").mkdir(parents=True)
    (shared_repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (shared_repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    blob.write_text("weights", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/blob-a")
    cache_dir.mkdir(parents=True)
    (cache_dir / "models--org--model").symlink_to(shared_repo_dir, target_is_directory=True)

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision=revision)

    assert result["ready"] is True
    assert result["local_layout"] == "hf_cache"
    assert result["file_count"] == 1
    assert result["broken_links"] == []


def test_download_checkpoints_local_check_detects_incomplete_file(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (repo_dir / "blobs" / "blob-a.incomplete").write_text("partial", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir)

    assert result["ready"] is False
    assert result["incomplete_files"][0]["path"].endswith("blob-a.incomplete")
    assert result["blocking_incomplete_files"][0]["path"].endswith("blob-a.incomplete")


def test_download_checkpoints_local_check_allows_orphan_incomplete_file(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    blob = repo_dir / "blobs" / "blob-a"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    blob.write_text("weights", encoding="utf-8")
    (repo_dir / "blobs" / "orphan.incomplete").write_text("partial", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/blob-a")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir)

    assert result["ready"] is True
    assert result["blocking_incomplete_files"] == []
    assert result["orphan_incomplete_files"][0]["path"].endswith("orphan.incomplete")


def test_download_checkpoints_local_check_requires_expected_revision(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "cache" / "hfd"
    repo_dir = cache_dir / "models--org--model"
    snapshot = repo_dir / "snapshots" / "abc123"
    (repo_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("weights", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="def456")

    assert result["ready"] is False
    assert result["expected_revision"] == "def456"
    assert result["revision_matches"] is False


def test_download_checkpoints_local_check_accepts_direct_hfd_layout(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "hfd"
    repo_dir = cache_dir / "org--model"
    (repo_dir / ".hfd").mkdir(parents=True)
    (repo_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456"}),
        encoding="utf-8",
    )
    (repo_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is True
    assert result["local_layout"] == "direct_hfd"
    assert result["direct_hfd_file_count"] == 1
    assert result["direct_hfd_revision"] == "abc123def456"
    assert result["direct_hfd_revision_matches"] is True


def test_download_checkpoints_local_check_accepts_snapshot_download_local_dir(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "hfd"
    repo_dir = cache_dir / "org--model"
    metadata_dir = repo_dir / ".cache" / "huggingface" / "download" / "subdir"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "model.safetensors.metadata").write_text(
        "abc123def456\netag\n123.0\n",
        encoding="utf-8",
    )
    (repo_dir / "subdir").mkdir()
    (repo_dir / "subdir" / "model.safetensors").write_text("weights", encoding="utf-8")

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is True
    assert result["local_layout"] == "direct_hfd"
    assert result["direct_hfd_file_count"] == 1
    assert result["direct_hfd_revision"] == "abc123def456"


def test_download_checkpoints_local_check_rejects_direct_hfd_metadata_only(tmp_path: Path) -> None:
    download_checkpoints = _load_script("download_checkpoints")
    cache_dir = tmp_path / "hfd"
    repo_dir = cache_dir / "org--model"
    (repo_dir / ".hfd").mkdir(parents=True)
    (repo_dir / ".hfd" / "repo_metadata.json").write_text(
        json.dumps({"sha": "abc123def456"}),
        encoding="utf-8",
    )

    result = download_checkpoints.check_local_checkpoint("org/model", cache_dir, expected_revision="abc123def456")

    assert result["ready"] is False
    assert result["local_layout"] == "missing"
    assert result["direct_hfd_file_count"] == 0


def test_model_zoo_scripts_are_stdlib_only() -> None:
    allowed_modules = set(sys.stdlib_module_names) | {"__future__", "worldfoundry", "yaml"}
    for path in (REPO_ROOT / "scripts" / "model_zoo").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                modules = {node.module.split(".", 1)[0]} if node.module else set()
            else:
                continue

            unexpected = modules - allowed_modules
            assert unexpected == set(), f"{path} imports {unexpected}"
