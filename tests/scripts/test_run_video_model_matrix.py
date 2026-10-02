from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "workspace" / "run_video_model_matrix.py"
SPEC = importlib.util.spec_from_file_location("worldfoundry_run_video_model_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_video_model_matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_video_model_matrix
SPEC.loader.exec_module(run_video_model_matrix)


def _model(load_kwargs: dict[str, object]) -> dict[str, object]:
    return {
        "default_variant_id": "default",
        "variants": [
            {
                "variant_id": "default",
                "model_ref": "",
                "checkpoints": [],
                "load_kwargs": load_kwargs,
            }
        ],
    }


def test_readiness_uses_model_refs_declared_in_load_kwargs(tmp_path: Path) -> None:
    model_path = tmp_path / "gamma"
    encoder_path = tmp_path / "reason"
    model_path.mkdir()
    encoder_path.mkdir()

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model(
            {
                "mode": "causal_few_step",
                "model_path": str(model_path),
                "text_encoder_path": str(encoder_path),
                "output_path": str(tmp_path / "not-created.mp4"),
            }
        ),
        tmp_path,
    )

    assert readiness["ready"] is True
    assert readiness["required_refs"] == [str(model_path), str(encoder_path)]
    assert readiness["missing_refs"] == []


def test_readiness_reuses_resolution_and_structure_scans_within_one_audit(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "shared-checkpoint"
    checkpoint.mkdir()
    calls = {"resolve": 0, "structure": 0}
    original_resolve = run_video_model_matrix._resolve_ref

    def counted_resolve(raw_ref: str, checkpoint_root: Path) -> str:
        calls["resolve"] += 1
        return original_resolve(raw_ref, checkpoint_root)

    def counted_structure(resolved_ref: str, checkpoint_root: Path) -> str:
        calls["structure"] += 1
        assert resolved_ref == str(checkpoint.resolve())
        assert checkpoint_root == tmp_path
        return ""

    monkeypatch.setattr(run_video_model_matrix, "_resolve_ref", counted_resolve)
    monkeypatch.setattr(
        run_video_model_matrix,
        "_checkpoint_artifact_structure_error",
        counted_structure,
    )
    resolution_cache: dict[str, str] = {}
    structure_cache: dict[str, str] = {}

    for _ in range(2):
        readiness = run_video_model_matrix._checkpoint_readiness(
            _model({"model_path": str(checkpoint)}),
            tmp_path,
            resolution_cache=resolution_cache,
            structure_cache=structure_cache,
        )
        assert readiness["ready"] is True

    assert calls == {"resolve": 1, "structure": 1}


def test_readiness_requires_every_declared_model_component(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    missing_base_model = tmp_path / "base-model"

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model(
            {
                "required_components": {
                    "checkpoint_path": str(checkpoint),
                    "base_model_root": str(missing_base_model),
                    "runtime_root": str(tmp_path),
                }
            }
        ),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert readiness["missing_refs"] == [str(missing_base_model)]


def test_readiness_treats_flux_path_as_a_required_model_component(tmp_path: Path) -> None:
    missing_flux = tmp_path / "black-forest-labs--FLUX.1-Fill-dev"

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"flux_path": str(missing_flux)}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert readiness["required_refs"] == [str(missing_flux)]
    assert readiness["missing_refs"] == [str(missing_flux)]


def test_readiness_accepts_complete_flux_diffusers_snapshot_with_optional_export_marker(
    tmp_path: Path,
) -> None:
    flux = tmp_path / "black-forest-labs--FLUX.1-Fill-dev"
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/tokenizer_config.json",
        "transformer/config.json",
        "vae/config.json",
    ):
        path = flux / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    optional_export = flux / "flux1-fill-dev.safetensors"
    optional_export.touch()
    Path(f"{optional_export}.aria2").write_bytes(b"in-progress")

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"flux_path": str(flux)}),
        tmp_path,
    )

    assert readiness["ready"] is True
    assert readiness["component_errors"] == []


def test_readiness_rejects_flux_snapshot_missing_directory_component(tmp_path: Path) -> None:
    flux = tmp_path / "black-forest-labs--FLUX.1-Fill-dev"
    flux.mkdir()
    (flux / "model_index.json").write_text("{}", encoding="utf-8")
    optional_export = flux / "flux1-fill-dev.safetensors"
    optional_export.touch()
    Path(f"{optional_export}.aria2").write_bytes(b"in-progress")

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"flux_path": str(flux)}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert readiness["reason"] == "required checkpoint component is structurally incomplete"
    assert "scheduler/scheduler_config.json" in readiness["component_errors"][0]


def test_readiness_rejects_model_specific_checkpoint_architecture_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "epoch-0.safetensors"
    checkpoint.touch()
    model = _model({})
    model["id"] = "echo-memory-spatial"
    model["variants"][0]["model_ref"] = str(checkpoint)
    monkeypatch.setattr(
        run_video_model_matrix,
        "_model_checkpoint_structure_error",
        lambda model_id, resolved_ref: (
            "spatial recipe requires spatial_memory_module weights"
            if model_id == "echo-memory-spatial" and resolved_ref == str(checkpoint)
            else ""
        ),
    )

    readiness = run_video_model_matrix._checkpoint_readiness(model, tmp_path)

    assert readiness["ready"] is False
    assert readiness["reason"] == "required checkpoint component is structurally incomplete"
    assert readiness["missing_refs"] == [str(checkpoint)]
    assert "spatial_memory_module" in readiness["component_errors"][0]


def test_model_checkpoint_inspection_exception_becomes_a_checkpoint_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from worldfoundry.base_models.diffusion_model.models.denoisers import (
        echo_memory_checkpoint,
    )

    class CorruptHeaderError(Exception):
        pass

    checkpoint = tmp_path / "epoch-0.safetensors"
    checkpoint.touch()

    def fail_inspection(*_args, **_kwargs) -> None:
        raise CorruptHeaderError("incomplete metadata, file not fully covered")

    monkeypatch.setattr(
        echo_memory_checkpoint,
        "inspect_echo_checkpoint",
        fail_inspection,
    )

    error = run_video_model_matrix._model_checkpoint_structure_error(
        "echo-memory-context-k20",
        str(checkpoint),
    )

    assert "checkpoint architecture is incompatible" in error
    assert "incomplete metadata" in error


def test_readiness_does_not_treat_runtime_model_id_as_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model(
            {
                "model_id": "rolling-forcing",
                "checkpoint_path": str(checkpoint),
            }
        ),
        tmp_path,
    )

    assert readiness["ready"] is True
    assert readiness["required_refs"] == [str(checkpoint)]


def test_readiness_audits_native_dit_and_codec_overrides(tmp_path: Path) -> None:
    dit = tmp_path / "dit"
    codec = tmp_path / "codec"
    dit.mkdir()
    codec.mkdir()

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model(
            {
                "checkpoint_overrides": {
                    "dit": str(dit),
                    "codec": str(codec),
                }
            }
        ),
        tmp_path,
    )

    assert readiness["ready"] is True
    assert readiness["required_refs"] == [str(dit), str(codec)]


def test_readiness_replaces_symbolic_model_id_checkpoint_with_model_ref(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gen3c"
    checkpoint.mkdir()
    model = _model({})
    model["id"] = "gen3c"
    model["model_ref"] = str(checkpoint)
    model["variants"][0]["checkpoints"] = [
        {"role": "primary", "uri": "gen3c", "required": True}
    ]

    readiness = run_video_model_matrix._checkpoint_readiness(model, tmp_path)

    assert readiness["ready"] is True
    assert readiness["required_refs"] == [str(checkpoint)]
    assert readiness["resolved_model_ref"] == str(checkpoint)


def test_relative_declared_reference_is_resolved_from_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "weights" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    monkeypatch.chdir(tmp_path)

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"checkpoint_path": "weights/model.pt"}),
        tmp_path / "checkpoints",
    )

    assert readiness["ready"] is True
    assert readiness["resolved_refs"] == [str(checkpoint.resolve())]


def test_absolute_legacy_hfd_reference_preserves_nested_component_path(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "publisher--model" / "dit" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    legacy_ref = tmp_path / "legacy" / "ckpt" / "hfd" / "publisher--model" / "dit" / "model.pth"

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"checkpoint_path": str(legacy_ref)}),
        checkpoint_root,
    )

    assert readiness["ready"] is True
    assert readiness["resolved_refs"] == [str(checkpoint.resolve())]


def test_readiness_rejects_structurally_incomplete_gemma_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    upsampler = tmp_path / "upsampler.safetensors"
    upsampler.touch()
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model(
            {
                "required_components": {
                    "checkpoint_path": str(checkpoint),
                    "spatial_upsampler_path": str(upsampler),
                    "gemma_root": str(gemma_root),
                }
            }
        ),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert readiness["reason"] == "required checkpoint component is structurally incomplete"
    assert readiness["missing_refs"] == [str(gemma_root)]
    assert "model-*.safetensors" in readiness["component_errors"][0]
    assert "tokenizer.model" in readiness["component_errors"][0]

    (gemma_root / "model-00001-of-00001.safetensors").touch()
    (gemma_root / "tokenizer.model").touch()
    readiness = run_video_model_matrix._checkpoint_readiness(
        _model(
            {
                "required_components": {
                    "checkpoint_path": str(checkpoint),
                    "spatial_upsampler_path": str(upsampler),
                    "gemma_root": str(gemma_root),
                }
            }
        ),
        tmp_path,
    )

    assert readiness["ready"] is True
    assert readiness["component_errors"] == []


def test_readiness_audits_explicit_runtime_and_python_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model(
            {
                "source_root": str(source),
                "python_executable": str(python),
            }
        ),
        tmp_path,
    )

    assert readiness["ready"] is True
    assert readiness["required_refs"] == [str(source), str(python)]


def test_readiness_requires_materialized_hyworld_scene_layout(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"scene_path": str(scene)}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert readiness["reason"] == "required checkpoint component is structurally incomplete"
    assert "render_results/global_pcd.ply" in readiness["component_errors"][0]

    point_cloud = scene / "render_results" / "global_pcd.ply"
    point_cloud.parent.mkdir()
    point_cloud.touch()
    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"scene_path": str(scene)}),
        tmp_path,
    )
    assert readiness["ready"] is True


def test_readiness_rejects_explicit_sparse_checkpoint_placeholder(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    with checkpoint.open("wb") as handle:
        handle.seek(2 * 1024 * 1024 - 1)
        handle.write(b"\0")

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"checkpoint_path": str(checkpoint)}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert readiness["reason"] == "required checkpoint component is structurally incomplete"
    assert "sparse" in readiness["component_errors"][0]


def test_readiness_accepts_readable_checkpoint_when_filesystem_reports_zero_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"materialized-checkpoint" + b"\1" * (2 * 1024 * 1024))
    original_stat = Path.stat

    def stat_with_zero_blocks(path: Path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == checkpoint:
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_size=result.st_size,
                st_blocks=0,
            )
        return result

    monkeypatch.setattr(Path, "stat", stat_with_zero_blocks)

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"checkpoint_path": str(checkpoint)}),
        tmp_path,
    )

    assert readiness["ready"] is True
    assert readiness["component_errors"] == []


def test_readiness_rejects_sparse_shard_named_by_checkpoint_index(tmp_path: Path) -> None:
    checkpoint = tmp_path / "indexed-model"
    checkpoint.mkdir()
    shard = checkpoint / "model-00001-of-00001.safetensors"
    with shard.open("wb") as handle:
        handle.seek(2 * 1024 * 1024 - 1)
        handle.write(b"\0")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"model_path": str(checkpoint)}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert str(shard) in readiness["component_errors"][0]


def test_readiness_rejects_aria2_control_file_for_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    Path(f"{checkpoint}.aria2").write_bytes(b"in-progress")

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"checkpoint_path": str(checkpoint)}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert ".aria2 exists" in readiness["component_errors"][0]


def test_readiness_rejects_aria2_control_file_for_dreamdojo_distcp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model" / "__0_0.distcp"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint" * (128 * 1024))
    Path(f"{checkpoint}.aria2").write_bytes(b"in-progress")

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"checkpoint_shards": [str(checkpoint)]}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert ".aria2 exists" in readiness["component_errors"][0]


def test_dreamdojo_input_blocker_requires_lerobot_statistics(tmp_path: Path) -> None:
    dataset = tmp_path / "GR1_robot"
    model = _model({"dataset_path": str(dataset)})
    model["id"] = "dreamdojo"
    readiness = {"ready": True}

    blocker = run_video_model_matrix._input_blocker(model, readiness)
    assert "meta/stats.json" in blocker

    stats = dataset / "meta" / "stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}", encoding="utf-8")
    assert run_video_model_matrix._input_blocker(model, readiness) == ""


def test_readiness_ignores_unindexed_optional_sparse_export(tmp_path: Path) -> None:
    checkpoint = tmp_path / "indexed-model"
    checkpoint.mkdir()
    required = checkpoint / "model-00001-of-00001.safetensors"
    required.write_bytes(b"required")
    optional = checkpoint / "optional-export.safetensors"
    with optional.open("wb") as handle:
        handle.seek(2 * 1024 * 1024 - 1)
        handle.write(b"\0")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"model_path": str(checkpoint)}),
        tmp_path,
    )

    assert readiness["ready"] is True


def test_readiness_rejects_unindexed_sparse_numbered_shard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    component = checkpoint / "text_encoder"
    component.mkdir(parents=True)
    (component / "model-00001-of-00002.safetensors").write_bytes(b"first")
    second = component / "model-00002-of-00002.safetensors"
    with second.open("wb") as handle:
        handle.seek(2 * 1024 * 1024 - 1)
        handle.write(b"\0")

    readiness = run_video_model_matrix._checkpoint_readiness(
        _model({"model_path": str(checkpoint)}),
        tmp_path,
    )

    assert readiness["ready"] is False
    assert str(second) in readiness["component_errors"][0]


def test_resolved_checkpoint_invalidates_previous_checkpoint_blocker() -> None:
    assert run_video_model_matrix._can_reuse_previous_row(
        {"status": "blocked_checkpoint", "blocker": "missing yesterday"},
        {"ready": True, "reason": ""},
        retry_this_model=False,
    ) is False


def test_unresolved_checkpoint_preserves_previous_checkpoint_blocker() -> None:
    assert run_video_model_matrix._can_reuse_previous_row(
        {"status": "blocked_checkpoint", "blocker": "missing yesterday"},
        {"ready": False, "reason": "still missing"},
        retry_this_model=False,
    ) is True


def test_current_checkpoint_reason_replaces_stale_reused_blocker(tmp_path: Path) -> None:
    model = _model({"checkpoint_path": str(tmp_path / "missing.pt")})
    readiness = run_video_model_matrix._checkpoint_readiness(model, tmp_path)
    row = {
        "status": "blocked_checkpoint",
        "blocker": "default variant declares no checkpoint or model_ref",
    }
    run_video_model_matrix._refresh_checkpoint_blocker(row, readiness)

    assert row["blocker"] == "required checkpoint path is not locally resolvable"


def test_retry_never_reuses_previous_terminal_row() -> None:
    assert run_video_model_matrix._can_reuse_previous_row(
        {"status": "failed"},
        {"ready": True, "reason": ""},
        retry_this_model=True,
    ) is False


def test_submission_retry_row_is_not_reused_after_runner_restart() -> None:
    assert run_video_model_matrix._can_reuse_previous_row(
        {"status": "submission_retry"},
        {"ready": True, "reason": ""},
        retry_this_model=False,
    ) is False


def test_submission_failure_requeues_until_finite_retry_budget() -> None:
    model = {"id": "temporary-workspace-outage"}
    row: dict[str, object] = {}
    queue: list[dict[str, object]] = []

    retrying = run_video_model_matrix._record_submission_failure(
        queue=queue,
        model=model,
        row=row,
        gpu=2,
        attempt=1,
        max_attempts=3,
        exc=ConnectionError("Workspace is restarting"),
    )

    assert retrying is True
    assert queue == [model]
    assert row["status"] == "submission_retry"
    assert row["submission_attempts"] == 1

    queue.clear()
    retrying = run_video_model_matrix._record_submission_failure(
        queue=queue,
        model=model,
        row=row,
        gpu=2,
        attempt=3,
        max_attempts=3,
        exc=ConnectionError("Workspace is still unavailable"),
    )

    assert retrying is False
    assert queue == []
    assert row["status"] == "submission_failed"
    assert row["submission_attempts"] == 3


def test_video_validation_records_content_hash(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"deterministic-video")
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=0, stdout='{"streams": [{"width": 16, "height": 16}]}', stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_video_model_matrix, "_find_executable", lambda name: name)
    monkeypatch.setattr(run_video_model_matrix.subprocess, "run", fake_run)

    result = run_video_model_matrix._validate_video(video)

    assert result["size_bytes"] == len(b"deterministic-video")
    assert result["sha256"] == hashlib.sha256(b"deterministic-video").hexdigest()
    assert result["full_decode_ok"] is True


def test_completed_job_runs_strict_temporal_validation(tmp_path: Path, monkeypatch) -> None:
    model_id = "strict-model"
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"strict-video-content")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "status": "succeeded",
                "preview_video": str(video),
                "artifacts": [str(video)],
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "manifest": str(manifest),
                    "path": str(video),
                    "ok": True,
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "decoded_frames": 17,
                    "width": 64,
                    "height": 48,
                    "fps": 8.0,
                    "size_bytes": video.stat().st_size,
                    "temporal_change": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_video_model_matrix.subprocess, "run", fake_run)

    validation = run_video_model_matrix._strictly_validate_completed_job(
        {"result": {"manifest_path": str(manifest)}},
        output_dir=tmp_path,
        model_id=model_id,
    )

    assert validation["valid"] is True
    assert validation["videos"][0]["temporal_change"] is True
    assert validation["videos"][0]["full_decode_ok"] is True


def test_gpu_idle_gate_rejects_memory_utilization_and_compute_processes() -> None:
    idle = {0: {"memory_mib": 7, "utilization_percent": 0, "compute_pids": []}}
    assert run_video_model_matrix._gpu_is_idle(
        idle,
        0,
        memory_limit_mib=32,
        utilization_limit_percent=0,
    ) is True

    for state in (
        {"memory_mib": 33, "utilization_percent": 0, "compute_pids": []},
        {"memory_mib": 7, "utilization_percent": 1, "compute_pids": []},
        {"memory_mib": 7, "utilization_percent": 0, "compute_pids": [123]},
    ):
        assert run_video_model_matrix._gpu_is_idle(
            {0: state},
            0,
            memory_limit_mib=32,
            utilization_limit_percent=0,
        ) is False


def test_gpu_idle_gate_can_ignore_only_explicitly_trusted_compute_processes() -> None:
    activity = {
        0: {"memory_mib": 3901, "utilization_percent": 0, "compute_pids": [42064]},
    }

    assert run_video_model_matrix._gpu_is_idle(
        activity,
        0,
        memory_limit_mib=5000,
        utilization_limit_percent=0,
        ignored_compute_pids={42064},
    ) is True
    assert run_video_model_matrix._gpu_is_idle(
        {0: {**activity[0], "compute_pids": [42064, 12345]}},
        0,
        memory_limit_mib=5000,
        utilization_limit_percent=0,
        ignored_compute_pids={42064},
    ) is False


def test_partial_gpu_scheduling_is_the_safe_shared_cluster_default() -> None:
    args = run_video_model_matrix.build_parser().parse_args(
        ["--checkpoint-root", "/checkpoints", "--output-dir", "/results"]
    )

    assert args.require_all_gpus_idle_before_start is False
    assert args.gpu_idle_ignored_compute_pids is None


def test_multi_gpu_models_are_deferred_to_exclusive_gang_phase() -> None:
    assert "4-GPU gang phase" in run_video_model_matrix._multi_gpu_gang_blocker(
        "lingbot-world-v2"
    )
    assert "4-GPU gang phase" in run_video_model_matrix._multi_gpu_gang_blocker(
        "wan2.1-vace"
    )
    assert run_video_model_matrix._multi_gpu_gang_blocker("cosmos-predict2.5") == ""


def test_job_elapsed_seconds_uses_persisted_timestamps() -> None:
    assert run_video_model_matrix._job_elapsed_seconds(
        {
            "started_at": "2026-08-30T12:00:00+00:00",
            "finished_at": "2026-08-30T12:01:02.500000+00:00",
        }
    ) == 62.5


def _write_strict_evidence(
    root: Path,
    *,
    model_id: str,
    report_name: str = "strict_video_validation_batch.jsonl",
) -> tuple[Path, Path, Path]:
    run_dir = root / f"20260830-120000-000000-{model_id}"
    run_dir.mkdir(parents=True)
    video = run_dir / f"{model_id}.mp4"
    video.write_bytes(b"strictly-decoded-video")
    manifest = run_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "status": "succeeded",
                "preview_video": str(video),
                "artifacts": [str(video)],
            }
        ),
        encoding="utf-8",
    )
    strict_report = root / report_name
    strict_report.write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "path": str(video),
                "ok": True,
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "decoded_frames": 81,
                "width": 832,
                "height": 480,
                "fps": 16.0,
                "size_bytes": video.stat().st_size,
                "temporal_change": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, video, strict_report


def test_persistent_strict_validation_recovers_evicted_workspace_job(tmp_path: Path) -> None:
    manifest, video, strict_report = _write_strict_evidence(
        tmp_path,
        model_id="evicted-video-model",
    )

    evidence = run_video_model_matrix._discover_persistent_evidence([tmp_path])

    row = evidence["evicted-video-model"]
    assert row["status"] == "completed"
    assert row["validation"]["valid"] is True
    assert row["validation"]["manifest_path"] == str(manifest)
    assert row["validation"]["videos"][0]["path"] == str(video)
    assert row["validation"]["videos"][0]["full_decode_ok"] is True
    assert row["persistent_evidence"]["source"] == str(strict_report)


def test_persistent_validation_uses_manifest_identity_not_directory_name(tmp_path: Path) -> None:
    _write_strict_evidence(tmp_path, model_id="manifest-model")

    evidence = run_video_model_matrix._discover_persistent_evidence([tmp_path])

    assert set(evidence) == {"manifest-model"}


def test_invalid_persistent_validation_is_not_completion_evidence(tmp_path: Path) -> None:
    _, video, strict_report = _write_strict_evidence(tmp_path, model_id="broken-model")
    video.unlink()

    evidence = run_video_model_matrix._discover_persistent_evidence([tmp_path])

    assert "broken-model" not in evidence
    assert strict_report.is_file()


def test_persistent_validation_rejects_changed_video_content(tmp_path: Path) -> None:
    _, video, _ = _write_strict_evidence(tmp_path, model_id="mutated-model")
    video.write_bytes(b"mutated-after-validation")

    evidence = run_video_model_matrix._discover_persistent_evidence([tmp_path])

    assert "mutated-model" not in evidence


def test_persistent_validation_requires_temporal_change(tmp_path: Path) -> None:
    _, _, strict_report = _write_strict_evidence(tmp_path, model_id="static-model")
    record = json.loads(strict_report.read_text(encoding="utf-8"))
    record["temporal_change"] = False
    strict_report.write_text(json.dumps(record) + "\n", encoding="utf-8")

    evidence = run_video_model_matrix._discover_persistent_evidence([tmp_path])

    assert "static-model" not in evidence


def test_strict_evidence_hash_cache_avoids_duplicate_file_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, _, strict_report = _write_strict_evidence(tmp_path, model_id="cached-model")
    original_sha256_file = run_video_model_matrix._sha256_file
    calls = 0

    def counted_sha256_file(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original_sha256_file(path)

    monkeypatch.setattr(run_video_model_matrix, "_sha256_file", counted_sha256_file)
    hash_cache: dict[tuple[str, int, int], str] = {}

    first = run_video_model_matrix._load_strict_validation_file(
        strict_report,
        hash_cache=hash_cache,
    )
    second = run_video_model_matrix._load_strict_validation_file(
        strict_report,
        hash_cache=hash_cache,
    )

    assert first and second
    assert calls == 1


def test_persistent_evidence_filter_skips_hashing_unselected_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_strict_evidence(tmp_path, model_id="unselected-model")
    calls = 0

    def counted_sha256_file(path: Path) -> str:
        nonlocal calls
        calls += 1
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(run_video_model_matrix, "_sha256_file", counted_sha256_file)

    evidence = run_video_model_matrix._discover_persistent_evidence(
        [tmp_path],
        model_ids={"selected-model"},
    )

    assert evidence == {}
    assert calls == 0


def test_gang_report_supplies_four_gpu_metrics_and_overrides_plain_evidence(
    tmp_path: Path,
) -> None:
    manifest, video, strict_report = _write_strict_evidence(
        tmp_path,
        model_id="four-gpu-model",
    )
    strict_record = json.loads(strict_report.read_text(encoding="utf-8"))
    gang_dir = tmp_path / "gang"
    gang_dir.mkdir()
    gang_report = gang_dir / "gang.json"
    gang_report.write_text(
        json.dumps(
            {
                "gpus": [0, 1, 2, 3],
                "models": [
                    {
                        "model_id": "four-gpu-model",
                        "status": "completed",
                        "job_id": "studio-00099",
                        "elapsed_seconds_observed": 123.5,
                        "peak_memory_mib": 40960,
                        "peak_memory_mib_by_gpu": {
                            "0": 40000,
                            "1": 40960,
                            "2": 40100,
                            "3": 40200,
                        },
                        "strict_validation": [strict_record],
                        "job": {
                            "id": "studio-00099",
                            "model_id": "four-gpu-model",
                            "status": "completed",
                            "result": {
                                "manifest_path": str(manifest),
                                "preview_video": str(video),
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = run_video_model_matrix._discover_persistent_evidence([tmp_path])[
        "four-gpu-model"
    ]

    assert row["gpu"] == "0,1,2,3"
    assert row["gpus"] == [0, 1, 2, 3]
    assert row["peak_memory_mib"] == 40960
    assert row["elapsed_seconds"] == 123.5
    assert row["persistent_evidence"]["source"] == str(gang_report)
