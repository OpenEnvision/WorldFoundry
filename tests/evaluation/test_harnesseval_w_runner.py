import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w import run_harnesseval_w as runner
from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.metrics import METRIC_IDS, normalize_results
from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.io import file_fingerprint
from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.protocols import CORE_SKILLS, FAMILIES
from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.report import OBSERVATION_SKILLS


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def case(family, case_id, core=0.6, observation=0.8, model="model-a"):
    return {
        "schema_version": "harnesseval.skill_eval",
        "model_id": model,
        "case_id": case_id,
        "taxonomy": {"probe_family": family},
        "case_score": {"score": core},
        "observation_quality": {"dimensions": {skill: observation for skill in OBSERVATION_SKILLS}},
    }


def test_macro_weights_families_equally(tmp_path):
    records = [case(f, str(i), core=0, observation=0) for i, f in enumerate(FAMILIES)]
    records += [case(FAMILIES[0], f"extra-{i}", core=1, observation=1) for i in range(9)]
    result = normalize_results(write(tmp_path / "scores.json", records))
    assert result["scores"]["overall_macro"] == 0.15
    assert result["case_count"] == 15


def test_common_cases_preserved_when_selecting_model(tmp_path):
    records = [case(f, f, model=m) for f in FAMILIES for m in ["model-a", "model-b"]]
    records += [case(FAMILIES[0], "only-a", core=1, observation=1)]
    path = write(tmp_path / "scores.json", records)
    with pytest.raises(ValueError, match="multiple models"):
        normalize_results(path)
    result = normalize_results(path, "model-a")
    assert result["scores"]["overall_macro"] == 0.7
    assert result["common_case_counts"][FAMILIES[0]] == 1
    assert result["case_count"] == 7


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1, 1.1, "0.7"])
def test_invalid_scores_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        normalize_results(write(tmp_path / "bad.json", [case(FAMILIES[0], "a", core=bad)]))


def test_duplicate_case_rejected(tmp_path):
    row = case(FAMILIES[0], "a")
    with pytest.raises(ValueError, match="duplicate"):
        normalize_results(write(tmp_path / "bad.json", [row, row]))


def test_partial_does_not_invent_overall_and_strict_fails(tmp_path):
    path = write(tmp_path / "scores.json", [case(FAMILIES[0], "a")])
    assert normalize_results(path)["scores"]["overall_macro"] is None
    assert runner.main(["--official-results-path", str(path), "--strict", "--output-dir", str(tmp_path / "out")]) == 1
    result = json.loads((tmp_path / "out/scorecard.json").read_text())
    assert result["run"]["status"] == "failed"
    assert not result["normalization_ok"]


def test_empty_and_unrecognized_inputs_fail(tmp_path):
    for payload in ([], {}, {"score": 0.8}):
        with pytest.raises(ValueError):
            normalize_results(write(tmp_path / "bad.json", payload))


def test_report_rejects_tampered_macro_and_coverage(tmp_path):
    path = write(tmp_path / "cases.json", [case(f, f) for f in FAMILIES])
    report = normalize_results(path)["report"]
    for mutate in [
        lambda r: r["leaderboard"][0].update(overall_macro=0.5),
        lambda r: r["common_case_counts"].update(exploratory_transition=True),
        lambda r: r["leaderboard"][0]["coverage"].update(exploratory_transition=0),
        lambda r: r.update(scoring_policy={}),
    ]:
        bad = copy.deepcopy(report)
        mutate(bad)
        with pytest.raises(ValueError):
            normalize_results(write(tmp_path / "bad.json", bad))


def test_fixture_evidence_and_contract_discovery(tmp_path):
    from worldfoundry.evaluation.tasks.catalog.dispatch import official_runner_spec
    from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract

    assert set(get_external_benchmark_contract("harnesseval-w").metric_ids) == set(METRIC_IDS)
    assert official_runner_spec("harnesseval-w").module == runner.MODULE
    assert runner.main(["--run-fixture", "--strict", "--output-dir", str(tmp_path)]) == 0
    card = json.loads((tmp_path / "scorecard.json").read_text())
    assert card["normalization_ok"]
    assert card["metrics"]["leaderboard"]["overall_macro"] == 0.7
    assert not any(card[k] for k in ("leaderboard_valid", "integration_evidence", "official_benchmark_verified"))
    assert len((tmp_path / "per_case_metrics.jsonl").read_text().splitlines()) == 6


@pytest.fixture
def cached_run(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required to exercise official video input validation")
    video = tmp_path / "template.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:r=8",
            "-t",
            "1",
            "-c:v",
            "mpeg4",
            "-y",
            str(video),
        ],
        check=True,
        timeout=30,
    )
    image = tmp_path / "initial.ppm"
    image.write_bytes(b"P6\n1 1\n255\n\x00\x00\xff")
    generation, plans, cache = (tmp_path / name for name in ("generation", "plans", "cache"))
    cases = []
    for index, family in enumerate(FAMILIES):
        cid = f"case-{index}"
        taxonomy = {
            "primary_axis": "transition_correctness" if index < 3 else "world_persistence",
            "probe_family": family,
        }
        cases.append({"case_id": cid, "taxonomy": taxonomy, "world": {"initial_observation": str(image)}})
        rollout = generation / "outputs" / taxonomy["primary_axis"] / family / "model-a" / cid
        rollout.mkdir(parents=True)
        shutil.copyfile(video, rollout / "output.mp4")
        write(
            rollout / "metadata.json",
            {"case_id": cid, "model_id": "model-a", "taxonomy": taxonomy, "output_video": "output.mp4"},
        )
        selections = [{"skill_id": skill, "role": "observation"} for skill in OBSERVATION_SKILLS]
        selections += [{"skill_id": CORE_SKILLS[family][0], "role": "core"}]
        write(
            plans / family / f"{cid}.skill_plan.json",
            {
                "schema_version": "harnesseval.skill_plan",
                "case_id": cid,
                "taxonomy": taxonomy,
                "selected_skills": selections,
                "validation": {"status": "ok", "selection_modified": False},
            },
        )
        write(
            cache / "bundles/model-a" / family / f"{cid}.metrics.json",
            {
                "schema_version": "harnesseval.metric_bundle",
                "model_id": "model-a",
                "case_id": cid,
                "probe_family": family,
                "video": file_fingerprint(rollout / "output.mp4"),
                "skill_results": [
                    {"skill_id": s["skill_id"], "score": 0.6 if s["role"] == "core" else 0.8, "status": "ok"}
                    for s in selections
                ],
            },
        )
    manifest = write(generation / "manifest.json", {"cases": cases})
    args = [
        "--run-official",
        "--generated-artifact-dir",
        str(generation),
        "--manifest",
        str(manifest),
        "--plan-root",
        str(plans),
        "--metric-cache-root",
        str(cache),
        "--model-id",
        "model-a",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    return args, tmp_path / "out", cache


def test_official_cached_scoring_and_resume(cached_run):
    args, output, _ = cached_run
    assert runner.main([*args, "--score-cached", "--strict"]) == 0
    card = json.loads((output / "scorecard.json").read_text())
    assert card["metrics"]["leaderboard"]["overall_macro"] == 0.7
    assert not card["leaderboard_valid"]
    assert json.loads((output / "completion_audit.json").read_text())["status"] == "passed"
    assert runner.main([*args, "--score-cached", "--strict"]) == 0
    execution = json.loads((output / "score_execution.json").read_text())
    assert execution["written"] == 0 and execution["cached"] == 6


def test_dry_run_and_case_limit_keep_complete_skills(cached_run):
    args, output, _ = cached_run
    assert runner.main([*args, "--dry-run", "--limit", "1"]) == 0
    card = json.loads((output / "scorecard.json").read_text())
    assert not card["normalization_ok"] and not card["evaluation"]["scored"]
    assert runner.main([*args, "--score-cached", "--limit", "1"]) == 0
    assert "overall_macro" not in json.loads((output / "scorecard.json").read_text())["metrics"]["leaderboard"]
    case_rows = (output / "per_case_metrics.jsonl").read_text().splitlines()
    assert len(case_rows) == 1 and len(json.loads(case_rows[0])["skills"]) == 5


def test_missing_skill_bundle_blocks_scoring(cached_run):
    args, output, cache = cached_run
    path = next(cache.rglob("*.metrics.json"))
    payload = json.loads(path.read_text())
    payload["skill_results"].pop()
    write(path, payload)
    assert runner.main([*args, "--score-cached"]) == 1
    assert "blocked" in json.loads((output / "scorecard.json").read_text())["run"]["error"]


def test_python_skill_workers_and_drift_stages(cached_run, monkeypatch):
    """Orchestration keeps complete cases and isolates staged drift model lifetimes."""
    import worldfoundry.runtime.jobs as jobs

    args, output, _ = cached_run
    config = write(
        output.parent / "backends.json", {"skills": {"drift_degradation_analyzer": {"mode": "staged_local"}}}
    )
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 37
        assert command[1:3] == ["-m", runner.MODULE]
        # Pre-existing fixture bundles stand in for expensive model outputs.
        return {"returncode": 0, "stdout": "completed", "stderr": ""}

    monkeypatch.setattr(jobs, "run_bounded_command", execute)
    assert runner.main([*args, "--backend-config", str(config), "--timeout", "37"]) == 0
    stages = [c[c.index("--drift-stage") + 1] for c in commands if "--drift-stage" in c]
    assert stages == ["physical", "render", "motion", "clip"]
    assert len(commands) == 14  # Four observations + six family core skills + four drift stages.
    assert len(list((output / "logs").glob("*.log"))) == 14


def test_workspace_forwards_selected_model_and_manifest(tmp_path):
    from worldfoundry.evaluation.tasks.catalog.workspace_registry import build_workspace_benchmark_command

    command = build_workspace_benchmark_command(
        {
            "benchmark_id": "harnesseval-w",
            "dataset_manifest": "cases.json",
            "params": {"generated_artifact_dir": "videos", "result_model_id": "model-a", "run_official": True},
        },
        tmp_path,
    )
    assert command[command.index("--model-id") + 1] == "model-a"
    assert command[command.index("--manifest") + 1] == "cases.json"
    assert "--run-official" in command


def test_motion_backend_reuses_metrics_across_videos(cached_run, monkeypatch):
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.metrics import video_quality
    from worldfoundry.evaluation.tasks.execution.runners.harnesseval_w.runtime.skill_backend.motion_quality import (
        LocalBackend,
    )

    args, output, _ = cached_run
    monkeypatch.setenv("HARNESSEVAL_WEIGHTS_ROOT", str(output.parent))
    loaded = []

    def metric_type(name):
        class Metric:
            def __init__(self, device):
                loaded.append((name, device))

            def compute(self, frames):
                assert frames
                return {name + "_score": 0.75}

        return Metric

    for name in ("dynamic_degree", "motion_smoothness"):
        metric = metric_type(name)
        monkeypatch.setattr(video_quality, f"get_{name}_metric", lambda metric=metric: metric)
    generation = Path(args[args.index("--generated-artifact-dir") + 1])
    videos = list(generation.rglob("output.mp4"))[:2]
    backend = LocalBackend(output.parent, device="cpu", cpu_workers=1)
    try:
        for video in videos:
            result = backend.evaluate(video, file_fingerprint(video))
            assert result["decoded_frames"] == 8
            assert all(value["raw_score"] == 0.75 for value in result["metrics"].values())
    finally:
        backend.close()
    assert backend.load_count == 1
    assert loaded == [("dynamic_degree", "cpu"), ("motion_smoothness", "cpu")]


def test_megasam_reuses_registered_unidepth_loader(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from worldfoundry.base_models.three_dimensions.slam import megasam_resident as resident

    model_dir = tmp_path / "registered"
    model_dir.mkdir()
    (model_dir / "model.safetensors").touch()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.delenv("HARNESSEVAL_UNIDEPTH_WEIGHTS", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR", str(model_dir))
    calls = []

    class Network:
        def to(self, device):
            calls.append(device)
            return self

        def eval(self):
            return self

    network = Network()

    def load_prior(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=network)

    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.base_models.three_dimensions.depth.unidepth",
        SimpleNamespace(UniDepth2Model=load_prior),
    )
    assert resident.load_unidepth_model(tmp_path / "legacy", "cuda:1") is network
    assert calls == [{"type": "l", "model_path": str(model_dir), "device": "cuda:1"}]


def test_megasam_explicit_unidepth_weights_do_not_silently_fallback(tmp_path, monkeypatch):
    from worldfoundry.base_models.three_dimensions.slam import megasam_resident as resident

    explicit = tmp_path / "model.safetensors"
    monkeypatch.setenv("HARNESSEVAL_UNIDEPTH_WEIGHTS", str(explicit))
    with pytest.raises(FileNotFoundError, match="HARNESSEVAL_UNIDEPTH_WEIGHTS"):
        resident.find_unidepth_weights(tmp_path)
    explicit.touch()
    assert resident.find_unidepth_weights(tmp_path) == explicit


def test_megasam_missing_registered_unidepth_weights_do_not_use_legacy(tmp_path, monkeypatch):
    from worldfoundry.base_models.three_dimensions.slam import megasam_resident as resident

    legacy = (
        tmp_path / "huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/snapshots"
        / resident.UNIDEPTH_REVISION / "model.safetensors"
    )
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    monkeypatch.delenv("HARNESSEVAL_UNIDEPTH_WEIGHTS", raising=False)
    monkeypatch.setenv("WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="Registered UniDepth weights missing"):
        resident.find_unidepth_weights(tmp_path)
    monkeypatch.delenv("WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR")
    monkeypatch.setitem(
        resident.BASE_MODEL_CAPABILITIES,
        "unidepth_v2_vitl14",
        type("Capability", (), {"assets": [type("Asset", (), {"check": lambda self: {
            "matched_path": None, "local_path": str(tmp_path / "missing")
        }})()]})(),
    )
    assert resident.find_unidepth_weights(tmp_path) == legacy


def test_unidepth_checkpoint_register_does_not_become_an_image_token():
    import torch

    from worldfoundry.base_models.three_dimensions.depth.unidepth.models.encoder import UniDepthEncoder

    encoder = UniDepthEncoder({"use_norm": True}, embed_dim=32, depth=4, num_heads=4, output_idx=[1, 2, 3, 4]).eval()
    image = torch.rand(1, 3, 28, 42)
    with torch.no_grad():
        features, tokens = encoder(image)
        encoder.register_tokens.fill_(1000)
        repeated, _ = encoder(image)
    assert len(features) == len(tokens) == 4
    assert all(value.shape == (1, 2, 3, 32) for value in features)
    assert all(value.shape == (1, 1, 32) for value in tokens)
    for before, after in zip(features, repeated):
        torch.testing.assert_close(before, after)


def test_unidepth_loader_rejects_incompatible_weights(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from worldfoundry.base_models.three_dimensions.depth import unidepth

    (tmp_path / "config.json").write_text("{}")
    save_file({"weight": torch.ones(3, 2), "bias": torch.ones(3)}, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(unidepth, "UniDepthV2", lambda config: torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="size mismatch"):
        unidepth.load_local_unidepth_v2(tmp_path)


def test_megasam_single_video_uses_shared_python_worker(tmp_path, monkeypatch):
    from worldfoundry.base_models.three_dimensions.slam import megasam

    invocations = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(megasam.subprocess, "run", lambda command, **kwargs: invocations.append((command, kwargs)))
    megasam.run_single(tmp_path / "input.mp4", tmp_path / "poses.npz", device="2")
    command, kwargs = invocations[0]
    assert command[1:3] == ["-m", "worldfoundry.base_models.three_dimensions.slam.megasam"]
    assert "--worker" in command
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert kwargs["check"] is True


def test_megasam_tracking_rejects_misaligned_frames(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from worldfoundry.base_models.three_dimensions.slam.mega_sam_runtime.camera_tracking import run_tracking

    monkeypatch.setitem(sys.modules, "lietorch", SimpleNamespace(SE3=None))
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("00000", "00001"):
        (frames / f"{name}.jpg").touch()
    with pytest.raises(ValueError, match="frame names must match"):
        run_tracking(frames, tmp_path / "mono", tmp_path / "metric", "scene", tmp_path / "out.npz", droid_factory=None)


def test_megasam_tracking_writes_only_camera_output(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import cv2
    import numpy as np
    import torch

    from worldfoundry.base_models.three_dimensions.slam.mega_sam_runtime.camera_tracking import run_tracking

    class SE3:
        def __init__(self, trajectory):
            self.trajectory = trajectory

        def inv(self):
            return self

        def matrix(self):
            return torch.eye(4).repeat(len(self.trajectory), 1, 1)

    monkeypatch.setitem(sys.modules, "lietorch", SimpleNamespace(SE3=SE3))
    frames = tmp_path / "frames"
    frames.mkdir()
    mono = tmp_path / "mono/scene"
    metric = tmp_path / "metric/scene"
    mono.mkdir(parents=True)
    metric.mkdir(parents=True)
    disparity = np.linspace(0.1, 0.5, 48 * 64, dtype=np.float32).reshape(48, 64)
    for i in range(3):
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), np.full((48, 64, 3), i, np.uint8))
        np.save(mono / f"{i:05d}.npy", disparity)
        np.savez(metric / f"{i:05d}.npz", depth=1 / (2 * disparity + 0.1), fov=60.0)
    seen = []

    class Droid:
        video = SimpleNamespace(intrinsics=torch.tensor([[20., 20., 16., 12.]]))

        def track(self, timestamp, image, depth, **kwargs):
            seen.append(timestamp)
            assert depth.isfinite().all()

        def track_final(self, *args, **kwargs):
            pass

        def terminate(self, stream, **kwargs):
            assert len(list(stream)) == 3
            return np.zeros((3, 7)), None, None

    output = run_tracking(frames, mono.parent, metric.parent, "scene", tmp_path / "poses.npz", droid_factory=lambda args: Droid())
    with np.load(output) as data:
        assert set(data.files) == {"cam_c2w", "intrinsic"}
        assert data["cam_c2w"].shape == (3, 4, 4)
        assert data["intrinsic"][0, 0] == 160
    assert seen == [0, 1, 2]
    assert not (tmp_path / "reconstructions").exists()
