from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "fastapi" dependency at import time; skip when it is unavailable.
pytest.importorskip("fastapi")

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient

import worldfoundry.studio.inference.requests as inference_requests
import worldfoundry.studio.workspace.app as workspace_app
from worldfoundry.studio.workspace.app import WORKSPACE_HTML, create_app
from worldfoundry.studio.workspace.jobs import StudioJobStore


@pytest.fixture(autouse=True)
def isolated_jobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    store = StudioJobStore(max_workers=1)
    monkeypatch.setattr(workspace_app, "JOBS", store)
    try:
        yield
    finally:
        store.shutdown(grace_seconds=1)


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(300):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"workspace job did not finish: {job_id}")


def test_workspace_sidebar_exposes_core_job_surfaces() -> None:
    assert 'data-view="inference">Inference' in WORKSPACE_HTML
    assert 'data-view="evaluation">Evaluation' in WORKSPACE_HTML
    assert 'new Set(["inference", "evaluation"])' in WORKSPACE_HTML
    assert 'id="inferDynamicFields"' in WORKSPACE_HTML
    assert "INFER_TASK_FIELD_CONTROLS" in WORKSPACE_HTML
    assert 'id="visualizerGrid"' in WORKSPACE_HTML
    assert 'id="artifactList"' in WORKSPACE_HTML
    assert 'id="evalPreset"' in WORKSPACE_HTML
    assert 'id="evalDatasetRoot"' in WORKSPACE_HTML
    assert 'id="evalDatasetManifest"' in WORKSPACE_HTML
    assert "function artifactCard(row)" in WORKSPACE_HTML
    assert "function renderEvaluationSummary(result)" in WORKSPACE_HTML
    assert "function applyEvaluationPreset()" in WORKSPACE_HTML


def test_workspace_index_exposes_evaluation_cache_controls() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/")
        settings = client.get("/api/settings")

    assert response.status_code == 200
    assert 'id="evalCacheMode"' in response.text
    assert "generation_cache_mode:" in response.text
    assert settings.status_code == 200
    assert {"device", "fps", "num_frames"} <= settings.json().keys()


@pytest.mark.parametrize("origin", ["https://example.invalid", "http://127.0.0.1:8000"])
def test_workspace_api_does_not_grant_cross_origin_access(origin: str) -> None:
    with TestClient(create_app()) as client:
        response = client.options(
            "/api/evaluation/prepare",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )

    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("transport", ["bearer", "query"])
def test_workspace_token_protects_api_routes(transport: str) -> None:
    credentials = (
        {"headers": {"Authorization": "Bearer secret-token"}}
        if transport == "bearer"
        else {"params": {"token": "secret-token"}}
    )
    with TestClient(create_app(auth_token="secret-token")) as client:
        assert client.get("/api/settings").status_code == 401
        assert client.post("/api/jobs/unknown/stop").status_code == 401
        assert client.get("/api/settings", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/", params={"token": "secret-token"}).status_code == 200
        assert client.get("/api/settings").status_code == 401
        assert client.get("/api/settings", **credentials).status_code == 200
        allowed = client.post("/api/jobs/unknown/stop", **credentials)

    assert allowed.status_code == 200
    assert allowed.json()["ok"] is False


def test_workspace_refuses_remote_bind_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(workspace_app, "create_app", lambda **kwargs: pytest.fail("server must not start"))  # ty: ignore[invalid-argument-type]  # ty 0.0.99 misreads pytest.fail's signature

    with pytest.raises(SystemExit, match="non-loopback host .*without authentication"):
        workspace_app.main(["--host", "0.0.0.0", "--port", "9999"])


def test_workspace_evaluation_prepare_reports_artifact_readiness(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")
    payload = {
        "job_type": "evaluation",
        "eval_mode": "score-artifacts",
        "benchmark_id": "vbench",
        "dataset_root": str(artifacts),
        "output_dir": str(tmp_path / "output"),
    }
    with TestClient(create_app()) as client:
        ready = client.post("/api/evaluation/prepare", json=payload)
        missing = client.post("/api/evaluation/prepare", json={**payload, "dataset_root": str(tmp_path / "missing")})
        jobs = client.get("/api/jobs")

    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["classification"] == "custom_dataset_metric_evaluation"
    assert ready.json()["leaderboard_candidate"] is False
    assert missing.status_code == 200
    assert missing.json()["ready"] is False
    assert any(issue["severity"] == "error" for issue in missing.json()["issues"])
    assert jobs.json() == []


def test_workspace_evaluation_prepare_preserves_model_cache_controls(tmp_path: Path) -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/evaluation/prepare",
            json={
                "job_type": "evaluation",
                "eval_mode": "model-benchmark",
                "model_id": "vchitect-2-t2v",
                "benchmark_id": "vbench",
                "dataset_root": str(tmp_path / "dataset"),
                "generation_cache_dir": str(tmp_path / "cache"),
                "generation_cache_mode": "read-write",
                "resume": True,
                "metric_batch_size": 7,
                "output_dir": str(tmp_path / "output"),
            },
        )
        jobs = client.get("/api/jobs")

    assert response.status_code == 200
    request = response.json()["execution"]["request"]
    assert request["dataset_root"] == str(tmp_path / "dataset")
    assert request["generation_cache_dir"] == str(tmp_path / "cache")
    assert request["generation_cache_mode"] == "read-write"
    assert request["resume"] is True
    assert request["metric_batch_size"] == 7
    assert jobs.json() == []


def test_workspace_auth_token_is_only_sent_to_same_origin_http_urls() -> None:
    assert "function isSameOriginHttpUrl(raw)" in WORKSPACE_HTML
    assert "url.origin === window.location.origin" in WORKSPACE_HTML
    assert '["http:", "https:"].includes(url.protocol)' in WORKSPACE_HTML
    assert "studioAuthToken && isSameOriginHttpUrl(url)" in WORKSPACE_HTML
    assert 'headers.set("Authorization", `Bearer ${studioAuthToken}`)' in WORKSPACE_HTML


def test_workspace_rejects_removed_visualizer(monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_app.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("a removed visualizer must not start a process"),  # ty: ignore[invalid-argument-type]  # ty 0.0.99 misreads pytest.fail's signature
    )

    with TestClient(create_app()) as client:
        visualizers = client.get("/api/visualizers").json()
        assert "world" in {row["mode"] for row in visualizers}
        assert "unified" not in {row["mode"] for row in visualizers}
        response = client.post("/api/visualizers/unified/launch", json={"model_id": "lingbot-world"})

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown visualizer: unified"


def test_workspace_catalog_models_expose_official_links() -> None:
    client = TestClient(create_app())
    models = {row["id"]: row for row in client.get("/api/models").json()}

    solaris = models["solaris"]["links"]
    assert solaris["github"] == "https://github.com/solaris-wm/solaris"
    assert solaris["project"] == "https://solaris-wm.github.io/"
    assert solaris["paper"] == "https://arxiv.org/abs/2602.22208"

    astra = models["astra"]["links"]
    assert astra["github"] == "https://github.com/EternalEvan/Astra"
    assert astra["project"] == "https://eternalevan.github.io/Astra-project/"
    assert astra["paper"] == "https://arxiv.org/abs/2512.08931"

    cogvideox = models["cogvideox_5b_t2v"]["links"]
    assert cogvideox["github"] == "https://github.com/THUDM/CogVideo"

    fantasyworld = models["fantasyworld-wan21"]["links"]
    assert fantasyworld["github"] == "https://github.com/Fantasy-AMAP/fantasy-world"
    assert fantasyworld["project"] == "https://fantasy-amap.github.io/fantasy-world/"
    assert fantasyworld["paper"] == "https://arxiv.org/abs/2509.21657"

    sana = models["sana"]["links"]
    assert sana["github"] == "https://github.com/NVlabs/Sana"
    assert sana["project"] == "https://nvlabs.github.io/Sana/"
    assert sana["paper"] == "https://arxiv.org/abs/2410.10629"

    assert "function catalogLinksHtml(links)" in WORKSPACE_HTML
    assert 'externalLink("Project", links.project)' in WORKSPACE_HTML


def test_workspace_model_contracts_drive_inference_inputs() -> None:
    client = TestClient(create_app())
    models = {row["id"]: row for row in client.get("/api/models").json()}

    depth_inputs = {item["field_id"] for item in models["depth-anything-v2"]["tasks"][0]["inputs"]}
    depth_fields = {item["field_id"]: item for item in models["depth-anything-v2"]["tasks"][0]["inputs"]}
    assert {"input_path", "grayscale", "input-size"} <= depth_inputs
    assert not {"fps", "frames", "steps", "guidance_scale"} & depth_inputs
    assert "output_path" not in depth_inputs
    assert depth_fields["input_path"]["default"].endswith("test_cases/images/000.png")

    video_inputs = {item["field_id"] for item in models["cogvideox_5b_t2v"]["tasks"][0]["inputs"]}
    assert "fps" in video_inputs
    assert "output_path" in video_inputs
    assert {"frames", "steps", "height", "width", "seed"} <= video_inputs
    video_field_ids = [item["field_id"] for item in models["cogvideox_5b_t2v"]["tasks"][0]["inputs"]]
    assert len(video_field_ids) == len(set(video_field_ids))

    lingbot_inputs = {item["field_id"] for item in models["lingbot-world"]["tasks"][0]["inputs"]}
    assert {"image", "interactions", "frames", "steps", "seed"} <= lingbot_inputs
    assert "output_path" not in lingbot_inputs

    worldcam_fields = {item["field_id"]: item for item in models["worldcam"]["tasks"][0]["inputs"]}
    assert worldcam_fields["fps"]["default"] == 30
    assert worldcam_fields["height"]["default"] == 480
    assert worldcam_fields["width"]["default"] == 832

    worldplay_fields = {item["field_id"]: item for item in models["hunyuan-worldplay"]["tasks"][0]["inputs"]}
    assert worldplay_fields["torchrun_nproc_per_node"]["default"] == 1
    assert worldplay_fields["torchrun_nproc_per_node"]["target"] == "load_kwargs"
    assert "multi-GPU run" in worldplay_fields["torchrun_nproc_per_node"]["description"]

    matrix2 = models["matrix-game-2"]
    matrix2_task = matrix2["tasks"][0]
    matrix2_fields = {item["field_id"]: item for item in matrix2_task["inputs"]}
    assert matrix2["default_task_id"] == "official-universal-image"
    assert matrix2["default_variant_id"] == "universal"
    assert matrix2_fields["image"]["default"].endswith("test_cases/matrix-game-2/universal/0000.png")
    assert matrix2_fields["mode"]["default"] == "universal"
    assert matrix2_fields["size"]["default"] == [352, 640]
    assert matrix2_fields["seed"]["default"] == 42
    assert matrix2_fields["official_bench_actions"]["default"] is True
    assert matrix2_task["default_call_kwargs"]["official_bench_actions"] is True

    matrix3 = models["matrix-game-3"]
    matrix3_task = matrix3["tasks"][0]
    matrix3_fields = {item["field_id"]: item for item in matrix3_task["inputs"]}
    assert matrix3["default_task_id"] == "official-cityscape-image"
    assert matrix3["default_variant_id"] == "official"
    assert (
        matrix3_fields["prompt"]["default"]
        == "A colorful, animated cityscape with a gas station and various buildings."
    )
    assert matrix3_fields["num_iterations"]["default"] == 12
    assert matrix3_fields["num_inference_steps"]["default"] == 3
    assert matrix3_fields["size"]["default"] == "704*1280"
    assert matrix3_fields["use_async_vae"]["default"] is False
    assert matrix3_fields["async_vae_warmup_iters"]["default"] == 0

    astra = models["astra"]
    astra_fields = {item["field_id"]: item for item in astra["tasks"][0]["inputs"]}
    assert astra["default_task_id"] == "official-sekai-image"
    assert astra_fields["image"]["default"].endswith("test_cases/astra/condition_images/garden_1.png")
    assert astra_fields["frames_per_generation"]["default"] == 8
    assert astra_fields["total_frames_to_generate"]["default"] == 24
    assert astra_fields["modality_type"]["default"] == "sekai"

    neoverse = models["neoverse"]
    neoverse_fields = {item["field_id"]: item for item in neoverse["tasks"][0]["inputs"]}
    assert neoverse["default_task_id"] == "official-tilt-up-video"
    assert neoverse_fields["video"]["default"].endswith("test_cases/neoverse/videos/robot.mp4")
    assert neoverse_fields["predefined_trajectory"]["default"] == "tilt_up"
    assert neoverse_fields["height"]["target"] == "load_kwargs"
    assert neoverse_fields["height"]["default"] == 336
    assert neoverse_fields["width"]["default"] == 560
    assert neoverse_fields["num_inference_steps"]["default"] == 4

    cosmos = models["cosmos-predict2.5"]
    cosmos_fields = {item["field_id"]: item for item in cosmos["tasks"][0]["inputs"]}
    assert cosmos["workload_type"] == "world"
    assert cosmos["default_task_id"] == "text-to-world-validation"
    assert cosmos_fields["input_path"]["required"] is False
    assert cosmos_fields["input_path"]["default"] == ""
    assert cosmos_fields["num_frames"]["target"] == "call_kwargs"
    assert cosmos_fields["num_frames"]["default"] == 93
    assert cosmos_fields["num_inference_steps"]["target"] == "call_kwargs"
    assert cosmos_fields["num_inference_steps"]["default"] == 35
    assert cosmos_fields["height"]["target"] == "call_kwargs"
    assert cosmos_fields["width"]["default"] == 1280

    cameractrl = models["cameractrl"]
    cameractrl_fields = {item["field_id"]: item for item in cameractrl["tasks"][0]["inputs"]}
    assert cameractrl["default_task_id"] == "trajectory-video"
    assert "mountain lake at sunrise" in cameractrl_fields["prompt"]["default"]
    assert cameractrl_fields["trajectory_file"]["default"].endswith(
        "test_cases/cameractrl/pose_files/0f47577ab3441480.txt"
    )
    assert cameractrl_fields["trajectory_file"]["target"] == "call_kwargs"

    cut3r = models["cut3r"]
    cut3r_fields = {item["field_id"]: item for item in cut3r["tasks"][0]["inputs"]}
    assert cut3r["default_task_id"] == "official-export"
    assert cut3r_fields["input_path"]["default"].endswith("test_cases/cut3r/examples/001")
    assert cut3r_fields["input_path"]["required"] is True

    da3 = models["depth-anything-v3"]
    da3_fields = {item["field_id"]: item for item in da3["tasks"][0]["inputs"]}
    assert da3["default_task_id"] == "official-soh-export"
    assert da3_fields["input_path"]["default"].endswith("test_cases/depth_anything_v3/examples/SOH")
    assert da3_fields["input_path"]["required"] is True

    wan2 = models["wan2.2-ti2v-5b"]
    wan2_fields = {item["field_id"]: item for item in wan2["tasks"][0]["inputs"]}
    assert wan2_fields["prompt"]["default"]
    assert wan2_fields["prompt"]["target"] == "prompt"

    hy15_t2v = models["hunyuanvideo-1.5-t2v"]
    hy15_t2v_fields = {item["field_id"]: item for item in hy15_t2v["tasks"][0]["inputs"]}
    assert hy15_t2v_fields["prompt"]["default"] == "A cat walks on a snowy street, cinematic, high quality."
    assert hy15_t2v_fields["frames"]["default"] == 121
    assert hy15_t2v_fields["steps"]["default"] == 50
    assert hy15_t2v_fields["height"]["default"] == 720
    assert hy15_t2v_fields["width"]["default"] == 1280

    hy15_i2v = models["hunyuanvideo-1.5-i2v"]
    hy15_i2v_fields = {item["field_id"]: item for item in hy15_i2v["tasks"][0]["inputs"]}
    assert hy15_i2v["workload_type"] == "i2v"
    assert hy15_i2v_fields["input_path"]["default"].endswith("test_cases/hunyuanvideo_i2v/0.jpg")
    assert hy15_i2v_fields["frames"]["default"] == 121
    assert hy15_i2v_fields["steps"]["default"] == 50

    mmaudio = models["mmaudio"]
    mmaudio_fields = {item["field_id"]: item for item in mmaudio["tasks"][0]["inputs"]}
    mmaudio_outputs = {item["artifact_id"]: item for item in mmaudio["tasks"][0]["outputs"]}
    assert mmaudio["default_task_id"] == "video-to-audio"
    assert mmaudio["workload_type"] == "v2a"
    assert mmaudio_fields["input_path"]["default"].endswith("test_cases/longcat_video/motorcycle.mp4")
    assert mmaudio_outputs["audio"]["kind"] == "audio"

    open_magvit2 = models["open-magvit2"]
    open_magvit2_outputs = {item["artifact_id"]: item for item in open_magvit2["tasks"][0]["outputs"]}
    assert open_magvit2["default_task_id"] == "image-generation"
    assert open_magvit2["workload_type"] == "image"
    assert open_magvit2_outputs["image"]["kind"] == "generated_image"

    matrix1_fields = {item["field_id"]: item for item in models["matrix-game-1"]["tasks"][0]["inputs"]}
    assert matrix1_fields["input_path"]["default"].endswith(
        "test_cases/matrix-game-1/official_initial_image/forest_00.jpg"
    )

    for model_id in (
        "diamond",
        "oasis-500m",
        "vid2world",
        "mineworld",
        "hunyuanworld-1",
        "pandora",
        "omnivinci",
        "emu3.5",
    ):
        assert model_id not in models

    vggt_fields = {item["field_id"]: item for item in models["vggt"]["tasks"][0]["inputs"]}
    assert vggt_fields["input_path"]["default"].endswith("test_cases/vggt/examples/kitchen/images")

    action_fields = {item["field_id"]: item for item in models["lapa"]["tasks"][0]["inputs"]}
    assert action_fields["input_path"]["default"].endswith("test_cases/test_vla_case1/droid/exterior_image_1_left.png")


def test_workspace_job_defaults_settings_persist(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "studio-settings.json"
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_SETTINGS_FILE", str(settings_path))
    workspace_app.SETTINGS.clear()
    workspace_app.SETTINGS.update(workspace_app.DEFAULT_SETTINGS)
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/settings",
            json={"values": {"fps": "12", "num_frames": "33", "attention_backend": "torch"}},
        )
        response.raise_for_status()
        assert response.json()["fps"] == 12
        assert response.json()["num_frames"] == 33
        assert json.loads(settings_path.read_text(encoding="utf-8"))["attention_backend"] == "torch"

        workspace_app.SETTINGS.clear()
        workspace_app.SETTINGS.update(workspace_app.DEFAULT_SETTINGS)
        client = TestClient(create_app())
        assert client.get("/api/settings").json()["fps"] == 12

        bad_response = client.post("/api/settings", json={"values": {"attention_backend": "sdpa"}})
        assert bad_response.status_code == 400
    finally:
        workspace_app.SETTINGS.clear()
        workspace_app.SETTINGS.update(workspace_app.DEFAULT_SETTINGS)


def test_workspace_routes_inference_artifacts_to_visualizers(tmp_path: Path) -> None:
    import numpy as np

    manifest = tmp_path / "manifest.json"
    image = tmp_path / "preview.png"
    mesh = tmp_path / "scene.glb"
    timeline = tmp_path / "timeline.rrd"
    point_ply = tmp_path / "points.ply"
    splat_ply = tmp_path / "gaussian.ply"
    point_npz = tmp_path / "points.npz"
    camera_npz = tmp_path / "camera.npz"
    for path in (manifest, image, mesh, timeline):
        path.write_bytes(b"worldfoundry")
    point_ply.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")
    splat_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 0",
                "property float opacity",
                "property float scale_0",
                "property float f_dc_0",
                "property float rot_0",
                "end_header",
            ]
        ),
        encoding="utf-8",
    )
    np.savez(point_npz, points=np.zeros((2, 3), dtype=np.float32), rgb=np.zeros((2, 3), dtype=np.uint8))
    np.savez(camera_npz, pose=np.eye(4, dtype=np.float32), intrinsics=np.eye(3, dtype=np.float32))

    assert workspace_app._visualizer_mode_for_artifact(str(point_ply)) == "points"
    assert workspace_app._visualizer_mode_for_artifact(str(point_npz)) == "points"
    assert workspace_app._visualizer_mode_for_artifact(str(camera_npz)) == ""
    assert workspace_app._visualizer_mode_for_artifact(str(splat_ply)) == "spark"

    record = workspace_app.RunRecord(
        run_id="pytest-visualizers",
        model_id="vggt-omega",
        display_name="VGGT-Omega",
        mode="inference",
        status="completed",
        output_dir=str(tmp_path),
        manifest_path=str(manifest),
        preview_image=str(image),
        preview_model=str(mesh),
        preview_splat=str(splat_ply),
        rrd_path=str(timeline),
        artifacts=[str(point_ply)],
    )

    payload = workspace_app._job_result_payload(record)
    assert payload is not None
    assert "visualizations" not in payload
    actions = workspace_app._result_visualization_actions(record)
    by_mode = {item["mode"]: item for item in actions}
    assert len(actions) == 3
    assert by_mode["points"]["path"] == str(mesh)
    assert by_mode["spark"]["path"] == str(splat_ply)
    assert by_mode["rerun"]["path"] == str(timeline)


def test_workspace_visualization_actions_deduplicate_geometry_artifacts(tmp_path: Path) -> None:
    import numpy as np

    output_dir = tmp_path / "inspatio-output"
    output_dir.mkdir()
    for index in range(12):
        np.savez(
            output_dir / f"frame_{index:03d}_depth.npz",
            depth=np.zeros((4, 4), dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
        )
        (output_dir / f"frame_{index:03d}.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
            encoding="utf-8",
        )
    primary_ply = output_dir / "scene.ply"
    primary_ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
        encoding="utf-8",
    )

    record = workspace_app.RunRecord(
        run_id="pytest-dedupe",
        model_id="inspatio-world",
        display_name="InSpatio-World",
        mode="inference",
        status="completed",
        output_dir=str(output_dir),
        manifest_path=str(output_dir / "manifest.json"),
        artifacts=sorted(str(path) for path in output_dir.rglob("*") if path.is_file()),
    )

    actions = workspace_app._result_visualization_actions(record)
    points_actions = [item for item in actions if item["mode"] == "points"]
    assert len(points_actions) == 1
    assert points_actions[0]["label"] == "Open in Viser"


def test_workspace_rejects_asset_gated_worldmodel_jobs(monkeypatch) -> None:
    monkeypatch.setattr(workspace_app, "dispatch_spec_for_inference", lambda *args, **kwargs: None)
    client = TestClient(create_app())

    for model_id in ("diamond", "vid2world"):
        response = client.post(
            "/api/jobs",
            json={
                "job_type": "inference",
                "workload_type": "world",
                "model_id": model_id,
                "input_path": "worldfoundry/data/test_cases/studio_demo/00/image.jpg",
                "params": {"interactions": ["forward"], "frames": 16, "fps": 16, "seed": 42},
            },
        )

        assert response.status_code == 400
        assert "not available in the Workspace inference catalog" in response.json()["detail"]


def test_workspace_runtime_options_are_model_capabilities() -> None:
    client = TestClient(create_app())
    models = {row["id"]: row for row in client.get("/api/models").json()}

    depth_options = models["depth-anything-v2"]["runtime_options"]
    assert not any(option["supported"] for option in depth_options.values())

    gamecraft_options = models["hunyuan-game-craft"]["runtime_options"]
    assert gamecraft_options["cpu_offload"]["supported"] is True
    assert gamecraft_options["torch_compile"]["supported"] is False

    matrix_options = models["matrix-game-2"]["runtime_options"]
    assert matrix_options["torch_compile"]["supported"] is True


def test_workspace_rejects_unsupported_runtime_options() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/jobs",
        json={
            "job_type": "inference",
            "model_id": "depth-anything-v2",
            "params": {"cpu_offload": True},
        },
    )

    assert response.status_code == 400
    assert "does not implement" in response.json()["detail"]


def test_workspace_rejects_inference_params_not_in_model_contract() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/jobs",
        json={
            "job_type": "inference",
            "model_id": "depth-anything-v2",
            "params": {"fps": 8},
        },
    )

    assert response.status_code == 400
    assert "does not use" in response.json()["detail"]


def test_workspace_rejects_undeclared_call_kwargs() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/jobs",
        json={
            "job_type": "inference",
            "model_id": "depth-anything-v2",
            "call_kwargs": {"fps": 8},
        },
    )

    assert response.status_code == 400
    assert "does not declare" in response.json()["detail"]


def test_workspace_accepts_curated_task_declared_call_kwargs() -> None:
    entry = workspace_app.find_entry("matrix-game-3")
    task = inference_requests._entry_inference_spec(entry).task()
    inference_requests._validate_inference_payload(
        entry,
        task,
        inference_requests.JobCreateRequest(
            model_id="matrix-game-3",
            call_kwargs={
                "num_iterations": 12,
                "num_inference_steps": 3,
                "size": "704*1280",
                "use_int8": True,
                "compile_vae": True,
                "use_async_vae": False,
                "async_vae_warmup_iters": 0,
            },
        ),
    )


def test_workspace_accepts_generic_video_runtime_params() -> None:
    entry = workspace_app.find_entry("cogvideox_2b_t2v")
    task = inference_requests._entry_inference_spec(entry).task()
    inference_requests._validate_inference_payload(
        entry,
        task,
        inference_requests.JobCreateRequest(
            model_id="cogvideox_2b_t2v",
            params={
                "num_frames": 17,
                "height": 480,
                "width": 720,
                "num_inference_steps": 10,
                "fps": 8,
                "seed": 42,
            },
        ),
    )


def test_workspace_accepts_dispatch_only_kwargs() -> None:
    entry = workspace_app.find_entry("lingbot-world")
    task = inference_requests._entry_inference_spec(entry).task()
    inference_requests._validate_inference_payload(
        entry,
        task,
        inference_requests.JobCreateRequest(
            model_id="lingbot-world",
            load_kwargs={"cuda_visible_devices": "0,1,2,3"},
            call_kwargs={"python_executable": "/tmp/worldfoundry-python"},
        ),
    )


def test_workspace_builds_official_astra_and_neoverse_run_kwargs() -> None:
    _, astra_kwargs = inference_requests.build_inference_run_kwargs(
        inference_requests.JobCreateRequest(
            model_id="astra",
            workload_type="i2v",
            call_kwargs={
                "frames_per_generation": 8,
                "total_frames_to_generate": 24,
                "num_inference_steps": 50,
                "modality_type": "sekai",
            },
            params={"interactions": ["forward"]},
        )
    )
    astra_call_kwargs = json.loads(astra_kwargs["call_kwargs_text"])
    assert astra_kwargs["input_path"].endswith("test_cases/astra/condition_images/garden_1.png")
    assert astra_call_kwargs["frames_per_generation"] == 8
    assert astra_call_kwargs["modality_type"] == "sekai"

    _, neoverse_kwargs = inference_requests.build_inference_run_kwargs(
        inference_requests.JobCreateRequest(
            model_id="neoverse",
            call_kwargs={
                "predefined_trajectory": "tilt_up",
                "num_frames": 81,
                "num_inference_steps": 4,
                "cfg_scale": 1.0,
            },
            load_kwargs={"height": 336, "width": 560},
        )
    )
    neoverse_call_kwargs = json.loads(neoverse_kwargs["call_kwargs_text"])
    neoverse_load_kwargs = json.loads(neoverse_kwargs["load_kwargs_text"])
    assert neoverse_kwargs["input_path"].endswith("test_cases/neoverse/videos/robot.mp4")
    assert neoverse_call_kwargs["predefined_trajectory"] == "tilt_up"
    assert neoverse_call_kwargs["num_inference_steps"] == 4
    assert neoverse_load_kwargs["height"] == 336
    assert neoverse_load_kwargs["width"] == 560


def test_workspace_rebinds_neoverse_catalog_model_path_to_resolved_ref(monkeypatch) -> None:
    entry = workspace_app.find_entry("neoverse")
    spec = inference_requests._entry_inference_spec(entry)
    variant = spec.variant()
    stale_model_path = "/home/ephemeral/ckpts/Yuppie1204--NeoVerse"
    patched_variant = replace(
        variant,
        load_kwargs={**variant.load_kwargs, "model_path": stale_model_path},
    )
    patched_spec = replace(
        spec,
        variants=tuple(patched_variant if item.variant_id == variant.variant_id else item for item in spec.variants),
    )
    monkeypatch.setattr(inference_requests, "_entry_inference_spec", lambda _entry: patched_spec)

    resolved_model_ref = "/mnt/persistent/ckpts/Yuppie1204--NeoVerse"
    _, run_kwargs = inference_requests.build_inference_run_kwargs(
        inference_requests.JobCreateRequest(
            model_id="neoverse",
            model_ref=resolved_model_ref,
        )
    )

    assert run_kwargs["model_ref"] == resolved_model_ref
    assert json.loads(run_kwargs["load_kwargs_text"])["model_path"] == resolved_model_ref


def test_workspace_preserves_explicit_neoverse_model_path(monkeypatch) -> None:
    entry = workspace_app.find_entry("neoverse")
    spec = inference_requests._entry_inference_spec(entry)
    variant = spec.variant()
    stale_model_path = "/home/ephemeral/ckpts/Yuppie1204--NeoVerse"
    patched_variant = replace(
        variant,
        load_kwargs={**variant.load_kwargs, "model_path": stale_model_path},
    )
    patched_spec = replace(
        spec,
        variants=tuple(patched_variant if item.variant_id == variant.variant_id else item for item in spec.variants),
    )
    monkeypatch.setattr(inference_requests, "_entry_inference_spec", lambda _entry: patched_spec)

    custom_model_path = "/custom/checkpoints/neoverse-quantized"
    _, run_kwargs = inference_requests.build_inference_run_kwargs(
        inference_requests.JobCreateRequest(
            model_id="neoverse",
            model_ref="/mnt/persistent/ckpts/Yuppie1204--NeoVerse",
            load_kwargs={"model_path": custom_model_path},
        ),
        validate=False,
    )

    assert json.loads(run_kwargs["load_kwargs_text"])["model_path"] == custom_model_path


def test_workspace_builds_cosmos_prompt_only_smoke_run_kwargs() -> None:
    _, cosmos_kwargs = inference_requests.build_inference_run_kwargs(
        inference_requests.JobCreateRequest(
            model_id="cosmos-predict2.5",
            workload_type="world",
        )
    )
    cosmos_call_kwargs = json.loads(cosmos_kwargs["call_kwargs_text"])
    cosmos_load_kwargs = json.loads(cosmos_kwargs["load_kwargs_text"])
    assert cosmos_kwargs["input_path"] == ""
    assert "nighttime city bus terminal" in cosmos_kwargs["prompt"]
    assert cosmos_call_kwargs["num_frames"] == 93
    assert cosmos_call_kwargs["num_inference_steps"] == 35
    assert cosmos_call_kwargs["height"] == 704
    assert cosmos_call_kwargs["width"] == 1280
    assert cosmos_call_kwargs["guidance_scale"] == 7.0
    assert cosmos_load_kwargs["required_components"]["text_encoder_model_path"]


def test_workspace_builds_cameractrl_official_defaults() -> None:
    _, cameractrl_kwargs = inference_requests.build_inference_run_kwargs(
        inference_requests.JobCreateRequest(
            model_id="cameractrl",
            workload_type="world",
        )
    )
    call_kwargs = json.loads(cameractrl_kwargs["call_kwargs_text"])
    assert "mountain lake at sunrise" in cameractrl_kwargs["prompt"]
    assert call_kwargs["trajectory_file"].endswith("test_cases/cameractrl/pose_files/0f47577ab3441480.txt")
    assert call_kwargs["original_pose_width"] == 1280
    assert call_kwargs["original_pose_height"] == 720


def test_workspace_maps_supported_runtime_options_to_real_kwargs() -> None:
    gamecraft_entry = workspace_app.find_entry("hunyuan-game-craft")
    call_kwargs, load_kwargs = inference_requests._merge_common_params(
        gamecraft_entry,
        inference_requests.JobCreateRequest(
            model_id="hunyuan-game-craft",
            params={"cpu_offload": True},
        ),
    )
    assert call_kwargs == {}
    assert load_kwargs["cpu_offload"] is True

    matrix_entry = workspace_app.find_entry("matrix-game-2")
    _, matrix_load_kwargs = inference_requests._merge_common_params(
        matrix_entry,
        inference_requests.JobCreateRequest(
            model_id="matrix-game-2",
            params={"torch_compile": True},
        ),
    )
    assert matrix_load_kwargs["torch_compile"] is True


def test_workspace_maps_common_params_to_model_specific_aliases() -> None:
    gamecraft_entry = workspace_app.find_entry("hunyuan-game-craft")
    call_kwargs, _ = inference_requests._merge_common_params(
        gamecraft_entry,
        inference_requests.JobCreateRequest(
            model_id="hunyuan-game-craft",
            params={"num_inference_steps": 12, "guidance_scale": 1.5},
        ),
    )
    assert call_kwargs["infer_steps"] == 12
    assert call_kwargs["cfg_scale"] == 1.5

    lingbot_entry = workspace_app.find_entry("lingbot-world")
    call_kwargs, _ = inference_requests._merge_common_params(
        lingbot_entry,
        inference_requests.JobCreateRequest(
            model_id="lingbot-world",
            params={"num_inference_steps": 20, "interactions": ["forward", "camera_r"]},
        ),
    )
    assert call_kwargs["sampling_steps"] == 20
    assert call_kwargs["interactions"] == ["forward", "camera_r"]


def test_workspace_api_runs_real_evaluation_job(tmp_path: Path) -> None:
    video_path = tmp_path / "generated.mp4"
    video_path.write_bytes(b"worldfoundry-test-video")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "status": "succeeded",
                "artifacts": {
                    "generated_video": {
                        "uri": str(video_path),
                        "kind": "video",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app())

    response = client.post(
        "/api/jobs",
        json={
            "job_type": "evaluation",
            "eval_mode": "existing-results",
            "benchmark_id": "vbench",
            "resume": True,
            "metric_batch_size": 7,
            "results_path": str(results_path),
            "output_dir": str(tmp_path / "eval_out"),
            "metrics": ["required_artifacts_present"],
            "required_artifacts": ["generated_video"],
        },
    )

    response.raise_for_status()
    job = _wait_for_job(client, response.json()["id"])
    assert job["status"] == "completed"
    assert job["job_type"] == "evaluation"
    assert job["result"]["mode"] == "existing-results"
    for key in ("manifest_path", "execution_plan_path", "scorecard_path"):
        assert Path(job["result"][key]).is_file()

    artifact_paths = {row["path"] for row in client.get("/api/artifacts").json() if row["job_id"] == job["id"]}
    assert job["result"]["manifest_path"] in artifact_paths
    assert job["result"]["execution_plan_path"] in artifact_paths
    assert job["result"]["scorecard_path"] in artifact_paths
    assert job["result"]["request"]["resume"] is True
    assert job["result"]["request"]["metric_batch_size"] == 7
    artifact_response = client.get("/api/artifacts/file", params={"path": job["result"]["scorecard_path"]})
    assert artifact_response.status_code == 200
    assert artifact_response.json()["schema_version"] == "worldfoundry-scorecard"


def test_workspace_rejects_unsupported_job_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def launch(*args: object, **kwargs: object) -> object:
        pytest.fail("unsupported jobs must not start a subprocess")  # ty: ignore[invalid-argument-type]  # ty 0.0.99 misreads pytest.fail's signature

    monkeypatch.setattr(workspace_app.subprocess, "Popen", launch)
    with TestClient(create_app()) as client:
        response = client.post("/api/jobs", json={"job_type": "training"})

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported job type: training"


def test_wait_for_visualizer_sends_bearer_token_and_requires_2xx_3xx(monkeypatch: pytest.MonkeyPatch) -> None:
    import email.message
    import urllib.error
    import urllib.request

    seen_headers: list[dict[str, str]] = []
    attempts = {"count": 0}

    class _FakeResponse:
        status = 200

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: urllib.request.Request, timeout: float = 2) -> _FakeResponse:
        seen_headers.append(dict(req.header_items()))
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", email.message.Message(), None)
        if attempts["count"] == 2:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", email.message.Message(), None)
        return _FakeResponse()

    monkeypatch.setattr(workspace_app.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(workspace_app.time, "sleep", lambda _s: None)

    assert workspace_app._wait_for_visualizer("http://127.0.0.1:8765/healthz", timeout=5.0, auth_token="secret-tok")
    assert attempts["count"] == 3
    assert all(headers.get("Authorization") == "Bearer secret-tok" for headers in seen_headers)


def test_wait_for_visualizer_returns_false_immediately_when_process_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    from typing import cast

    class _DeadProcess:
        def poll(self) -> int:
            return 1

    def fail_urlopen(*args: object, **kwargs: object) -> object:
        pytest.fail("urlopen should not be called after process exited")  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(workspace_app.urllib.request, "urlopen", fail_urlopen)
    started = time.monotonic()
    ready = workspace_app._wait_for_visualizer(
        "http://127.0.0.1:8765/healthz",
        timeout=60.0,
        process=cast(subprocess.Popen[str], _DeadProcess()),
    )
    elapsed = time.monotonic() - started
    assert ready is False
    assert elapsed < 1.0


def test_launch_visualizer_supports_legacy_wait_monkeypatch_and_validates_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_app, "VISUALIZER_MANAGED", {})
    monkeypatch.setattr(
        workspace_app.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("rerun without asset or URL must not spawn a process"),  # ty: ignore[invalid-argument-type]
    )
    with TestClient(create_app()) as client:
        missing = client.post("/api/visualizers/rerun/launch", json={"model_id": "vggt-omega"})
    assert missing.status_code == 400
    assert "asset path or simulator URL" in missing.json()["detail"]

    media_file = tmp_path / "preview.png"
    media_file.write_bytes(b"png")

    class _AliveProcess:
        pid = 12345

        def poll(self) -> None:
            return None

    probed: list[tuple[str, float]] = []

    def legacy_wait(url: str, *, timeout: float = 35.0) -> bool:
        probed.append((url, timeout))
        return True

    monkeypatch.setattr(workspace_app.subprocess, "Popen", lambda *args, **kwargs: _AliveProcess())
    monkeypatch.setattr(workspace_app, "_wait_for_visualizer", legacy_wait)
    monkeypatch.setattr(workspace_app, "_tcp_port_available", lambda host, port: True)
    monkeypatch.setattr(workspace_app, "_workspace_visualizer_dir", lambda: tmp_path)

    with TestClient(create_app()) as client:
        launched = client.post(
            "/api/visualizers/media/launch",
            json={"model_id": "vggt-omega", "asset_path": str(media_file)},
        )

    assert launched.status_code == 200
    assert launched.json()["running"] is True
    assert len(probed) == 1
    assert probed[0][0].endswith("/healthz")


def test_visualizer_reusable_checks_simulator_url_host_and_port() -> None:
    import subprocess
    from typing import cast

    class _AliveProcess:
        def poll(self) -> None:
            return None

    local_record = workspace_app.ManagedVisualizer(
        mode="media",
        title="Media Preview",
        url="http://127.0.0.1:18720/",
        health_url="http://127.0.0.1:18720/healthz",
        host="127.0.0.1",
        port=18720,
        model_id="vggt-omega",
        asset_path="/tmp/a.png",
        command=[],
        log_path=None,
        started_at=time.time(),
        params={},
        process=cast(subprocess.Popen[str], _AliveProcess()),
        external=False,
    )
    assert workspace_app._visualizer_reusable(
        local_record,
        model_id="vggt-omega",
        asset_path="/tmp/a.png",
        params={},
        simulator_url="",
        host="127.0.0.1",
        port=None,
    )
    assert workspace_app._visualizer_reusable(
        local_record,
        model_id="vggt-omega",
        asset_path="/tmp/a.png",
        params={},
        simulator_url="",
        host="127.0.0.1",
        port=18720,
    )
    assert not workspace_app._visualizer_reusable(
        local_record,
        model_id="vggt-omega",
        asset_path="/tmp/a.png",
        params={},
        simulator_url="",
        host="0.0.0.0",
        port=None,
    )
    assert not workspace_app._visualizer_reusable(
        local_record,
        model_id="vggt-omega",
        asset_path="/tmp/a.png",
        params={},
        simulator_url="",
        host="127.0.0.1",
        port=18721,
    )
    assert not workspace_app._visualizer_reusable(
        local_record,
        model_id="vggt-omega",
        asset_path="/tmp/a.png",
        params={},
        simulator_url="http://127.0.0.1:9999/",
        host="127.0.0.1",
        port=None,
    )

    ext_record = workspace_app.ManagedVisualizer(
        mode="rerun",
        title="Rerun",
        url="http://127.0.0.1:9876/",
        health_url="http://127.0.0.1:9876/",
        host="127.0.0.1",
        port=9876,
        model_id="vggt-omega",
        asset_path="",
        command=[],
        log_path=None,
        started_at=time.time(),
        params={},
        process=None,
        external=True,
    )
    assert workspace_app._visualizer_reusable(
        ext_record,
        model_id="vggt-omega",
        asset_path="",
        params={},
        simulator_url="http://127.0.0.1:9876/",
        host="127.0.0.1",
        port=None,
    )
    assert not workspace_app._visualizer_reusable(
        ext_record,
        model_id="vggt-omega",
        asset_path="",
        params={},
        simulator_url="http://127.0.0.1:9877/",
        host="127.0.0.1",
        port=None,
    )
    assert not workspace_app._visualizer_reusable(
        ext_record,
        model_id="vggt-omega",
        asset_path="",
        params={},
        simulator_url="",
        host="127.0.0.1",
        port=None,
    )


def test_non_loopback_workspace_bind_host_defaults_visualizers_except_viser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_app, "VISUALIZER_MANAGED", {})
    monkeypatch.setattr(workspace_app, "_tcp_port_available", lambda host, port: True)
    monkeypatch.setattr(workspace_app, "_workspace_visualizer_dir", lambda: tmp_path)

    spawned_commands: list[list[str]] = []
    probed_tokens: list[str] = []

    class _AliveProcess:
        pid = 54321

        def poll(self) -> None:
            return None

    def fake_popen(cmd: list[str], **kwargs: object) -> _AliveProcess:
        spawned_commands.append(list(cmd))
        return _AliveProcess()

    def fake_wait(
        url: str,
        *,
        timeout: float = 35.0,
        process: object = None,
        auth_token: str = "",
    ) -> bool:
        probed_tokens.append(auth_token)
        return True

    monkeypatch.setattr(workspace_app.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(workspace_app, "_wait_for_visualizer", fake_wait)

    asset_img = tmp_path / "img.png"
    asset_img.write_bytes(b"png")
    asset_ply = tmp_path / "cloud.ply"
    asset_ply.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")

    headers = {"Authorization": "Bearer remote-secret"}
    with TestClient(create_app(auth_token="remote-secret", bind_host="0.0.0.0")) as client:
        rows = {row["mode"]: row for row in client.get("/api/visualizers", headers=headers).json()}
        assert rows["media"]["default_host"] == "0.0.0.0"
        assert rows["spark"]["default_host"] == "0.0.0.0"
        assert rows["world"]["default_host"] == "0.0.0.0"
        assert rows["points"]["default_host"] == "127.0.0.1"

        media_res = client.post(
            "/api/visualizers/media/launch",
            headers=headers,
            json={"model_id": "vggt-omega", "asset_path": str(asset_img)},
        )
        points_res = client.post(
            "/api/visualizers/points/launch",
            headers=headers,
            json={"model_id": "vggt-omega", "asset_path": str(asset_ply)},
        )

    assert media_res.status_code == 200
    assert media_res.json()["host"] == "0.0.0.0"
    assert points_res.status_code == 200
    assert points_res.json()["host"] == "127.0.0.1"
    assert spawned_commands[0][spawned_commands[0].index("--host") + 1] == "0.0.0.0"
    assert spawned_commands[1][spawned_commands[1].index("--host") + 1] == "127.0.0.1"
    assert probed_tokens == ["remote-secret", "remote-secret"]


def test_job_payload_exposes_logs_total_past_200_lines() -> None:
    job = workspace_app.JOBS.submit_run(
        title="Long Log Job",
        model_id="vggt-omega",
        display_name="VGGT-Omega",
        action="inference",
        job_type="inference",
        metadata={},
        run_callable=lambda current: (
            [current.append_log("stdout", f"line-{i}\n") for i in range(250)],
            {"status": "completed"},
        )[1],
    )
    with TestClient(create_app()) as client:
        payload = _wait_for_job(client, job.job_id)
        delta = client.get(f"/api/jobs/{job.job_id}/logs", params={"after": 200}).json()

    assert len(payload["logs"]) == 200
    assert payload["logs_total"] > 200
    assert delta["offset"] == payload["logs_total"]
    assert len(delta["logs"]) == payload["logs_total"] - 200


def test_models_filter_matches_variant_workloads_and_ui_has_image_and_v2a() -> None:
    assert '<option value="image">Image</option>' in WORKSPACE_HTML
    assert '<option value="v2a">Video to Audio</option>' in WORKSPACE_HTML
    with TestClient(create_app()) as client:
        i2v_ids = {row["id"] for row in client.get("/api/models", params={"workload_type": "i2v"}).json()}
        image_ids = {row["id"] for row in client.get("/api/models", params={"workload_type": "image"}).json()}
        v2a_ids = {row["id"] for row in client.get("/api/models", params={"workload_type": "v2a"}).json()}

    assert "cogvideox_5b_t2v" in i2v_ids
    assert "open-magvit2" in image_ids
    assert "mmaudio" in v2a_ids


def test_load_settings_from_disk_ignores_invalid_values_and_save_failure_preserves_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_path = tmp_path / "studio-settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "fps": 24,
                "num_frames": "not-an-int",
                "attention_backend": "unsupported-backend",
                "guidance_scale": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_SETTINGS_FILE", str(settings_path))
    workspace_app.SETTINGS.clear()
    workspace_app.SETTINGS.update(workspace_app.DEFAULT_SETTINGS)
    try:
        with TestClient(create_app()) as client:
            loaded = client.get("/api/settings").json()
            assert loaded["fps"] == 24
            assert loaded["num_frames"] == workspace_app.DEFAULT_SETTINGS["num_frames"]
            assert loaded["attention_backend"] == workspace_app.DEFAULT_SETTINGS["attention_backend"]
            assert loaded["guidance_scale"] == workspace_app.DEFAULT_SETTINGS["guidance_scale"]

            def fail_save(_values: object = None) -> None:
                raise OSError("read-only filesystem")

            monkeypatch.setattr(workspace_app, "_save_settings_to_disk", fail_save)
            failed = client.post("/api/settings", json={"values": {"fps": 60}})
            assert failed.status_code == 500
            assert client.get("/api/settings").json()["fps"] == 24
    finally:
        workspace_app.SETTINGS.clear()
        workspace_app.SETTINGS.update(workspace_app.DEFAULT_SETTINGS)


def test_create_job_prepared_evaluation_value_error_returns_400(tmp_path: Path) -> None:
    artifacts = tmp_path / "videos"
    artifacts.mkdir()
    (artifacts / "sample.mp4").write_bytes(b"video")
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/jobs",
            json={
                "job_type": "evaluation",
                "eval_mode": "score-artifacts",
                "benchmark_id": "vbench",
                "dataset_root": str(artifacts),
                "resume": True,
                "output_dir": str(tmp_path / "out"),
            },
        )
    assert response.status_code == 400
    assert "resume/metric_batch_size" in response.json()["detail"]


def test_initial_studio_job_counter_scans_runtime_jobs_and_evaluations(tmp_path: Path) -> None:
    (tmp_path / "runtime_jobs" / "studio-00003").mkdir(parents=True)
    (tmp_path / "evaluations" / "studio-00009").mkdir(parents=True)
    (tmp_path / "evaluations" / "studio-not-a-number").mkdir(parents=True)
    assert workspace_app._initial_studio_job_counter(str(tmp_path)) == 9


def test_evaluation_report_md_is_registered_and_served_with_markdown_media_type(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_out"
    eval_dir.mkdir()
    report_path = eval_dir / "report.md"
    report_path.write_text("# Evaluation Report\n\nAll good.\n", encoding="utf-8")
    scorecard_path = eval_dir / "scorecard.json"
    scorecard_path.write_text('{"schema_version": "worldfoundry-scorecard"}\n', encoding="utf-8")

    job = workspace_app.JOBS.submit_run(
        title="Eval Report Job",
        model_id="materialized-results",
        display_name="Evaluation",
        action="evaluation",
        job_type="evaluation",
        metadata={"output_dir": str(eval_dir)},
        run_callable=lambda _job: {
            "status": "completed",
            "output_dir": str(eval_dir),
            "scorecard_path": str(scorecard_path),
        },
    )
    with TestClient(create_app()) as client:
        _wait_for_job(client, job.job_id)
        artifact_paths = {row["path"] for row in client.get("/api/artifacts").json() if row["job_id"] == job.job_id}
        assert str(report_path) in artifact_paths
        res = client.get("/api/artifacts/file", params={"path": str(report_path)})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/markdown")
    assert "# Evaluation Report" in res.text


def test_media_types_cover_extended_formats() -> None:
    expected = {
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".json": "application/json; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }
    for ext, mime in expected.items():
        assert workspace_app._file_media_type(Path(f"sample{ext}")) == mime


def test_workspace_accepts_unremapped_call_kwargs_for_wow_and_gen3c() -> None:
    wow_entry = workspace_app.find_entry("wow")
    wow_task = inference_requests._entry_inference_spec(wow_entry).task()
    inference_requests._validate_inference_payload(
        wow_entry,
        wow_task,
        inference_requests.JobCreateRequest(
            model_id="wow",
            call_kwargs={"steps": 50, "num_frames": 41, "seed": 42},
        ),
    )

    gen3c_entry = workspace_app.find_entry("gen3c")
    gen3c_task = inference_requests._entry_inference_spec(gen3c_entry).task()
    inference_requests._validate_inference_payload(
        gen3c_entry,
        gen3c_task,
        inference_requests.JobCreateRequest(
            model_id="gen3c",
            call_kwargs={"guidance": 1.0, "num_steps": 35, "num_video_frames": 121},
        ),
    )


def test_workspace_html_js_behaviors_with_node() -> None:
    import shutil
    import subprocess

    node_bin = shutil.which("node") or str(Path.home() / ".cache" / "worldfoundry" / "node22" / "bin" / "node")
    if not Path(node_bin).is_file():
        pytest.skip("node runtime not available")  # ty: ignore[too-many-positional-arguments]  # ty 0.0.99 misreads pytest.skip's signature

    html_path = Path(workspace_app.__file__).resolve().parents[1] / "assets" / "web" / "workspace.html"
    script = r"""
const assert = require("assert");
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[1], "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(match, "inline script missing");
const source = match[1];

// Extract helper functions without running page bootstrap side effects.
const initIdx = source.indexOf("initResizableSplits();");
assert.ok(initIdx > 0, "bootstrap marker missing");
const headlessSource = source.slice(0, initIdx) + "\nglobalThis.__state = state;\n";

function makeElement(id, tag = "div") {
  const attrs = new Map();
  const classes = new Set();
  const el = {
    id,
    tagName: tag.toUpperCase(),
    value: "",
    defaultValue: "",
    checked: false,
    type: tag === "input" ? "text" : "",
    dataset: {},
    innerHTML: "",
    textContent: "",
    options: [],
    children: [],
    scrollTop: 0,
    scrollHeight: 100,
    clientHeight: 100,
    classList: {
      add: (...c) => c.forEach(x => classes.add(x)),
      remove: (...c) => c.forEach(x => classes.delete(x)),
      toggle: (c, force) => {
        if (force === undefined) {
          if (classes.has(c)) classes.delete(c);
          else classes.add(c);
        } else if (force) classes.add(c);
        else classes.delete(c);
      },
      contains: c => classes.has(c),
    },
    setAttribute(k, v) {
      attrs.set(k, String(v));
      if (k === "src") {
        this._src = String(v);
        this._srcSets = (this._srcSets || 0) + 1;
      }
    },
    getAttribute(k) {
      return attrs.has(k) ? attrs.get(k) : null;
    },
    removeAttribute(k) {
      attrs.delete(k);
      if (k === "src") this._src = "";
    },
    get src() {
      return this._src || "";
    },
    set src(v) {
      this.setAttribute("src", v);
    },
    closest() {
      return this;
    },
    appendChild(opt) {
      this.options.push(opt);
    },
    querySelectorAll(sel) {
      if (sel === ":scope > .muted" && this._hasMuted) {
        return [{ remove: () => { this._hasMuted = false; this.innerHTML = this.innerHTML.replace(/<span class="muted">[^<]*<\/span>/g, ""); } }];
      }
      return [];
    },
    insertAdjacentHTML(_pos, chunk) {
      this.innerHTML += chunk;
    },
    showModal() {},
    close() {},
  };
  return el;
}

const elements = new Map();
const getEl = (id) => {
  if (!elements.has(id)) elements.set(id, makeElement(id));
  return elements.get(id);
};

const sandbox = {
  URL,
  URLSearchParams,
  Set,
  Map,
  JSON,
  Number,
  Boolean,
  String,
  Array,
  Object,
  Date,
  Math,
  Promise,
  CSS: { escape: s => s },
  console,
  setTimeout,
  clearTimeout,
  window: {
    location: { href: "http://10.20.30.40:7870/?token=tok123", origin: "http://10.20.30.40:7870", hostname: "10.20.30.40", search: "?token=tok123" },
    open: () => {},
    addEventListener: () => {},
  },
  document: {
    getElementById: getEl,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: (tag) => makeElement("", tag),
  },
  alert: (msg) => { throw new Error("unexpected alert: " + msg); },
  fetch: async () => ({ ok: true, json: async () => ({}) }),
  Headers: class { set() {} },
};

vm.createContext(sandbox);
vm.runInContext(headlessSource, sandbox);
const state = sandbox.__state;

// 1. applyInferValue: do not remap call_kwargs/load_kwargs, only params.
const p = { params: {}, call_kwargs: {}, load_kwargs: {} };
sandbox.applyInferValue("call_kwargs", "steps", 50, p);
sandbox.applyInferValue("call_kwargs", "guidance", 1.0, p);
sandbox.applyInferValue("load_kwargs", "steps", 20, p);
sandbox.applyInferValue("params", "steps", 30, p);
sandbox.applyInferValue("params", "frames", 81, p);
assert.strictEqual(p.call_kwargs.steps, 50);
assert.strictEqual(p.call_kwargs.num_inference_steps, undefined);
assert.strictEqual(p.call_kwargs.guidance, 1.0);
assert.strictEqual(p.call_kwargs.guidance_scale, undefined);
assert.strictEqual(p.load_kwargs.steps, 20);
assert.strictEqual(p.params.num_inference_steps, 30);
assert.strictEqual(p.params.num_frames, 81);

// 2. viewerUrlForBrowser: non-loopback token append + Rerun nested url rewrite + no third-party token leak.
assert.strictEqual(
  sandbox.viewerUrlForBrowser("http://127.0.0.1:8765/"),
  "http://10.20.30.40:8765/?token=tok123"
);
assert.strictEqual(
  sandbox.viewerUrlForBrowser("http://127.0.0.1:8765/?token=existing"),
  "http://10.20.30.40:8765/?token=existing"
);
const rerunUrl = new URL(
  sandbox.viewerUrlForBrowser("http://127.0.0.1:9876/?url=ws%3A%2F%2F127.0.0.1%3A9877&renderer=webgpu")
);
assert.strictEqual(rerunUrl.hostname, "10.20.30.40");
assert.strictEqual(rerunUrl.searchParams.get("url"), "ws://10.20.30.40:9877");
assert.strictEqual(rerunUrl.searchParams.get("token"), "tok123");
assert.strictEqual(
  sandbox.viewerUrlForBrowser("https://external.example.com:8080/"),
  "https://external.example.com:8080/"
);

// 3. visualizerHostForLaunch keeps points on loopback even on non-loopback browser.
state.visualizers = [
  { mode: "points", title: "Viser", default_host: "127.0.0.1", default_port: 18590, default_model: "vggt-omega", aliases: [], status: null },
  { mode: "media", title: "Media", default_host: "0.0.0.0", default_port: 18720, default_model: "vggt-omega", aliases: [], status: null },
];
assert.strictEqual(sandbox.visualizerHostForLaunch("points"), "127.0.0.1");
assert.strictEqual(sandbox.visualizerHostForLaunch("media"), "0.0.0.0");

// 4. syncVisualizerPreview reloads iframe when started_at or asset_path changes even with identical URL.
const frame = getEl("visualizerPreviewFrame");
state.visualizers = [
  {
    mode: "media",
    title: "Media",
    status: { running: true, url: "http://127.0.0.1:18720/", started_at: 100, asset_path: "/tmp/a.png" },
  },
  {
    mode: "spark",
    title: "Spark",
    status: { running: true, url: "http://127.0.0.1:8765/", started_at: 200, asset_path: "/tmp/s.splat" },
  },
];
sandbox.syncVisualizerPreview("media");
assert.strictEqual(frame._srcSets, 1);
sandbox.syncVisualizerPreview("media");
assert.strictEqual(frame._srcSets, 1);
state.visualizers[0].status.started_at = 101;
state.visualizers[0].status.asset_path = "/tmp/b.png";
sandbox.syncVisualizerPreview("media");
assert.strictEqual(frame._srcSets, 2);
sandbox.syncVisualizerPreview("spark");
assert.strictEqual(state.visualizerPreviewMode, "spark");

// 5. visualizerCard populates inputs from status and preserves user drafts.
const cardHtml = sandbox.visualizerCard({
  mode: "media",
  title: "Media",
  default_model: "vggt-omega",
  default_port: 18720,
  aliases: ["preview"],
  requires_asset: true,
  accepts_external_url: false,
  status: { running: true, model_id: "cosmos-predict2.5", port: 18725, asset_path: "/tmp/out.mp4", url: "http://127.0.0.1:18725/", external: false, params: {} },
});
assert.ok(cardHtml.includes('value="cosmos-predict2.5"'));
assert.ok(cardHtml.includes('value="18725"'));
assert.ok(cardHtml.includes('value="/tmp/out.mp4"'));
state.visualizerDrafts.media = { asset: "/tmp/user-edited.mp4" };
const draftCardHtml = sandbox.visualizerCard({
  mode: "media",
  title: "Media",
  default_model: "vggt-omega",
  default_port: 18720,
  aliases: ["preview"],
  requires_asset: true,
  accepts_external_url: false,
  status: { running: true, model_id: "cosmos-predict2.5", port: 18725, asset_path: "/tmp/out.mp4", url: "http://127.0.0.1:18725/", external: false, params: {} },
});
assert.ok(draftCardHtml.includes('value="/tmp/user-edited.mp4"'));

// 6. updateJobLogs accepts bare array, removes stale muted placeholder, and updates past 200 lines.
const logPanel = getEl("jobLogPanel");
state.logStreamFilter = "all";
state.detailLogCount = 0;
state.detailLogTotal = 0;
logPanel.dataset.logFilter = "all";
logPanel.dataset.activeFilter = "all";
logPanel.innerHTML = '<span class="muted">No logs yet.</span>';
logPanel._hasMuted = true;
const initialRows = [{ stream: "stdout", text: "hello\n" }, { stream: "stderr", text: "warn\n" }];
assert.strictEqual(sandbox.updateJobLogs({ logs: initialRows, logs_total: 2, status: "running" }), true);
assert.ok(!logPanel.innerHTML.includes("No logs yet."));
assert.ok(logPanel.innerHTML.includes("hello"));
assert.ok(logPanel.innerHTML.includes("warn"));
state.logStreamFilter = "stderr";
assert.strictEqual(sandbox.updateJobLogs(initialRows, { forceRebuild: true }), true);
assert.ok(!logPanel.innerHTML.includes("hello"));
assert.ok(logPanel.innerHTML.includes("warn"));

// Windowed >200 lines rebuild check.
state.logStreamFilter = "all";
const rows200 = Array.from({ length: 200 }, (_, i) => ({ stream: "stdout", text: `line-${i}\n` }));
sandbox.updateJobLogs({ logs: rows200, logs_total: 200, status: "running" }, { forceRebuild: true });
const shifted200 = Array.from({ length: 200 }, (_, i) => ({ stream: "stdout", text: `line-${i + 5}\n` }));
assert.strictEqual(sandbox.updateJobLogs({ logs: shifted200, logs_total: 205, status: "running" }), true);
assert.ok(logPanel.innerHTML.includes("line-204"));
assert.strictEqual(state.detailLogTotal, 205);

// 7. refineFromRecord populates standard/dynamic fields and keeps only unmapped kwargs in callJson/loadJson.
(async () => {
  const dynNoTiled = makeElement("inferDyn_no_tiled", "input");
  dynNoTiled.type = "checkbox";
  dynNoTiled.dataset = { inferFieldId: "no-tiled", target: "call_kwargs", kind: "boolean", required: "false" };
  sandbox.document.querySelectorAll = (sel) => {
    if (sel === "[data-infer-field-id]") return [dynNoTiled];
    return [];
  };
  state.models = [
    {
      id: "wow",
      name: "WoW",
      category: "World",
      workload_type: "world",
      backend: "from_pretrained",
      supports_from_pretrained: true,
      supports_api_init: false,
      supports_attention_backend: false,
      default_variant_id: "default",
      default_task_id: "official",
      runtime_options: {},
      variants: [{ variant_id: "default", label: "Default", model_ref: "wow-ref", call_kwargs: { steps: 50, num_frames: 41, seed: 42 }, load_kwargs: {} }],
      tasks: [{
        task_id: "official",
        label: "Official",
        inputs: [
          { field_id: "prompt", target: "prompt", default: "default prompt" },
          { field_id: "input_path", target: "input_path", default: "/tmp/default.png" },
          { field_id: "steps", target: "call_kwargs", kind: "integer", default: 50 },
          { field_id: "num_frames", target: "call_kwargs", kind: "integer", default: 41 },
          { field_id: "seed", target: "call_kwargs", kind: "integer", default: 42 },
          { field_id: "no_tiled", target: "call_kwargs", kind: "boolean", default: false },
        ],
        outputs: [],
      }],
    },
    {
      id: "cogvideox_5b_t2v",
      name: "CogVideoX",
      category: "Video",
      workload_type: "t2v",
      default_variant_id: "cogvideox_5b_t2v",
      default_task_id: "default",
      runtime_options: {},
      variants: [
        { variant_id: "cogvideox_5b_t2v", label: "5B T2V", workload_type: "t2v" },
        { variant_id: "cogvideox_5b_i2v", label: "5B I2V", workload_type: "i2v" },
      ],
      tasks: [{ task_id: "default", label: "Default", inputs: [{ field_id: "input_path", target: "input_path" }], outputs: [] }],
    },
  ];
  sandbox.fetch = async (url) => {
    const text = String(url);
    if (text.includes("/api/jobs/studio-00001")) {
      return {
        ok: true,
        json: async () => ({
          id: "studio-00001",
          model_id: "wow",
          status: "completed",
          metadata: {
            variant_id: "default",
            task_profile_id: "official",
            prompt: "prior prompt",
            device: "cuda",
            request: {
              call_kwargs: { steps: 30, num_frames: 25, seed: 7, no_tiled: true, custom_extra: "keep-me" },
              load_kwargs: { custom_load: 99 },
            },
          },
          result: { preview_video: "/tmp/out.mp4" },
        }),
      };
    }
    return { ok: true, json: async () => ({}) };
  };
  await sandbox.refineFromRecord("studio-00001", "");
  assert.strictEqual(getEl("steps").value, "30");
  assert.strictEqual(getEl("numFrames").value, "25");
  assert.strictEqual(getEl("seed").value, "7");
  assert.strictEqual(dynNoTiled.checked, true);
  assert.deepStrictEqual(JSON.parse(getEl("callJson").value), { custom_extra: "keep-me" });
  assert.deepStrictEqual(JSON.parse(getEl("loadJson").value), { custom_load: 99 });

  // 8. populateModelSelect matches variant-level workload_type override.
  getEl("workloadType").value = "i2v";
  sandbox.populateModelSelect();
  assert.ok(getEl("modelSelect").innerHTML.includes("cogvideox_5b_t2v"));

  // 9. renderDetail ignores stale async response when activeJob changed mid-flight.
  state.jobs = [
    { id: "job-a", status: "completed", model_name: "A", job_type: "inference", logs: [] },
    { id: "job-b", status: "completed", model_name: "B", job_type: "inference", logs: [] },
  ];
  state.activeJob = "job-a";
  state.detailRenderKey = "";
  let releaseJobA;
  sandbox.fetch = (url) => {
    if (String(url).includes("job-a")) {
      return new Promise(resolve => {
        releaseJobA = () => resolve({
          ok: true,
          json: async () => ({ id: "job-a", status: "completed", model_name: "ModelA", job_type: "inference", logs: [], logs_total: 0 }),
        });
      });
    }
    return Promise.resolve({
      ok: true,
      json: async () => ({ id: "job-b", status: "completed", model_name: "ModelB", job_type: "inference", logs: [], logs_total: 0 }),
    });
  };
  const pendingA = sandbox.renderDetail();
  state.activeJob = "job-b";
  await sandbox.renderDetail();
  assert.ok(getEl("jobDetail").innerHTML.includes("ModelB"));
  releaseJobA();
  await pendingA;
  assert.ok(getEl("jobDetail").innerHTML.includes("ModelB"));
  assert.ok(!getEl("jobDetail").innerHTML.includes("ModelA"));

  // 10. initializeWorkspace awaits loadSettings() before loadModels().
  const order = [];
  let releaseSettings;
  sandbox.fetch = (url) => {
    const text = String(url);
    if (text.includes("/api/settings")) {
      return new Promise(resolve => {
        releaseSettings = () => {
          order.push("settings-resolved");
          resolve({ ok: true, json: async () => ({ fps: 16, num_frames: 81, height: 720, width: 1280, num_inference_steps: 30, guidance_scale: 7.5, seed: -1, device: "cuda", attention_backend: "auto", backend: "auto" }) });
        };
      });
    }
    if (text.includes("/api/models")) {
      order.push("models-requested");
      return Promise.resolve({ ok: true, json: async () => state.models });
    }
    if (text.includes("/api/jobs")) {
      return Promise.resolve({ ok: true, json: async () => [] });
    }
    return Promise.resolve({ ok: true, json: async () => ({ benchmarks: [], metrics: [], models: [], examples: [] }) });
  };
  const initPromise = sandbox.initializeWorkspace();
  await new Promise(r => setTimeout(r, 10));
  assert.deepStrictEqual(order, []);
  releaseSettings();
  await initPromise;
  assert.deepStrictEqual(order, ["settings-resolved", "models-requested"]);
})().catch(err => { console.error(err); process.exit(1); });
"""

    result = subprocess.run([node_bin, "-e", script, str(html_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"JS behavior test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
