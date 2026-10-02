from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "fastapi" dependency at import time; skip when it is unavailable.
pytest.importorskip("fastapi")

import json
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

import worldfoundry.studio.serving.workspace as workspace_app
from worldfoundry.studio.serving.workspace import WORKSPACE_HTML, create_app


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
    assert 'data-view="training">Training' in WORKSPACE_HTML
    assert 'new Set(["inference", "evaluation", "training"])' in WORKSPACE_HTML
    assert 'id="inferDynamicFields"' in WORKSPACE_HTML
    assert "INFER_TASK_FIELD_CONTROLS" in WORKSPACE_HTML
    assert 'id="trainingDevices"' in WORKSPACE_HTML
    assert 'id="trainingGpusPerNode"' in WORKSPACE_HTML
    assert 'id="trainingSpSize"' in WORKSPACE_HTML
    assert 'id="trainingDataRoot"' in WORKSPACE_HTML
    assert 'id="trainingCkptRoot"' in WORKSPACE_HTML
    assert 'id="visualizerGrid"' in WORKSPACE_HTML
    assert 'id="artifactList"' in WORKSPACE_HTML
    assert 'id="evalPreset"' in WORKSPACE_HTML
    assert 'id="evalDatasetRoot"' in WORKSPACE_HTML
    assert 'id="evalDatasetManifest"' in WORKSPACE_HTML
    assert "function artifactCard(row)" in WORKSPACE_HTML
    assert "function renderEvaluationSummary(result)" in WORKSPACE_HTML
    assert "function applyEvaluationPreset()" in WORKSPACE_HTML
    assert "function trainingEnvOverrides()" in WORKSPACE_HTML
    assert "function trainingExtraArgs()" in WORKSPACE_HTML
    assert "function trainingDefaultSpSize()" in WORKSPACE_HTML
    assert "function syncTrainingRuntimeDefaults()" in WORKSPACE_HTML


def test_workspace_auth_token_is_only_sent_to_same_origin_http_urls() -> None:
    assert "function isSameOriginHttpUrl(raw)" in WORKSPACE_HTML
    assert 'url.origin === window.location.origin' in WORKSPACE_HTML
    assert '["http:", "https:"].includes(url.protocol)' in WORKSPACE_HTML
    assert "studioAuthToken && isSameOriginHttpUrl(url)" in WORKSPACE_HTML
    assert 'headers.set("Authorization", `Bearer ${studioAuthToken}`)' in WORKSPACE_HTML


def test_workspace_rejects_removed_visualizer(monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_app.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("a removed visualizer must not start a process"),
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

    cogvideox_2b = models["cogvideox_2b_t2v"]["links"]
    assert cogvideox_2b["github"] == "https://github.com/THUDM/CogVideo"

    fantasyworld = models["fantasyworld-wan21"]["links"]
    assert fantasyworld["github"] == "https://github.com/Fantasy-AMAP/fantasy-world"
    assert fantasyworld["project"] == "https://fantasy-amap.github.io/fantasy-world/"
    assert fantasyworld["paper"] == "https://arxiv.org/abs/2509.21657"

    voyager = models["hunyuan-world-voyager"]["links"]
    assert voyager["github"] == "https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager"

    sana = models["sana-600m-512px"]["links"]
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
    assert worldplay_fields["torchrun_nproc_per_node"]["default"] == 8
    assert worldplay_fields["torchrun_nproc_per_node"]["target"] == "load_kwargs"
    assert "N_INFERENCE_GPU" in worldplay_fields["torchrun_nproc_per_node"]["description"]

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
    assert matrix3_fields["prompt"]["default"] == "A colorful, animated cityscape with a gas station and various buildings."
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

    cosmos = models["cosmos-predict2p5"]
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

    wan2 = models["wan-2p2"]
    wan2_fields = {item["field_id"]: item for item in wan2["tasks"][0]["inputs"]}
    assert "anthropomorphic cats" in wan2_fields["prompt"]["default"]

    hy15_t2v = models["hunyuanvideo-1.5-t2v"]
    hy15_t2v_fields = {item["field_id"]: item for item in hy15_t2v["tasks"][0]["inputs"]}
    assert hy15_t2v_fields["prompt"]["default"] == "A cat walks on a snowy street, cinematic, high quality."
    assert hy15_t2v_fields["frames"]["default"] == 9
    assert hy15_t2v_fields["steps"]["default"] == 8
    assert hy15_t2v_fields["resolution"]["default"] == "480p"
    assert hy15_t2v_fields["nproc-per-node"]["default"] == 8

    hy15_i2v = models["hunyuanvideo-1.5-i2v"]
    hy15_i2v_fields = {item["field_id"]: item for item in hy15_i2v["tasks"][0]["inputs"]}
    assert hy15_i2v["workload_type"] == "i2v"
    assert hy15_i2v_fields["input_path"]["default"].endswith("test_cases/hunyuanvideo_i2v/0.jpg")
    assert hy15_i2v_fields["frames"]["default"] == 9
    assert hy15_i2v_fields["enable-step-distill"]["default"] is True

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
    task = workspace_app._entry_inference_spec(entry).task()
    workspace_app._validate_inference_payload(
        entry,
        task,
        workspace_app.JobCreateRequest(
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
    task = workspace_app._entry_inference_spec(entry).task()
    workspace_app._validate_inference_payload(
        entry,
        task,
        workspace_app.JobCreateRequest(
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
    task = workspace_app._entry_inference_spec(entry).task()
    workspace_app._validate_inference_payload(
        entry,
        task,
        workspace_app.JobCreateRequest(
            model_id="lingbot-world",
            load_kwargs={"cuda_visible_devices": "0,1,2,3"},
            call_kwargs={"python_executable": "/tmp/worldfoundry-python"},
        ),
    )


def test_workspace_builds_official_astra_and_neoverse_run_kwargs() -> None:
    _, astra_kwargs = workspace_app._inference_run_kwargs(
        workspace_app.JobCreateRequest(
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

    _, neoverse_kwargs = workspace_app._inference_run_kwargs(
        workspace_app.JobCreateRequest(
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
    spec = workspace_app._entry_inference_spec(entry)
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
    monkeypatch.setattr(workspace_app, "_entry_inference_spec", lambda _entry: patched_spec)

    resolved_model_ref = "/mnt/persistent/ckpts/Yuppie1204--NeoVerse"
    _, run_kwargs = workspace_app._inference_run_kwargs(
        workspace_app.JobCreateRequest(
            model_id="neoverse",
            model_ref=resolved_model_ref,
        )
    )

    assert run_kwargs["model_ref"] == resolved_model_ref
    assert json.loads(run_kwargs["load_kwargs_text"])["model_path"] == resolved_model_ref


def test_workspace_preserves_explicit_neoverse_model_path(monkeypatch) -> None:
    entry = workspace_app.find_entry("neoverse")
    spec = workspace_app._entry_inference_spec(entry)
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
    monkeypatch.setattr(workspace_app, "_entry_inference_spec", lambda _entry: patched_spec)

    custom_model_path = "/custom/checkpoints/neoverse-quantized"
    _, run_kwargs = workspace_app._inference_run_kwargs(
        workspace_app.JobCreateRequest(
            model_id="neoverse",
            model_ref="/mnt/persistent/ckpts/Yuppie1204--NeoVerse",
            load_kwargs={"model_path": custom_model_path},
        ),
        validate=False,
    )

    assert json.loads(run_kwargs["load_kwargs_text"])["model_path"] == custom_model_path


def test_workspace_builds_cosmos_prompt_only_smoke_run_kwargs() -> None:
    _, cosmos_kwargs = workspace_app._inference_run_kwargs(
        workspace_app.JobCreateRequest(
            model_id="cosmos-predict2p5",
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
    _, cameractrl_kwargs = workspace_app._inference_run_kwargs(
        workspace_app.JobCreateRequest(
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
    call_kwargs, load_kwargs = workspace_app._merge_common_params(
        gamecraft_entry,
        workspace_app.JobCreateRequest(
            model_id="hunyuan-game-craft",
            params={"cpu_offload": True},
        ),
    )
    assert call_kwargs == {}
    assert load_kwargs["cpu_offload"] is True

    matrix_entry = workspace_app.find_entry("matrix-game-2")
    _, matrix_load_kwargs = workspace_app._merge_common_params(
        matrix_entry,
        workspace_app.JobCreateRequest(
            model_id="matrix-game-2",
            params={"torch_compile": True},
        ),
    )
    assert matrix_load_kwargs["torch_compile"] is True


def test_workspace_maps_common_params_to_model_specific_aliases() -> None:
    gamecraft_entry = workspace_app.find_entry("hunyuan-game-craft")
    call_kwargs, _ = workspace_app._merge_common_params(
        gamecraft_entry,
        workspace_app.JobCreateRequest(
            model_id="hunyuan-game-craft",
            params={"num_inference_steps": 12, "guidance_scale": 1.5},
        ),
    )
    assert call_kwargs["infer_steps"] == 12
    assert call_kwargs["cfg_scale"] == 1.5

    lingbot_entry = workspace_app.find_entry("lingbot-world")
    call_kwargs, _ = workspace_app._merge_common_params(
        lingbot_entry,
        workspace_app.JobCreateRequest(
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


def test_workspace_evaluation_smoke_preset_and_artifact_link() -> None:
    client = TestClient(create_app())
    catalog = client.get("/api/evaluation/catalog").json()
    preset = next(item for item in catalog["examples"] if item["id"] == "existing-results-validation")
    assert Path(preset["results_path"]).is_file()
    assert preset["metrics"] == ["artifact_count", "required_artifacts_present"]
    assert preset["required_artifacts"] == ["generated_video"]

    output_dir = Path(workspace_app.MANAGER.workspace_root) / "evaluations" / "pytest-eval-validation"
    response = client.post(
        "/api/jobs",
        json={
            "job_type": "evaluation",
            "eval_mode": preset["eval_mode"],
            "benchmark_id": preset["benchmark_id"],
            "model_id": preset["model_id"],
            "dataset_id": preset["dataset_id"],
            "results_path": preset["results_path"],
            "output_dir": str(output_dir),
            "metrics": preset["metrics"],
            "required_artifacts": preset["required_artifacts"],
        },
    )
    response.raise_for_status()
    job = _wait_for_job(client, response.json()["id"])
    assert job["status"] == "completed"
    assert job["result"]["artifact_count"] == 1

    artifact_response = client.get("/api/artifacts/file", params={"path": job["result"]["scorecard_path"]})
    assert artifact_response.status_code == 200
    assert artifact_response.json()["schema_version"] == "worldfoundry-scorecard"


def test_workspace_api_builds_real_training_plan_job(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/jobs",
        json={
            "job_type": "training",
            "training_target_id": "wan-action2v",
            "training_stage_id": "ar-teacher-forcing",
            "output_dir": str(tmp_path / "train_out"),
            "plan_only": True,
        },
    )

    response.raise_for_status()
    job = _wait_for_job(client, response.json()["id"])
    assert job["status"] == "completed"
    assert job["job_type"] == "training"
    assert job["result"]["plan_only"] is True
    assert job["result"]["training_plan"]["model_id"] == "wan-action2v"
    assert "worldfoundry.training.visual_generation.wan.train" in job["result"]["training_plan"]["command"]
    assert Path(job["result"]["plan_path"]).is_file()
    assert Path(job["result"]["result_path"]).is_file()

    artifact_paths = {row["path"] for row in client.get("/api/artifacts").json() if row["job_id"] == job["id"]}
    assert job["result"]["plan_path"] in artifact_paths
    assert job["result"]["result_path"] in artifact_paths


def test_workspace_training_job_uses_concrete_command_when_not_plan_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    class FakeStdout:
        def __init__(self) -> None:
            self.rows = ["trainer started\n"]

        def readline(self) -> str:
            return self.rows.pop(0) if self.rows else ""

        def read(self) -> str:
            return ""

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            commands.append(tuple(command))
            self.stdout = FakeStdout()
            self._exit_code = None

        def poll(self):
            if not self.stdout.rows:
                self._exit_code = 0
            return self._exit_code

        def wait(self) -> int:
            self._exit_code = 0
            return 0

        def terminate(self) -> None:
            self._exit_code = -15

        def kill(self) -> None:
            self._exit_code = -9

    def fake_select(readable, _writeable, _errors, _timeout):
        stream = readable[0]
        return ([stream], [], []) if stream.rows else ([], [], [])

    monkeypatch.setattr(workspace_app.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(workspace_app.select, "select", fake_select)
    monkeypatch.setattr(
        workspace_app,
        "_training_asset_report",
        lambda plan: {"model_id": plan.target.id, "stage_id": plan.stage.id, "ok": True, "checks": []},
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/jobs",
        json={
            "job_type": "training",
            "training_target_id": "wan-action2v",
            "training_stage_id": "ar-teacher-forcing",
            "output_dir": str(tmp_path / "train_out"),
            "plan_only": False,
        },
    )

    response.raise_for_status()
    job = _wait_for_job(client, response.json()["id"])
    assert job["status"] == "completed"
    assert job["result"]["plan_only"] is False
    assert commands
    assert "worldfoundry.training.visual_generation.wan.train" in commands[0]
    assert Path(job["result"]["plan_path"]).is_file()
    assert Path(job["result"]["result_path"]).is_file()


def test_workspace_training_job_fails_fast_on_asset_check(tmp_path: Path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("training subprocess should not start when assets are missing")

    monkeypatch.setattr(workspace_app.subprocess, "Popen", fail_if_called)
    monkeypatch.setattr(
        workspace_app,
        "_training_asset_report",
        lambda plan: {
            "model_id": plan.target.id,
            "stage_id": plan.stage.id,
            "ok": False,
            "checks": [
                {
                    "name": "wan-data-path",
                    "ok": False,
                    "category": "dataset",
                    "path": str(tmp_path / "missing"),
                    "detail": "dataset path does not exist",
                }
            ],
        },
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/jobs",
        json={
            "job_type": "training",
            "training_target_id": "wan-action2v",
            "training_stage_id": "bidirectional-sft",
            "output_dir": str(tmp_path / "train_out"),
            "plan_only": False,
        },
    )

    response.raise_for_status()
    job = _wait_for_job(client, response.json()["id"])
    assert job["status"] == "failed"
    assert "asset check failed" in job["error"]
    result_path = tmp_path / "train_out" / "training_result.json"
    assert result_path.is_file()
