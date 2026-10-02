from pathlib import Path

from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import (
    get_benchmark_generation_adapter,
)
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_prompts import (
    materialize_iworldbench_generation_requests,
    resolve_metadata_csv_path,
)


def test_materialize_resolves_released_first_frame_path(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset" / "all_pack"
    frame = data_root / "assets" / "example.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    metadata = data_root / "metadata.csv"
    metadata.write_text(
        "sample_id,first_frame_path,source_video_filename\ndiff_0001,assets/example.jpg,example.mp4\n",
        encoding="utf-8",
    )

    request = materialize_iworldbench_generation_requests(meta_csv_path=metadata)[0]

    assert request.inputs["first_frame"] == str(frame.resolve())
    assert "first_frame_path" not in request.inputs


def _write_official_layout(root: Path) -> None:
    all_pack = root / "dataset" / "all_pack"
    (all_pack / "assets").mkdir(parents=True)
    for name in ("diff-a.jpg", "diff-b.jpg", "mem.jpg", "camera-following.jpg"):
        (all_pack / "assets" / name).write_bytes(b"frame")

    inference_controls = root / "camera_trajectories" / "inference_txt"
    inference_controls.mkdir(parents=True)
    for name in ("camera_1_0_1.txt", "camera_2_0_1.txt", "memory_1.txt"):
        (inference_controls / name).write_text("control", encoding="utf-8")

    source_controls = root / "camera_trajectories" / "source_camera_txt"
    source_controls.mkdir(parents=True)
    (source_controls / "scene_camera_001.txt").write_text("source control", encoding="utf-8")

    (all_pack / "metadata.csv").write_text(
        "sample_id,task,first_frame_path,control_txt_path,source_camera_npz_path\n"
        "shared,Diff,assets/diff-a.jpg,camera_1_0_1.txt,reference-a.npz\n"
        "shared,Diff,assets/diff-b.jpg,camera_2_0_1.txt,reference-b.npz\n"
        "mem_0001,Mem,assets/mem.jpg,memory_1.txt,memory-reference.npz\n"
        "wrong,DiffExtra,assets/diff-a.jpg,camera_1_0_1.txt,wrong.npz\n",
        encoding="utf-8",
    )
    (all_pack / "camera_following_metadata.csv").write_text(
        "sample_id,task,first_frame_path,source_camera_txt_path,source_camera_npz_path\n"
        "scene_camera_001,CameraFollowing,assets/camera-following.jpg,"
        "camera_trajectories/source_camera_txt/scene_camera_001.txt,source-reference.npz\n",
        encoding="utf-8",
    )


def test_dataset_root_split_controls_and_collision_free_names(tmp_path: Path) -> None:
    _write_official_layout(tmp_path)

    metadata = resolve_metadata_csv_path(dataset_root=tmp_path, split="diff")
    requests = materialize_iworldbench_generation_requests(dataset_root=tmp_path, split="Diff")

    assert metadata == (tmp_path / "dataset" / "all_pack" / "metadata.csv").resolve()
    assert [request.sample_id for request in requests] == [
        "shared_camera_1_0_1",
        "shared_camera_2_0_1",
    ]
    assert [request.inputs["official_video_name"] for request in requests] == [
        "shared_camera_1_0_1.mp4",
        "shared_camera_2_0_1.mp4",
    ]
    assert "prompt" not in requests[0].inputs
    assert requests[0].controls == {
        "control_txt_path": str((tmp_path / "camera_trajectories" / "inference_txt" / "camera_1_0_1.txt").resolve()),
        "metadata_control_txt_path": "camera_1_0_1.txt",
        "control_column": "control_txt_path",
        "control_type": "diff",
    }
    request_payload = {"inputs": requests[0].inputs, "controls": requests[0].controls}
    assert "npz" not in repr(request_payload).lower()


def test_adapter_forwards_explicit_dataset_root_and_split(tmp_path: Path) -> None:
    _write_official_layout(tmp_path)
    adapter = get_benchmark_generation_adapter("iworld-bench")

    assert adapter is not None
    requests = adapter.materialize_requests(limit=1, dataset_root=tmp_path, split="mem")

    assert len(requests) == 1
    assert requests[0].split == "mem"
    assert requests[0].sample_id == "mem_0001_memory_1"
    assert requests[0].controls["metadata_control_txt_path"] == "memory_1.txt"


def test_camera_following_uses_source_control_without_reference_npz(tmp_path: Path) -> None:
    _write_official_layout(tmp_path)

    request = materialize_iworldbench_generation_requests(
        dataset_root=tmp_path,
        split="camera_following",
    )[0]

    assert request.sample_id == "scene_camera_001"
    assert request.inputs["official_video_name"] == "scene_camera_001.mp4"
    assert request.controls["control_column"] == "source_camera_txt_path"
    assert request.controls["control_txt_path"] == str(
        (tmp_path / "camera_trajectories" / "source_camera_txt" / "scene_camera_001.txt").resolve()
    )
    assert "npz" not in repr({"inputs": request.inputs, "controls": request.controls}).lower()
