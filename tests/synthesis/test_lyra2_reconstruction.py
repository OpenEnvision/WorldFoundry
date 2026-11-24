"""Weight-free wiring checks for the in-process Lyra-2 Gaussian reconstruction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")
pytest.importorskip("addict")
pytest.importorskip("plyfile")
pytest.importorskip("scipy")

import cv2

import worldfoundry.core.media.codecs.video as video_codecs
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.model.utils import gs_renderer
from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v2
from worldfoundry.base_models.three_dimensions.point_clouds.lyra.runtime_v2 import Lyra2Representation
from worldfoundry.synthesis.visual_generation.lyra_2 import reconstruction
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference import depth_utils
from worldfoundry.synthesis.visual_generation.lyra_2.reconstruction import (
    Lyra2ReconstructionOutputs,
    reconstruct_video_gaussians,
)

NUM_FRAMES, SIZE, FPS = 6, 64, 8.0
DA3_INDICES = [0, 1, 3, 5]


def _write_video(path: Path) -> Path:
    """Write a small MP4 with ``NUM_FRAMES`` frames of ``SIZE`` x ``SIZE`` pixels."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    for idx in range(NUM_FRAMES):
        writer.write(np.full((SIZE, SIZE, 3), 20 * idx, dtype=np.uint8))
    writer.release()
    return path


def _c2w(num: int) -> torch.Tensor:
    """Return a non-degenerate camera-to-world trajectory ``(num, 4, 4)``."""
    poses = torch.eye(4).repeat(num, 1, 1)
    for idx in range(num):
        angle = 0.1 * idx
        poses[idx, :3, :3] = torch.tensor(
            [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
        )
        poses[idx, :3, 3] = torch.tensor([0.3 * idx, 0.05 * idx**2, 0.1 * idx])
    return poses


class _FakeDA3:
    """DA3 stand-in returning predictions whose Gaussians follow ``gs_down_ratio``."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.calls: list[dict[str, Any]] = []

    def eval(self) -> _FakeDA3:
        return self

    def inference(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        num = len(kwargs["image"])
        extrinsics = kwargs["extrinsics"]
        if extrinsics is None:
            extrinsics = torch.linalg.inv(_c2w(num)).numpy()
        intrinsics = np.tile(np.array([[48.0, 0.0, 32.0], [0.0, 48.0, 32.0], [0.0, 0.0, 1.0]], np.float32), (num, 1, 1))
        gaussians = None
        if kwargs["infer_gs"]:
            from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.specs import (
                Gaussians,
            )

            count = num * (SIZE // kwargs["gs_down_ratio"]) ** 2
            gaussians = Gaussians(
                means=torch.randn(1, count, 3),
                scales=torch.full((1, count, 3), 0.05),
                rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, count, 4).clone(),
                harmonics=torch.zeros(1, count, 3, 1),
                opacities=torch.full((1, count), 0.5),
            )
        return SimpleNamespace(
            extrinsics=np.asarray(extrinsics, dtype=np.float32)[:, :3],
            intrinsics=intrinsics,
            depth=np.ones((num, SIZE, SIZE), dtype=np.float32),
            gaussians=gaussians,
        )


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace model loading, ViPE, rasterization and encoding with recording fakes."""
    state = SimpleNamespace(da3=_FakeDA3(), loads=[], vipe=[], renders=[], videos=[])

    def load_da3_model(**kwargs: Any) -> _FakeDA3:
        state.loads.append(kwargs)
        return state.da3

    def run_vipe(frames: list[np.ndarray], **kwargs: Any) -> Any:
        state.vipe.append({"num_frames": len(frames), **kwargs})
        intrinsics = torch.tensor([[48.0, 48.0, 32.0, 32.0]]).repeat(len(frames), 1)
        return reconstruction._VipePoses(
            intrinsics=intrinsics,
            extrinsics_c2w=_c2w(len(frames)),
            frame_ids=torch.arange(len(frames)),
        )

    def render(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        state.renders.append(kwargs)
        views = kwargs["extrinsics"].shape[1]
        height, width = kwargs["image_shape"]
        return torch.full((1, views, 3, height, width), 0.5), torch.zeros(1, views, height, width)

    def save_video(frames: np.ndarray, path: Path, **kwargs: Any) -> None:
        state.videos.append({"shape": frames.shape, **kwargs})
        Path(path).write_bytes(b"mp4")

    monkeypatch.setattr(depth_utils, "load_da3_model", load_da3_model)
    monkeypatch.setattr(reconstruction, "_run_vipe", run_vipe)
    monkeypatch.setattr(gs_renderer, "run_renderer_in_chunk_w_trj_mode", render)
    monkeypatch.setattr(video_codecs, "save_video_h264", save_video)
    return state


def test_reconstruction_conditions_da3_on_vipe_poses_and_writes_artifacts(
    tmp_path: Path, fakes: SimpleNamespace
) -> None:
    video = _write_video(tmp_path / "generated.mp4")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")

    outputs = reconstruct_video_gaussians(
        video, tmp_path / "recon", da3_model_path_custom=checkpoint, device="cpu", da3_max_frames=4
    )

    assert fakes.loads == [
        {
            "da3_model_name": reconstruction.DEFAULT_DA3_MODEL_NAME,
            "da3_model_path_custom": str(checkpoint),
            "device": "cpu",
        }
    ]
    assert fakes.vipe[0]["num_frames"] == NUM_FRAMES and fakes.vipe[0]["fast_mode"] is True
    assert fakes.vipe[0]["overrides"][0] == "pipeline=default"

    (call,) = fakes.da3.calls
    w2c_vipe = torch.linalg.inv(_c2w(NUM_FRAMES)).numpy()
    np.testing.assert_allclose(call["extrinsics"], w2c_vipe[DA3_INDICES], atol=1e-6)
    assert call["intrinsics"].shape == (4, 3, 3)
    assert {key: call[key] for key in ("align_to_input_pose", "infer_gs", "use_aligned_pred_cam")} == {
        "align_to_input_pose": False,
        "infer_gs": True,
        "use_aligned_pred_cam": True,
    }
    assert (call["gs_down_ratio"], call["gs_ds_feature_mode"], call["gs_scale_extra_multiplier"]) == (2, True, 1.0)
    assert (call["process_res"], call["process_res_method"]) == (SIZE, "upper_bound_resize")

    root = tmp_path / "recon"
    assert outputs == Lyra2ReconstructionOutputs(
        output_dir=root,
        reconstructed_scene_path=root / "reconstructed_scene.ply",
        render_video_path=root / "gs_trajectory.mp4",
        camera_path=root / "cameras.npz",
        vipe_prediction_path=root / "vipe_predictions.npz",
    )
    for path in (
        outputs.reconstructed_scene_path,
        outputs.render_video_path,
        outputs.camera_path,
        outputs.vipe_prediction_path,
        root / ".done",
    ):
        assert path is not None and path.is_file()

    assert outputs.vipe_prediction_path is not None
    with np.load(outputs.vipe_prediction_path) as vipe:
        assert vipe["w2c_vipe"].shape == (NUM_FRAMES, 4, 4)
        assert vipe["indices_da3"].tolist() == DA3_INDICES
    with np.load(outputs.camera_path) as cameras:
        assert cameras["no_vipe"].tolist() == [0] and cameras["use_da3_render_pose"].tolist() == [1]
        # DA3 predictions equal the ViPE poses here, so the aligned render path matches them at keyframes.
        np.testing.assert_allclose(cameras["w2c_render"][DA3_INDICES], w2c_vipe[DA3_INDICES], atol=1e-4)

    (render,) = fakes.renders
    assert render["extrinsics"].shape == (1, NUM_FRAMES, 4, 4)
    assert render["image_shape"] == (SIZE, SIZE) and render["trj_mode"] == "original"
    # GS output is SIZE / 2 per side; the PLY export trims a 1-pixel border at that size.
    assert render["gaussians"].means.shape[1] == len(DA3_INDICES) * (SIZE // 2 - 2) ** 2
    assert fakes.videos == [{"shape": (NUM_FRAMES, SIZE, SIZE, 3), "fps": FPS, "crf": 18, "preset": "slow"}]

    reconstruct_video_gaussians(video, root, da3_model_path_custom=checkpoint, device="cpu")
    assert len(fakes.loads) == 1


def test_reconstruction_without_vipe_conditions_gaussians_on_da3_pose_pass(
    tmp_path: Path, fakes: SimpleNamespace
) -> None:
    video = _write_video(tmp_path / "generated.mp4")

    outputs = reconstruct_video_gaussians(
        video, tmp_path / "recon", da3_model_path_custom=None, device="cpu", no_vipe=True, render_fps=12
    )

    assert fakes.vipe == []
    pose_call, gs_call = fakes.da3.calls
    assert pose_call["extrinsics"] is None and pose_call["infer_gs"] is False
    np.testing.assert_allclose(gs_call["extrinsics"], torch.linalg.inv(_c2w(NUM_FRAMES)).numpy(), atol=1e-6)
    assert gs_call["infer_gs"] is True and gs_call["use_aligned_pred_cam"] is True
    assert outputs.vipe_prediction_path is None
    assert not (tmp_path / "recon" / "vipe_predictions.npz").exists()
    with np.load(outputs.camera_path) as cameras:
        assert cameras["no_vipe"].tolist() == [1]
        assert cameras["w2c_da3_pred"].shape == (NUM_FRAMES, 4, 4)
        assert cameras["intrinsics_render"].shape == (NUM_FRAMES, 3, 3)
    assert fakes.videos[0]["fps"] == 12


def test_representation_runs_reconstruction_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    video = tmp_path / "generated.mp4"
    video.write_bytes(b"mp4")
    output_dir = tmp_path / "reconstruction"

    def reconstruct(*args: Any, **kwargs: Any) -> Lyra2ReconstructionOutputs:
        calls.append((args, kwargs))
        root = Path(args[1])
        return Lyra2ReconstructionOutputs(
            output_dir=root,
            reconstructed_scene_path=root / "reconstructed_scene.ply",
            render_video_path=root / "gs_trajectory.mp4",
            camera_path=root / "cameras.npz",
            vipe_prediction_path=root / "vipe_predictions.npz",
        )

    monkeypatch.setattr(reconstruction, "reconstruct_video_gaussians", reconstruct)
    representation = Lyra2Representation(
        repo_root=str(tmp_path),
        device="cuda:1",
        da3_model_path_custom="/weights/recon/model.pt",
        force=True,
        render_fps=10.0,
        render_chunk_size=4,
        vipe_overrides=["pipeline.slam.optimize_intrinsics=true"],
    )

    result = representation.get_representation({"video_path": str(video), "output_dir": str(output_dir)})

    (args, kwargs) = calls[0]
    assert args == (str(video), str(output_dir))
    assert kwargs == {
        "da3_model_path_custom": "/weights/recon/model.pt",
        "da3_model_name": runtime_v2.DEFAULT_DA3_MODEL_NAME,
        "device": "cuda:1",
        "force": True,
        "no_vipe": False,
        "vipe_overrides": ["pipeline.slam.optimize_intrinsics=true"],
        "render_fps": 10.0,
        "render_chunk_size": 4,
    }
    assert result == {
        "video_path": str(video),
        "output_dir": str(output_dir),
        "reconstructed_scene_path": str(output_dir / "reconstructed_scene.ply"),
        "render_video_path": str(output_dir / "gs_trajectory.mp4"),
        "camera_path": str(output_dir / "cameras.npz"),
        "vipe_prediction_path": str(output_dir / "vipe_predictions.npz"),
    }


def test_vipe_overrides_compose_into_a_pose_only_default_config(tmp_path: Path) -> None:
    pytest.importorskip("hydra")
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.config import parse_typed_config

    config = parse_typed_config(
        "default",
        hydra_args=[*reconstruction._vipe_default_overrides(tmp_path), *reconstruction._VIPE_FAST_MODE_OVERRIDES],
    )

    pipeline = config.pipeline
    assert pipeline.init.instance is None
    assert pipeline.slam.optimize_intrinsics is False
    assert pipeline.post.depth_align_model is None
    assert (pipeline.output.save_viz, pipeline.output.save_artifacts, pipeline.output.save_slam_map) == (
        False,
        False,
        False,
    )
    assert pipeline.output.path == str(tmp_path)


def test_reconstruction_runs_in_process_on_canonical_vipe_and_da3() -> None:
    reconstruction_text = Path(reconstruction.__file__).read_text(encoding="utf-8")
    representation_text = Path(runtime_v2.__file__).read_text(encoding="utf-8")

    assert "worldfoundry.base_models.three_dimensions.general_3d.vipe" in reconstruction_text
    assert "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3" in reconstruction_text
    assert "reconstruct_video_gaussians" in representation_text
    for marker in ("subprocess", "sys.path", "sys.executable", "argparse"):
        assert marker not in reconstruction_text
        assert marker not in representation_text
