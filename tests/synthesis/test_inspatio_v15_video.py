"""Real video I/O contracts, run in the model environment with FFmpeg/OpenCV."""


import json

import cv2
import numpy as np
import pytest
import torch

from worldfoundry.synthesis.visual_generation.inspatio_world.v15_io import iter_video_chunks, video_writer
from worldfoundry.synthesis.visual_generation.inspatio_world.v15_scene import stage_direct_input, video_metadata


def read_rgb(path):
    reader = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        reader.release()
    return np.stack(frames)


def test_bounded_decode_pads_last_frame_and_image_source_repeats_first(tmp_path):
    path = tmp_path / "source.mp4"
    frames = np.stack([np.full((16, 16, 3), index * 40, dtype=np.uint8) for index in range(5)])
    with video_writer(path, 16, 16, 16, depth=True) as writer:
        for frame in frames:
            writer.write(frame.tobytes())
    np.testing.assert_array_equal(read_rgb(path), frames)
    chunks = list(iter_video_chunks(path, 5, 9, video_size=(16, 16)))
    assert [chunk.shape[0] for chunk in chunks] == [1, 4, 4]
    decoded = torch.cat(chunks).permute(0, 2, 3, 1).numpy()
    np.testing.assert_array_equal(decoded[:5], frames)
    np.testing.assert_array_equal(decoded[5:], np.repeat(frames[-1: ], 4, axis=0))
    single = torch.cat(list(iter_video_chunks(path, 5, 9, video_size=(16, 16), repeat_first=True)))
    assert torch.all(single == 0)
    with pytest.raises(ValueError, match="shorter"):
        list(iter_video_chunks(path, 9, 9, video_size=(16, 16)))


def test_failed_encode_preserves_existing_output_and_removes_temporary(tmp_path):
    path = tmp_path / "pred.mp4"
    path.write_bytes(b"previous-valid-artifact")
    with pytest.raises(RuntimeError, match="cancelled"):
        with video_writer(path, 16, 16, 16) as writer:
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8).tobytes())
            raise RuntimeError("cancelled")
    assert path.read_bytes() == b"previous-valid-artifact"
    assert list(tmp_path.glob("*.mp4")) == [path]


@pytest.mark.parametrize("size", [(480, 832), (16, 32)])
def test_direct_video_staging_preserves_frame_cadence_and_validates_trajectory(tmp_path, size):
    height, width = size
    source = tmp_path / "source.mp4"
    with video_writer(source, 12, width, height, depth=True) as writer:
        for index in range(5):
            writer.write(np.full((height, width, 3), index * 40, dtype=np.uint8).tobytes())
    trajectory = tmp_path / "target.txt"
    np.savetxt(trajectory, np.tile(np.eye(4), (5, 1, 1)).reshape(-1, 16))
    directory = tmp_path / "prepared"
    record = stage_direct_input(directory, videos=source, prompt="room", traj_txt_path=trajectory)
    assert record["valid_frames"] == record["views"] == 5
    assert video_metadata(directory / "input/video.mp4") == (5, 12.0, (480, 832))
    assert (directory / "input/video.mp4").is_symlink() is (size == (480, 832))
    assert json.loads((directory / "scene.json").read_text())["target_intrinsics"] == "estimated_source_frame_000000_fixed"
    assert source.is_file()
    np.savetxt(trajectory, np.tile(np.eye(4), (4, 1, 1)).reshape(-1, 16))
    with pytest.raises(ValueError, match="one Tcw matrix"):
        stage_direct_input(tmp_path / "invalid", videos=source, prompt="room", traj_txt_path=trajectory)
