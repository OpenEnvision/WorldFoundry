# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Weight-free camera replay with OpenCV and staged synthetic pose artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from worldfoundry.evaluation.tasks.execution.runners.worldmembench import run_worldmembench_official_runner as runner

pytestmark = pytest.mark.unit


def test_native_camera_runner_scores_staged_poses_without_model_runtimes(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (24, 16))
    assert writer.isOpened()
    for _ in range(9):
        writer.write(np.zeros((16, 24, 3), dtype=np.uint8))
    writer.release()
    poses = np.repeat(np.eye(4)[None], 9, axis=0)
    angles = np.linspace(0, 0.5, 9)
    poses[:, 0, 0] = poses[:, 1, 1] = np.cos(angles)
    poses[:, 1, 0] = np.sin(angles)
    poses[:, 0, 1] = -np.sin(angles)
    poses[:, :3, 3] = np.column_stack((angles, angles**2, np.sin(angles)))
    np.savez(tmp_path / "target.npz", c2w=poses)
    np.savez(tmp_path / "da3.npz", extrinsics=np.linalg.inv(poses), frame_indices=np.arange(9))
    np.savez(tmp_path / "vipe.npz", c2w=poses, frame_indices=np.arange(9))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "item_id": "case",
                "model_id": "model_a",
                "video": "video.mp4",
                "annotations": {"camera_motion_type": "mixed"},
                "artifacts": {"target_poses": "target.npz", "da3": "da3.npz", "vipe": "vipe.npz"},
            }
        )
        + "\n"
    )
    args = runner.build_parser().parse_args(
        [
            "--run-official",
            "--input-path",
            str(manifest),
            "--device",
            "cpu",
            "--metric",
            "camera_rotation,camera_translation",
            "--python",
            sys.executable,
            "--weights-dir",
            str(tmp_path / "assets"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    scorecard, code = runner.run(args)
    assert code == 0, (tmp_path / "out/runtime_stderr.log").read_text()
    for name in ("camera_rotation", "camera_translation"):
        metric = scorecard["metrics"]["per_metric"][name]
        assert metric["raw_score"] == pytest.approx(0.0, abs=1e-6)
        assert not metric["higher_is_better"]
    assert len((tmp_path / "out/per_sample_scores.jsonl").read_text().splitlines()) == 2
