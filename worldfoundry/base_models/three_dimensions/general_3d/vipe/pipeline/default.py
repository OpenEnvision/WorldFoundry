# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Module for base_models -> three_dimensions -> general_3d -> vipe -> pipeline -> default.py functionality."""

import logging
import pickle
from pathlib import Path

import torch
from omegaconf import DictConfig

from worldfoundry.base_models.three_dimensions.general_3d.vipe.slam.system import SLAMOutput, SLAMSystem
from worldfoundry.base_models.three_dimensions.general_3d.vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils import io
from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.cameras import CameraType
from worldfoundry.studio.visualization.plugins.scene3d.projection import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from worldfoundry.base_models.three_dimensions.general_3d.vipe.adaptive_models import AdaptiveDepthModels
from .processors import (
    AdaptiveDepthProcessor,
    GeoCalibIntrinsicsProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
)

logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    """Default annotation pipeline implementation."""
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        """Init.

        Args:
            init: The init.
            slam: The slam.
            post: The post.
            output: The output.

        Returns:
            The return value.
        """
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)
        self._geocalib_model = None
        self._slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        self._adaptive_depth_models: AdaptiveDepthModels | None = None

    def set_output_path(self, path: str | Path) -> None:
        """Point a reused pipeline at the next video's artifact directory."""
        self.out_path = Path(path)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        """Helper function to add init processors.

        Args:
            video_stream: The video stream.

        Returns:
            The return value.
        """
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        intrinsics_processor = GeoCalibIntrinsicsProcessor(
            video_stream,
            camera_type=self.camera_type,
            model=self._geocalib_model,
        )
        self._geocalib_model = intrinsics_processor.model
        init_processors.append(intrinsics_processor)
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        """Helper function to add post processors.

        Args:
            view_idx: The view idx.
            video_stream: The video stream.
            slam_output: The slam output.

        Returns:
            The return value.
        """
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            if depth_align_model.startswith("mvd_"):
                post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
            else:
                processor = AdaptiveDepthProcessor(
                    slam_output,
                    view_idx,
                    depth_align_model,
                    shared_models=self._adaptive_depth_models,
                    metric_depth=getattr(self._slam_pipeline, "metric_depth", None),
                )
                self._adaptive_depth_models = processor.shared_models()
                post_processors.append(processor)
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        """Run.

        Args:
            video_data: The video data.

        Returns:
            The return value.
        """
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_output = self._slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Dumping artifacts for all views in the streams
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, output_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
