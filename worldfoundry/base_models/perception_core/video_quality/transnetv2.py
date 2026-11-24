"""TransNetV2 PyTorch runtime and checkpoint resolver."""

from __future__ import annotations

import os
from pathlib import Path

# Set MPS fallback before PyTorch imports.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gc
import random
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

# Silence MPS fallback warnings since we're intentionally using the fallback
warnings.filterwarnings("ignore", message=".*MPS backend.*will fall back to run on the CPU.*")


def checkpoint_path() -> Path:
    """Resolve the TransNetV2 checkpoint from the shared capability registry."""
    asset = BASE_MODEL_CAPABILITIES["wbench_transnetv2"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


class TransNetV2(nn.Module):
    # Class variable to track if auto-detection message has been shown
    _auto_detection_message_shown = False

    def __init__(
        self,
        F=16,
        L=3,
        S=2,
        D=1024,
        use_many_hot_targets=True,
        use_frame_similarity=True,
        use_color_histograms=True,
        use_mean_pooling=False,
        dropout_rate=0.5,
        use_convex_comb_reg=False,  # not supported
        use_resnet_features=False,  # not supported
        use_resnet_like_top=False,  # not supported
        frame_similarity_on_last_layer=False,
        device="auto",
        weights_path: str | os.PathLike[str] | None = None,
    ):
        super().__init__()

        # Handle device auto-detection
        if device == "auto":
            device = self._detect_best_device()

        # Warn about MPS inconsistency but honor user's explicit choice
        if device == "mps":
            print("WARNING: MPS device has numerical inconsistency issues.")
            print("   This neural network architecture has operations that fall back to CPU")
            print("   inconsistently, causing different scene detection results vs. pure CPU.")
            print("")

        self.device = torch.device(device)
        self._input_size = (27, 48, 3)

        # Enable deterministic algorithms for consistent results across devices
        self._setup_deterministic_behavior()

        # Internal memory optimization settings (always enabled, not exposed to user)
        self.memory_efficient = True  # Always enable memory optimizations

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplementedError("Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0)]
            + [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2**i) for i in range(1, L)]
        )

        self.frame_sim_layer = (
            FrameSimilarity(
                sum([(F * 2**i) * 4 for i in range(L)]),
                lookup_window=101,
                output_dim=128,
                similarity_dim=128,
                use_bias=True,
            )
            if use_frame_similarity
            else None
        )
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling

        # CRITICAL FIX: Load pre-trained weights
        self._load_pretrained_weights(weights_path)

        self.eval()
        # Keep model on requested device for performance
        self.to(self.device)

    def _setup_deterministic_behavior(self):
        """Setup deterministic behavior for consistent results across devices"""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Enable deterministic algorithms where possible
        # Note: This may impact performance but ensures consistency
        try:  # noqa: SIM105 - The handler carries an explanatory comment.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except BaseException:
            # Fallback for older PyTorch versions
            pass

        # Set deterministic behavior for CuDNN
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _load_pretrained_weights(self, weights_path: str | os.PathLike[str] | None = None):
        """Load the pre-trained model weights"""
        model_path = Path(weights_path) if weights_path is not None else checkpoint_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        # Load weights and apply to model
        state_dict = torch.load(str(model_path), map_location="cpu")  # Load to CPU first
        self.load_state_dict(state_dict)

    def _cleanup_memory(self):
        """Clean up GPU memory to prevent accumulation - but NOT during inference"""
        if self.memory_efficient:
            if str(self.device) == "mps":
                # Force MPS memory cleanup
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            elif str(self.device) == "cuda":
                # Force CUDA memory cleanup
                torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()

    @staticmethod
    def _detect_best_device():
        """
        Automatically detect the best available device
        Priority: CUDA > CPU > MPS (MPS has consistency issues)
        """
        if torch.cuda.is_available():
            return "cuda"
        # Check if MPS is available but skip it due to consistency issues
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Only show the message once per program run
            if not TransNetV2._auto_detection_message_shown:
                print("MPS device detected but not used due to numerical inconsistency issues.")
                print("   Use --device mps to explicitly enable MPS (faster but inconsistent results).")
                TransNetV2._auto_detection_message_shown = True
            return "cpu"
        else:
            return "cpu"

    def get_video_fps(self, video_path: str) -> float:
        """
        Extract FPS from video file using ffmpeg

        Args:
            video_path: Path to the video file

        Returns:
            FPS as float, defaults to 25.0 if extraction fails
        """
        try:
            import ffmpeg

            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
            if video_stream is None:
                return 25.0

            fps_str = video_stream["r_frame_rate"]
            # Handle fraction format like "25/1" or "30000/1001"
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)

            return fps
        except Exception:
            return 25.0

    @staticmethod
    def frame_to_timestamp(frame_number: int, fps: float) -> str:
        """
        Convert frame number to timestamp in ss.mmm format

        Args:
            frame_number: Frame index
            fps: Frames per second

        Returns:
            Timestamp string in format "ss.mmm"
        """
        seconds = frame_number / fps
        return f"{seconds:.3f}"

    def forward(self, inputs):
        assert (
            isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8
        ), "incorrect input type and/or shape"
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot

    def predict_raw(self, frames):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, (
            "Input shape must be [batch, frames, height, width, 3]."
        )
        single_frame_pred, all_frames_pred = self.forward(frames)
        single_frame_pred = torch.sigmoid(single_frame_pred)
        all_frames_pred = torch.sigmoid(all_frames_pred["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames, quiet=False):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, (
            "Input shape must be [frames, height, width, 3]."
        )

        def input_iterator():
            # Use original algorithm parameters - don't change for memory management
            window_size = 100
            step_size = 50  # Keep original step size for all devices

            # Original working logic
            no_padded_frames_start = 25
            no_padded_frames_end = (
                25 + step_size - (len(frames) % step_size if len(frames) % step_size != 0 else step_size)
            )

            start_frame = torch.unsqueeze(frames[0], 0)
            end_frame = torch.unsqueeze(frames[-1], 0)
            padded_inputs = torch.cat(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            batch_count = 0
            while ptr + window_size <= len(padded_inputs):
                batch = padded_inputs[ptr : ptr + window_size]
                ptr += step_size
                batch_count += 1
                yield batch[np.newaxis], batch_count

        predictions = []

        # Create progress bar only if not quiet
        pbar = None if quiet else tqdm(total=len(frames), desc="Processing frames", unit="frame")

        try:
            for batch_input, _batch_num in input_iterator():
                with torch.no_grad():  # Ensure no gradients are computed
                    single_frame_pred, all_frames_pred = self.predict_raw(batch_input)

                    # Extract the non-overlapping portion (original logic)
                    start_idx = 25
                    end_idx = 75  # 25 + 50

                    predictions.append(
                        (
                            single_frame_pred[0, start_idx:end_idx, 0].cpu().clone(),
                            all_frames_pred[0, start_idx:end_idx, 0].cpu().clone(),
                        )
                    )

                    # Update progress bar if present
                    if pbar is not None:
                        processed_frames = min(len(predictions) * 50, len(frames))
                        pbar.update(processed_frames - pbar.n)
        finally:
            # Ensure progress bar is closed
            if pbar is not None:
                pbar.close()

        # Concatenate results efficiently
        single_frame_pred = torch.cat([single_ for single_, _ in predictions], 0)
        all_frames_pred = torch.cat([all_ for _, all_ in predictions], 0)

        return single_frame_pred[: len(frames)], all_frames_pred[: len(frames)]  # remove extra padded frames

    def predict_video(self, video_fn: str, quiet=False):
        """
        Get raw frame predictions.

        This method returns raw probabilities that require post-processing.
        For processed scene boundaries, consider using detect_scenes().

        Returns:
            Tuple of (video_frames, single_frame_predictions, all_frame_predictions)
        """
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError(  # noqa: B904 - Keeps the implicit exception context.
                "For `predict_video` function `ffmpeg` needs to be installed in order to extract "
                "individual frames from video file. Install `ffmpeg` command line tool and then "
                "install python wrapper by `pip install ffmpeg-python`."
            )

        if not quiet:
            print(f"Extracting frames from {video_fn}")

        video_stream, err = (
            ffmpeg.input(video_fn)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True)
        )

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        # Use inference device (CPU) for consistent results across all devices
        video = torch.from_numpy(np.array(video, copy=True)).to(self.device)
        return (video, *self.predict_frames(video, quiet=quiet))

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def predictions_to_scenes_with_data(
        self,
        predictions: np.ndarray | torch.Tensor,
        fps: float | None = None,
        video_path: str | None = None,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Convert predictions to structured scene data with timestamps and metadata

        Args:
            predictions: Single frame predictions array/tensor
            fps: Video FPS (if None and video_path provided, will extract from video)
            video_path: Path to video file (used to extract FPS if fps not provided)
            threshold: Threshold for scene boundary detection

        Returns:
            List of dictionaries with scene information including:
            - shot_id: Scene number (1-indexed)
            - start_frame: Starting frame index
            - end_frame: Ending frame index
            - start_time: Starting timestamp (if FPS available)
            - end_time: Ending timestamp (if FPS available)
            - probability: Maximum probability in the scene
        """
        # Convert to numpy if tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()

        # Get FPS if not provided
        if fps is None and video_path is not None:
            fps = self.get_video_fps(video_path)

        # Get basic scene boundaries
        scenes = self.predictions_to_scenes(predictions, threshold)

        # Build structured data
        output_data = []
        for i, scene in enumerate(scenes):
            start_frame = int(scene[0])
            end_frame = int(scene[1])

            # Get the maximum probability in this scene range
            scene_probs = predictions[start_frame : end_frame + 1]
            max_probability = float(np.max(scene_probs)) if len(scene_probs) > 0 else 0.0

            scene_data = {
                "shot_id": i + 1,  # Start from 1
                "start_frame": start_frame,
                "end_frame": end_frame,
                "probability": max_probability,
            }

            # Add timestamps if FPS is available
            if fps is not None:
                scene_data["start_time"] = self.frame_to_timestamp(start_frame, fps)
                scene_data["end_time"] = self.frame_to_timestamp(end_frame, fps)

            output_data.append(scene_data)

        return output_data

    def analyze_video(self, video_path: str, threshold: float = 0.5, quiet: bool = False) -> dict[str, Any]:
        """
        Comprehensive video analysis with raw predictions and scene data.

        Mid-level method that returns both raw predictions and processed scenes.
        Use detect_scenes() for simple scene detection.

        Args:
            video_path: Path to video file
            threshold: Scene boundary detection threshold
            quiet: Whether to suppress progress output

        Returns:
            Dictionary containing:
            - video_frames: Raw video frames
            - single_frame_predictions: Single frame predictions tensor
            - all_frame_predictions: All frame predictions tensor
            - fps: Video FPS
            - scenes: List of scene dictionaries
            - total_scenes: Number of scenes detected
        """
        # Get video FPS
        fps = self.get_video_fps(video_path)

        # Get predictions
        video_frames, single_frame_predictions, all_frame_predictions = self.predict_video(video_path, quiet=quiet)

        # Convert predictions to numpy for scene detection
        single_frame_np = single_frame_predictions.cpu().detach().numpy()

        # Get structured scene data
        scenes = self.predictions_to_scenes_with_data(single_frame_np, fps=fps, threshold=threshold)

        return {
            "video_frames": video_frames,
            "single_frame_predictions": single_frame_predictions,
            "all_frame_predictions": all_frame_predictions,
            "fps": fps,
            "scenes": scenes,
            "total_scenes": len(scenes),
        }

    def get_scene_count(self, video_path: str, threshold: float = 0.5) -> int:
        """
        Get the number of scenes in a video.

        Args:
            video_path: Path to video file
            threshold: Scene detection threshold

        Returns:
            Number of scenes detected

        Example:
            >>> model = TransNetV2()
            >>> count = model.get_scene_count("video.mp4")
            >>> print(f"Video has {count} scenes")
        """
        scenes = self.detect_scenes(video_path, threshold)
        return len(scenes)

    def get_scene_timestamps(self, video_path: str, threshold: float = 0.5) -> list[tuple]:
        """
        Get the timestamps of scene boundaries.

        Args:
            video_path: Path to video file
            threshold: Scene detection threshold

        Returns:
            List of (start_time, end_time) tuples as floats in seconds

        Example:
            >>> model = TransNetV2()
            >>> timestamps = model.get_scene_timestamps("video.mp4")
            >>> for start, end in timestamps[:3]:
            ...     print(f"Scene: {start:.1f}s - {end:.1f}s")
        """
        scenes = self.detect_scenes(video_path, threshold)
        return [(float(scene["start_time"]), float(scene["end_time"])) for scene in scenes]

    def detect_scenes(self, video_path: str, threshold: float = 0.5) -> list[dict[str, Any]]:
        """
        Detect scene boundaries in a video.

        Primary scene detection method. Returns scene boundaries with timestamps.

        Args:
            video_path: Path to video file
            threshold: Scene boundary detection threshold (0.0-1.0, default: 0.5)

        Returns:
            List of scene dictionaries, each containing:
            - shot_id: Scene number (1-indexed)
            - start_frame: Starting frame index
            - end_frame: Ending frame index
            - start_time: Start timestamp in seconds (string)
            - end_time: End timestamp in seconds (string)
            - probability: Maximum detection probability in scene

        Example:
            >>> model = TransNetV2()
            >>> scenes = model.detect_scenes("video.mp4")
            >>> for scene in scenes[:3]:
            ...     print(f"Scene {scene['shot_id']}: {scene['start_time']}s - {scene['end_time']}s")
        """
        # Get video FPS
        fps = self.get_video_fps(video_path)

        # Get predictions
        video_frames, single_frame_predictions, all_frame_predictions = self.predict_video(video_path, quiet=True)

        # Convert predictions to numpy for scene detection
        single_frame_np = single_frame_predictions.cpu().detach().numpy()

        # Get structured scene data
        scenes = self.predictions_to_scenes_with_data(single_frame_np, fps=fps, threshold=threshold)

        return scenes


class StackedDDCNNV2(nn.Module):
    def __init__(
        self,
        in_filters,
        n_blocks,
        filters,
        shortcut=True,
        use_octave_conv=False,  # not supported
        pool_type="avg",
        stochastic_depth_drop_prob=0.0,
    ):
        super().__init__()

        if use_octave_conv:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")

        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    octave_conv=use_octave_conv,
                    activation=functional.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ]
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    x = shortcut if random.random() < self.stochastic_depth_drop_prob else x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None, octave_conv=False):  # not supported
        super().__init__()

        if octave_conv:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")

        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    def __init__(
        self,
        in_filters,
        filters,
        dilation_rate,
        separable=True,
        octave=False,  # not supported
        use_bias=True,
        kernel_initializer=None,
    ):  # not supported
        super().__init__()

        if octave:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")
        if kernel_initializer is not None:
            raise NotImplementedError("Kernel initializers are not implemented in Pytorch version of Transnet!")

        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(
                in_filters, 2 * filters, kernel_size=(1, 3, 3), dilation=(1, 1, 1), padding=(0, 1, 1), bias=False
            )
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(
        self,
        in_filters,
        similarity_dim=128,
        lookup_window=101,
        output_dim=128,
        stop_gradient=False,  # not supported
        use_bias=False,
    ):
        super().__init__()

        if stop_gradient:
            raise NotImplementedError("Stop gradient not implemented in Pytorch version of Transnet!")

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(
            similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2]
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super().__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(
            0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device)
        )

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(
            similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2]
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities
