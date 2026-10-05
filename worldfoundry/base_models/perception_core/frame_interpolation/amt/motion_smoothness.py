"""Shared AMT motion-smoothness inference for video evaluation suites."""

from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from worldfoundry.core.io import list_numbered_frame_paths
from worldfoundry.core.execution.inference_runtime import (
    is_accelerator_out_of_memory,
    resolve_inference_batch_size,
)

from .utils.build_utils import build_from_cfg
from .utils.utils import InputPadder, check_dim_and_resize, img2tensor

OUTPUT_OFFLOAD_ENV = "WORLDFOUNDRY_AMT_OUTPUT_OFFLOAD"
LEGACY_OUTPUT_OFFLOAD_ENV = "WORLDFOUNDRY_MIRABENCH_AMT_OUTPUT_OFFLOAD"
MINIMUM_MODEL_SIDE = 128


def _output_offload_mode() -> str:
    value = os.environ.get(
        OUTPUT_OFFLOAD_ENV,
        os.environ.get(LEGACY_OUTPUT_OFFLOAD_ENV, "auto"),
    ).strip().lower()
    aliases = {
        "1": "always",
        "true": "always",
        "yes": "always",
        "on": "always",
        "0": "never",
        "false": "never",
        "no": "never",
        "off": "never",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "always", "never"}:
        raise ValueError(
            f"{OUTPUT_OFFLOAD_ENV} must be one of auto, always, or never; got {value!r}"
        )
    return value


class FrameProcess:
    """Decode videos or deterministic numbered-frame directories."""

    def get_frames(self, video_path: str | Path) -> list[np.ndarray]:
        frame_list: list[np.ndarray] = []
        video = cv2.VideoCapture(str(video_path))
        try:
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            video.release()
        if not frame_list:
            raise ValueError(f"No decodable frames found in video: {video_path}")
        return frame_list

    def get_frames_from_img_folder(self, img_folder: str | Path) -> list[np.ndarray]:
        # Official AMT/VBench frame folders commonly use ``000000.png`` while
        # WorldFoundry's shared extractor emits ``frames_000001.png``.  Accept
        # both deterministic layouts without treating unrelated PNGs as frames.
        paths = list_numbered_frame_paths(img_folder, prefix="")
        if not paths:
            paths = list_numbered_frame_paths(img_folder)
        if not paths:
            raise ValueError(f"No numbered PNG frames found in directory: {img_folder}")
        frame_list: list[np.ndarray] = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to decode image frame: {path}")
            frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame_list

    @staticmethod
    def extract_frame(frame_list: Sequence[Any], start_from: int = 0) -> list[Any]:
        return list(frame_list[start_from::2])


class MotionSmoothness:
    """Score temporal smoothness with one resident AMT interpolation model."""

    def __init__(self, config: str | Path, ckpt: str | Path, device: str | torch.device):
        self.device = torch.device(device)
        self.config = str(config)
        self.ckpt = str(ckpt)
        self.niters = 1
        self.initialization()
        self.load_model()

    def load_model(self) -> None:
        network_cfg = OmegaConf.load(self.config).network
        network_name = network_cfg.name
        print(f"Loading [{network_name}] from [{self.ckpt}]...")
        self.model = build_from_cfg(network_cfg)
        # The official checkpoint contains only tensor state plus a historical
        # ``typing.OrderedDict`` container. Allowlist that exact container
        # without falling back to unrestricted pickle execution.
        with torch.serialization.safe_globals([typing.OrderedDict]):
            checkpoint = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def initialization(self) -> None:
        if self.device.type == "cuda":
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print(f"VRAM available: {self.vram_avail / 1024**2:.1f} MB")
        else:
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(0.5, dtype=torch.float32, device=self.device).view(1, 1, 1, 1)
        self.fp = FrameProcess()

    def keep_outputs_on_device(self, inputs: Sequence[torch.Tensor], output_count: int) -> bool:
        """Keep AMT outputs resident when doing so cannot crowd model working memory."""

        if self.device.type != "cuda":
            return True
        mode = _output_offload_mode()
        if mode == "always":
            return False
        if mode == "never":
            return True
        if not inputs or output_count < 1:
            return False

        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except (RuntimeError, TypeError):
            return False

        # The resident output list and the final contiguous conversion batch coexist.
        frame_bytes = inputs[0].numel() * inputs[0].element_size()
        estimated_bytes = 2 * output_count * frame_bytes
        return estimated_bytes <= int(free_bytes * 0.25)

    def _predict_midpoints(
        self,
        inputs: Sequence[torch.Tensor],
        *,
        scale: float,
        keep_outputs: bool,
    ) -> list[torch.Tensor]:
        pair_count = len(inputs) - 1
        if pair_count < 1:
            return []
        current_batch_size = resolve_inference_batch_size(
            2,
            device=self.device,
            scope="amt",
            maximum=pair_count,
        )
        offset = 0
        predictions: list[torch.Tensor] = []

        while offset < pair_count:
            count = min(current_batch_size, pair_count - offset)
            first_batch: torch.Tensor | None = None
            second_batch: torch.Tensor | None = None
            prediction_batch: torch.Tensor | None = None
            retry_batch_size: int | None = None
            try:
                if count == 1:
                    first_batch = inputs[offset].to(self.device, non_blocking=True)
                    second_batch = inputs[offset + 1].to(self.device, non_blocking=True)
                else:
                    first_batch = torch.cat(inputs[offset : offset + count], dim=0).to(
                        self.device,
                        non_blocking=True,
                    )
                    second_batch = torch.cat(inputs[offset + 1 : offset + count + 1], dim=0).to(
                        self.device,
                        non_blocking=True,
                    )
                embt = self.embt.expand(count, -1, -1, -1)
                prediction_batch = self.model(
                    first_batch,
                    second_batch,
                    embt,
                    scale_factor=scale,
                    eval=True,
                )["imgt_pred"].detach()
                if not keep_outputs:
                    prediction_batch = prediction_batch.cpu()
            except Exception as exc:
                if not is_accelerator_out_of_memory(exc) or count <= 1:
                    raise
                first_batch = None
                second_batch = None
                prediction_batch = None
                retry_batch_size = max((count + 1) // 2, 1)

            if retry_batch_size is not None:
                current_batch_size = retry_batch_size
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if prediction_batch is None or len(prediction_batch) != count:
                raise RuntimeError("AMT interpolation did not preserve the pair batch dimension")
            predictions.extend(prediction_batch.split(1, dim=0))
            offset += count
        return predictions

    @staticmethod
    def _outputs_to_images(outputs: Sequence[torch.Tensor]) -> list[np.ndarray]:
        if not outputs:
            return []
        batch = torch.cat([output.detach() for output in outputs], dim=0)
        batch.mul_(255.0)
        images = batch.permute(0, 2, 3, 1).cpu().numpy()
        if not np.isfinite(images).all():
            raise RuntimeError("AMT produced non-finite interpolation values")
        return list(images.clip(0, 255).astype(np.uint8))

    def motion_score(self, video_path: str | Path) -> float:
        path = Path(video_path)
        if path.is_dir():
            frames = self.fp.get_frames_from_img_folder(path)
        elif path.is_file():
            frames = self.fp.get_frames(path)
        else:
            raise FileNotFoundError(f"Video or frame directory not found: {path}")

        inputs = [img2tensor(frame) for frame in self.fp.extract_frame(frames)]
        if len(inputs) <= 1:
            raise ValueError(f"The number of input frames must be greater than one; got {len(inputs)}")
        inputs = check_dim_and_resize(inputs)
        height, width = inputs[0].shape[-2:]

        available_ratio = max(
            (self.vram_avail - self.anchor_memory_bias) / self.anchor_memory,
            1.0 / 256.0,
        )
        scale = min(1.0, self.anchor_resolution / (height * width) * np.sqrt(available_ratio))
        scale = 16.0 / np.floor(16.0 / np.sqrt(scale))
        if scale < 1:
            print(f"Due to limited VRAM, the video will be scaled by {scale:.2f}")

        # AMT-S builds a four-level correlation pyramid.  An effective side
        # below 128 collapses the coarsest level and produces NaNs.  Replicate
        # padding preserves the source pixels and is cropped away after
        # interpolation, unlike resizing which would alter the metric.
        minimum_input_side = int(np.ceil(MINIMUM_MODEL_SIDE / scale))
        padder = InputPadder(
            inputs[0].shape,
            int(16 / scale),
            minimum_size=minimum_input_side,
        )
        inputs = padder.pad(*inputs)
        iterations = int(self.niters)
        final_output_count = (len(inputs) - 1) * (2**iterations) + 1
        keep_outputs = self.keep_outputs_on_device(inputs, final_output_count)
        if keep_outputs:
            inputs = [frame.to(self.device, non_blocking=True) for frame in inputs]

        with torch.inference_mode():
            for _ in range(iterations):
                predictions = self._predict_midpoints(
                    inputs,
                    scale=scale,
                    keep_outputs=keep_outputs,
                )
                outputs = [inputs[0]]
                for prediction, second_input in zip(predictions, inputs[1:]):
                    outputs.extend((prediction, second_input))
                inputs = outputs

        outputs = padder.unpad(*inputs)
        interpolated_frames = self._outputs_to_images(outputs)
        vfi_score = self.vfi_score(frames, interpolated_frames)
        return (255.0 - vfi_score) / 255.0

    def vfi_score(
        self,
        original_frames: Sequence[np.ndarray],
        interpolated_frames: Sequence[np.ndarray],
    ) -> float:
        original = self.fp.extract_frame(original_frames, start_from=1)
        interpolated = self.fp.extract_frame(interpolated_frames, start_from=1)
        # With an even source-frame count the final odd frame has no following
        # even frame, so no midpoint can be predicted for it.  This one trailing
        # target is intentionally excluded; any larger mismatch is a real
        # alignment error.
        trailing_targets = len(original) - len(interpolated)
        if trailing_targets not in {0, 1}:
            raise ValueError(
                "The original and interpolated frame counts do not match: "
                f"{len(original)} != {len(interpolated)}"
            )
        if trailing_targets:
            original = original[:-trailing_targets]
        if not interpolated:
            raise ValueError("At least one interpolated frame is required")
        scores = [self.get_diff(left, right) for left, right in zip(original, interpolated)]
        return float(np.mean(scores))

    @staticmethod
    def get_diff(img1: np.ndarray, img2: np.ndarray) -> float:
        return float(np.mean(cv2.absdiff(img1, img2)))


__all__ = [
    "FrameProcess",
    "LEGACY_OUTPUT_OFFLOAD_ENV",
    "MotionSmoothness",
    "OUTPUT_OFFLOAD_ENV",
]
