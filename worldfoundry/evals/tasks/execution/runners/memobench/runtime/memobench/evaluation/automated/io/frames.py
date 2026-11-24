import os

import cv2
import numpy as np


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


class FrameReader:
    """Reads a directory of zero-padded PNG frames (e.g. 00000.png, 00001.png, ...)."""

    def __init__(self, frames_dir: str, max_side: int = 640):
        self.frames_dir = frames_dir
        self.max_side = max_side

        files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".png"))
        if not files:
            raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
        self._files = files
        self._n = len(files)

    @property
    def num_frames(self) -> int:
        return self._n

    def get(self, idx: int) -> np.ndarray:
        idx = int(max(0, min(idx, self._n - 1)))
        path = os.path.join(self.frames_dir, self._files[idx])
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Failed to read {path}")
        return _resize_max_side(frame, self.max_side)

    def iter_indices(self, start: int, end: int, step: int):
        start = max(0, start)
        end = min(self._n - 1, end)
        yield from range(start, end + 1, step)


class VideoReader:
    """Reads frames from a video file (mp4 etc.) on demand.

    Keeps the VideoCapture open and only seeks when access is non-sequential,
    so iterating all frames in order is efficient.
    Same interface as FrameReader: num_frames, get(idx), iter_indices().
    """

    def __init__(self, video_path: str, max_side: int = 640):
        self.video_path = video_path
        self.max_side = max_side
        self._cap = cv2.VideoCapture(video_path)
        self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._pos = -1
        if self._n <= 0:
            self._cap.release()
            raise FileNotFoundError(f"Cannot read video or frame count is 0: {video_path}")

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()

    @property
    def num_frames(self) -> int:
        return self._n

    def get(self, idx: int) -> np.ndarray:
        idx = int(max(0, min(idx, self._n - 1)))
        if idx != self._pos + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {self.video_path}")
        self._pos = idx
        return _resize_max_side(frame, self.max_side)

    def iter_indices(self, start: int, end: int, step: int):
        start = max(0, start)
        end = min(self._n - 1, end)
        yield from range(start, end + 1, step)
