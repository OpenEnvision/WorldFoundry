"""Minimal NudeNet ONNX inference runtime."""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as ort_state

_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


def model_path() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_NUDENET_ONNX")
    if explicit:
        return Path(explicit).expanduser()
    return Path(__file__).resolve().parent / "best.onnx"


def _read_image(image: np.ndarray, target_size: int = 320):
    img_height, img_width = image.shape[:2]
    aspect = img_width / img_height
    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.

    resize_factor = math.sqrt((img_width**2 + img_height**2) / (new_width**2 + new_height**2))
    image = cv2.resize(image, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height
    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]
    image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    image = cv2.resize(image, (target_size, target_size))
    image_data = image.astype("float32") / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    return image_data, resize_factor, pad_left, pad_top


def _postprocess(output, resize_factor, pad_left, pad_top):
    outputs = np.transpose(np.squeeze(output[0]))
    boxes = []
    scores = []
    class_ids = []
    for row in outputs:
        classes_scores = row[4:]
        max_score = np.amax(classes_scores)
        if max_score < 0.2:
            continue
        class_id = int(np.argmax(classes_scores))
        x, y, w, h = row[0], row[1], row[2], row[3]
        left = int(round((x - w * 0.5 - pad_left) * resize_factor))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        top = int(round((y - h * 0.5 - pad_top) * resize_factor))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        width = int(round(w * resize_factor))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        height = int(round(h * resize_factor))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        class_ids.append(class_id)
        scores.append(max_score)
        boxes.append([left, top, width, height])

    detections = []
    for i in cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45):
        detections.append({"class": _LABELS[class_ids[i]], "score": float(scores[i]), "box": boxes[i]})
    return detections


class NudeDetector:
    def __init__(self, providers=None, model: str | os.PathLike | None = None):
        self.onnx_session = onnxruntime.InferenceSession(
            str(model_path() if model is None else model),
            providers=providers if providers else ort_state.get_available_providers(),
        )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.input_name = model_inputs[0].name

    def detect(self, image: np.ndarray):
        preprocessed_image, resize_factor, pad_left, pad_top = _read_image(image, self.input_width)
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        return _postprocess(outputs, resize_factor, pad_left, pad_top)
