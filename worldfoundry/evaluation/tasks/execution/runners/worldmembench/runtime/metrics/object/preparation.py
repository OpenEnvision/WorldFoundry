# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Read a frozen source mask and target description, without generating annotations."""

import json
from pathlib import Path

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError

SOURCE_PATH_FIELDS = ("condition_image", "target_object_mask_first_frame")


def source_path(sample: dict, field: str) -> str:
    value = (sample.get("annotations") or {}).get(field)
    if not isinstance(value, str) or not value.strip():
        raise DataInvalidError(f"missing annotations.{field}")
    path = Path(value).expanduser().resolve()
    return str(path)


def load_source(image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    try:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        raw = np.load(mask_path, allow_pickle=False)
    except Exception as error:
        raise DataInvalidError(f"cannot read object source image/mask: {error}") from error
    if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.shape != rgb.shape[:2]:
        raise DataInvalidError("object_mask.npy must be a 2-D array matching the source image size")
    # Accept boolean masks, 0/1 arrays, and the common 0/255 binary encoding.
    if not (np.isin(raw, [0, 1]).all() or np.isin(raw, [0, 255]).all()):
        raise DataInvalidError("object_mask.npy must be binary (bool, 0/1, or 0/255)")
    mask = raw.astype(bool)
    if not mask.any():
        raise DataInvalidError("object_mask.npy cannot be empty")
    return rgb, mask


def prepare_for_evaluation(samples: list[dict], *, expected_revisits: int | None) -> None:
    """Resolve case sidecars once; source contents are checked when scoring."""
    for sample in samples:
        if sample.get("category") not in (None, "object"):
            continue
        sample.pop("object_preparation_error", None)
        try:
            annotations = dict(sample.get("annotations") or {})
            if not all(field in annotations for field in (*SOURCE_PATH_FIELDS, "target")):
                case_dir = Path(sample["video"]).parent
                prompt = json.loads((case_dir / "prompt.json").read_text(encoding="utf-8"))
                obj = prompt["object"]
                annotations.update(
                    condition_image=str((case_dir / prompt.get("first_frame", "first_frame.jpg")).resolve()),
                    target_object_mask_first_frame=str(
                        (case_dir / obj.get("object_mask", "object_mask.npy")).resolve()
                    ),
                    target=obj["target"],
                )
            if expected_revisits is not None:
                annotations["expected_revisits"] = expected_revisits
            else:
                annotations.setdefault("expected_revisits", None)
            sample["annotations"] = annotations
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            sample["object_preparation_error"] = {
                "status": "data_invalid",
                "message": f"invalid Object prompt/annotations: {error}",
            }
