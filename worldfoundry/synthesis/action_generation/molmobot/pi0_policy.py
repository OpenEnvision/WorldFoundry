"""Inference-only MolmoBot-Pi0 adapter without MolmoSpaces simulator imports.

The input transform and first-action decode follow the pinned upstream
PiJointPosPolicy (allenai/MolmoBot 33c0ca77). The official class imports the
entire simulator and fetches remote assets at import time; neither is needed
to run its OpenPI checkpoint on an offline observation.
"""

from __future__ import annotations

import dataclasses
import filecmp
import pickle
from pathlib import Path
from typing import Any


EXO_INPUT_KEY = "observation/exterior_image_1_left"
WRIST_INPUT_KEY = "observation/wrist_image_left"


def _resize_crop_pad(image: Any) -> Any:
    """Match upstream resize_with_crop(360, 640), then pad(224, 224)."""
    import numpy as np
    from PIL import Image

    source = Image.fromarray(np.asarray(image))
    width, height = source.size
    scale = max(360 / height, 640 / width)
    scaled_h, scaled_w = int(height * scale), int(width * scale)
    resized = source.resize((scaled_w, scaled_h), resample=Image.BILINEAR)
    left, top = (scaled_w - 640) // 2, (scaled_h - 360) // 2
    cropped = resized.crop((left, top, left + 640, top + 360))
    ratio = max(640 / 224, 360 / 224)
    padded_h, padded_w = int(360 / ratio), int(640 / ratio)
    content = cropped.resize((padded_w, padded_h), resample=Image.BILINEAR)
    canvas = Image.new(content.mode, (224, 224), 0)
    canvas.paste(content, (max(0, int((224 - padded_w) / 2)), max(0, int((224 - padded_h) / 2))))
    return np.asarray(canvas)


class Pi0JointPositionPolicy:
    """Load the official Pi0 config and infer one joint-position action."""

    def __init__(self, checkpoint_dir: str, *, device: str, camera_keys: tuple[str, str]) -> None:
        # Unpickling is only reached after runtime.py verifies the exact
        # train_config.pkl SHA-256 supplied in the selected runtime variant.
        import molmobot_pi0.config_openpi as _molmobot_config  # noqa: F401
        from openpi.models.pi0_config import Pi0Config
        from openpi.training.config import TrainConfig

        self.checkpoint_dir = Path(checkpoint_dir)
        with (self.checkpoint_dir / "assets" / "train_config.pkl").open("rb") as stream:
            train_config = pickle.load(stream)
        if not isinstance(train_config, TrainConfig) or not isinstance(train_config.model, Pi0Config):
            raise TypeError("MolmoBot-Pi0 checkpoint does not contain an OpenPI Pi0 TrainConfig.")
        self.train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
        )
        self.device = device
        self.camera_keys = camera_keys
        self.model: Any = None

    def prepare_model(self) -> None:
        if self.model is not None:
            return
        import openpi.models_pytorch.transformers_replace as replacements
        import transformers
        from openpi.policies.policy_config import create_trained_policy
        import numpy as np

        source = Path(replacements.__path__[0])
        target = Path(transformers.__path__[0])
        if transformers.__version__ != "4.53.2" or any(
            not (target / path.relative_to(source)).is_file()
            or not filecmp.cmp(path, target / path.relative_to(source), shallow=False)
            for path in source.rglob("*.py")
        ):
            raise RuntimeError(
                "MolmoBot-Pi0 requires OpenPI's Transformers 4.53.2 overrides; "
                "run python -m worldfoundry.synthesis.action_generation.molmobot.openpi_setup "
                "inside the dedicated environment."
            )
        self.model = create_trained_policy(
            self.train_config,
            self.checkpoint_dir,
            pytorch_device=self.device if ":" in self.device else None,
        )
        self.model.infer({
            EXO_INPUT_KEY: np.zeros((224, 224, 3), dtype=np.uint8),
            WRIST_INPUT_KEY: np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "place the red block on the green block",
        })

    def reset(self) -> None:
        """Each WorldFoundry call infers a new first action; no episode buffer."""

    def prepare_input(self, observation: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        exo, wrist = self.camera_keys
        return {
            EXO_INPUT_KEY: _resize_crop_pad(observation[exo]),
            WRIST_INPUT_KEY: _resize_crop_pad(observation[wrist]),
            "observation/joint_position": np.asarray(observation["qpos"]["arm"]),
            "observation/gripper_position": np.asarray(observation["qpos"]["gripper"]),
            "prompt": observation["task"],
        }

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        self.prepare_model()
        output = np.asarray(self.model.infer(self.prepare_input(observation))["actions"])
        if output.ndim != 2 or output.shape[1] != 8 or output.shape[0] < 1:
            raise RuntimeError(f"MolmoBot-Pi0 expected an action chunk of shape (T, 8), got {output.shape}.")
        first = output[0]
        return {
            "arm": first[:7].reshape(7),
            "gripper": np.clip(first[7], 0.0, 1.0) * np.array([255.0]),
        }


__all__ = ["Pi0JointPositionPolicy"]
