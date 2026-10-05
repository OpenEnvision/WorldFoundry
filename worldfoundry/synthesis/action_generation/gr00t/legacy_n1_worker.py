"""Worker for NVIDIA's pinned original GR00T-N1 release.

Invoked by legacy_n1.py in a separate environment with Transformers 4.45.2.
The official source checkout is supplied through PYTHONPATH by the parent.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


def _write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def _install_eager_attention() -> None:
    """Use standard PyTorch attention without editing NVIDIA's release files."""
    from gr00t.model.backbone import eagle_backbone

    official_auto_config = eagle_backbone.AutoConfig

    class EagerEagleConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            config = official_auto_config.from_pretrained(*args, **kwargs)
            config.vision_config._attn_implementation = "eager"
            config.llm_config._attn_implementation = "eager"
            return config

    eagle_backbone.AutoConfig = EagerEagleConfig


def _check_model_weights(model: object, checkpoint_dir: Path) -> dict[str, object]:
    from safetensors import safe_open

    model_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device="cpu") as reader:
        checkpoint_shapes = {
            key: tuple(reader.get_slice(key).get_shape()) for key in reader.keys()
        }
    missing = sorted(model_shapes.keys() - checkpoint_shapes.keys())
    unexpected = sorted(checkpoint_shapes.keys() - model_shapes.keys())
    mismatched = sorted(
        key
        for key in model_shapes.keys() & checkpoint_shapes.keys()
        if model_shapes[key] != checkpoint_shapes[key]
    )
    record = {
        "checkpoint_tensors": len(checkpoint_shapes),
        "model_tensors": len(model_shapes),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }
    allowed_unused = {"action_head.decode_layer.bias", "action_head.decode_layer.weight"}
    if missing or mismatched or not set(unexpected).issubset(allowed_unused):
        raise ValueError(f"Original GR00T-N1 weight-key mismatch: {record}")
    return record


def _load_video_observation(image_paths: dict[str, str], video_keys: list[str]) -> dict[str, object]:
    import numpy as np

    if "__single__" in image_paths:
        if len(image_paths) != 1 or len(video_keys) != 1:
            raise ValueError("Original GR00T-N1 needs a named image for every configured camera.")
        selected = {video_keys[0]: image_paths["__single__"]}
    else:
        missing = sorted(set(video_keys) - image_paths.keys())
        if missing:
            raise ValueError(f"Original GR00T-N1 observation is missing cameras: {missing}")
        selected = {key: image_paths[key] for key in video_keys}
    observation = {}
    for key, path in selected.items():
        image = np.load(path, allow_pickle=False)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Invalid original N1 input image {key}: {image.shape}, {image.dtype}")
        observation[key] = image[None, ...]
    return observation


def run(request_path: Path) -> None:
    import numpy as np
    import torch
    import transformers

    if transformers.__version__ != "4.45.2":
        raise RuntimeError(
            f"Original GR00T-N1 requires transformers==4.45.2; found {transformers.__version__}"
        )
    torch.set_num_threads(8)
    _install_eager_attention()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    request = json.loads(request_path.read_text())
    checkpoint = Path(request["checkpoint_dir"]).resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "gr00t_n1":
        raise ValueError(f"Unexpected GR00T-N1 model_type: {config.get('model_type')}")
    data_config_name = request["data_config_name"]
    if data_config_name not in DATA_CONFIG_MAP:
        raise ValueError(f"Unknown original N1 data config: {data_config_name}")
    data_config = DATA_CONFIG_MAP[data_config_name]
    embodiment = EmbodimentTag(str(request["embodiment_tag"]).lower())
    torch.manual_seed(int(request["seed"]))

    policy = Gr00tPolicy(
        model_path=str(checkpoint),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=embodiment,
        device=request["device"],
    )
    load_record = _check_model_weights(policy.model, checkpoint)
    work = request_path.parent
    _write(work / "load-record.json", load_record)
    _write(work / "status.json", {"status": "predicting", "load_record": load_record})

    observation = _load_video_observation(request["image_paths"], data_config.video_keys)
    observation[data_config.language_keys[0]] = [request["instruction"]]
    supplied_state = request["state"]
    for key in data_config.state_keys:
        if key not in supplied_state:
            raise ValueError(f"Original N1 observation is missing {key}")
        state = np.asarray(supplied_state[key], dtype=np.float32)
        if state.ndim == 3 and state.shape[0] == 1:
            state = state[0]
        if state.ndim == 1:
            state = state[None, ...]
        if state.ndim != 2 or state.shape[0] != 1 or not np.isfinite(state).all():
            raise ValueError(f"Invalid original N1 state {key}: {state.shape}")
        observation[key] = state

    started = time.monotonic()
    action = policy.get_action(observation)
    checks = {}
    serialized = {}
    for key, value in action.items():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != int(config["action_horizon"]):
            raise ValueError(f"Invalid original N1 action shape {key}: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite original N1 action {key}")
        checks[key] = {"shape": list(array.shape), "min": float(array.min()), "max": float(array.max())}
        serialized[key] = array.tolist()
    if not checks:
        raise ValueError("Original N1 returned no action groups")
    payload = {
        "schema_version": "worldfoundry-gr00t-action-trace",
        "status": "success",
        "model_id": "gr00t-n1-2b",
        "backend": "worldfoundry.gr00t.official_n1_release_subprocess",
        "backend_quality": "official_checkpoint_wrapper",
        "artifact_kind": "action_trace",
        "checkpoint_dir": str(checkpoint),
        "device": request["device"],
        "embodiment_tag": embodiment.value,
        "image_source": request["image_source"],
        "instruction": request["instruction"],
        "seed": request["seed"],
        "action": serialized,
        "info": {
            "source_revision": request["source_revision"],
            "transformers_version": transformers.__version__,
            "data_config_name": data_config_name,
            "load_record": load_record,
            "action_checks": checks,
            "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
        },
        "duration_seconds": round(time.monotonic() - started, 3),
        "metadata": request.get("extra_metadata", {}),
    }
    _write(Path(request["output_path"]), payload)
    _write(work / "status.json", {"status": "success", "action_checks": checks, "load_record": load_record})


if __name__ == "__main__":
    request_file = Path(sys.argv[1]).resolve()
    try:
        run(request_file)
    except BaseException as exc:
        _write(
            request_file.parent / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        raise
