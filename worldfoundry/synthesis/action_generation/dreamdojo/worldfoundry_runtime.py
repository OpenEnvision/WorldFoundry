from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import package_module_root as package_root
from worldfoundry.core.io import write_json
from worldfoundry.core.io.paths import hfd_root_path


def _resolve_hfd_root() -> Path:
    return hfd_root_path()


DEFAULT_DREAMDOJO_ROOT = _resolve_hfd_root() / "nvidia--DreamDojo"
DEFAULT_DREAMDOJO_CHECKPOINTS = DEFAULT_DREAMDOJO_ROOT / "2B_GR1_post-train"
DEFAULT_DREAMDOJO_DATASET = Path("datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot")


class DreamDojoRuntime:
    """WorldFoundry adapter for the packaged DreamDojo action-conditioned runtime."""

    MODEL_ID = "dreamdojo"
    DISPLAY_NAME = "DreamDojo"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoints_dir: str | Path = DEFAULT_DREAMDOJO_CHECKPOINTS,
        dataset_path: str | Path = DEFAULT_DREAMDOJO_DATASET,
        experiment: str = "dreamdojo_2b_480_640_gr1",
        checkpoint_interval: int = 5000,
    ) -> None:
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "robot_world_model"
        self.device = device
        self.checkpoints_dir = str(Path(checkpoints_dir).expanduser())
        self.dataset_path = str(Path(dataset_path).expanduser())
        self.experiment = experiment
        self.checkpoint_interval = int(checkpoint_interval)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "DreamDojoRuntime":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoints_dir"] = str(pretrained_model_path)
        options.update(kwargs)
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            checkpoints_dir=(
                options.get("checkpoints_dir")
                or options.get("checkpoint_dir")
                or options.get("pretrained_model_path")
                or DEFAULT_DREAMDOJO_CHECKPOINTS
            ),
            dataset_path=options.get("dataset_path") or DEFAULT_DREAMDOJO_DATASET,
            experiment=str(options.get("experiment") or "dreamdojo_2b_480_640_gr1"),
            checkpoint_interval=int(options.get("checkpoint_interval", 5000)),
        )

    @staticmethod
    def _runtime_root() -> Path:
        return package_root("worldfoundry.synthesis.action_generation.dreamdojo.dreamdojo_runtime")

    @staticmethod
    def _repo_src_root() -> Path:
        return Path(__file__).resolve().parents[5]

    def _checkpoint_path(self) -> Path:
        checkpoints_dir = Path(self.checkpoints_dir).expanduser().resolve()
        latest_file = checkpoints_dir / "latest_checkpoint.txt"
        if latest_file.is_file():
            latest = latest_file.read_text(encoding="utf-8").strip()
            checkpoint_dir = checkpoints_dir / latest
        else:
            checkpoint_dir = checkpoints_dir
        checkpoint_path = checkpoint_dir / "model_ema_bf16.pt"
        if checkpoint_path.is_file():
            return checkpoint_path
        distcp_dir = checkpoint_dir / "model"
        if distcp_dir.is_dir():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "worldfoundry.synthesis.action_generation.dreamdojo.convert_distcp",
                    str(distcp_dir),
                    str(checkpoint_dir),
                ],
                check=True,
            )
            if checkpoint_path.is_file():
                return checkpoint_path
            raise RuntimeError(f"DreamDojo checkpoint conversion did not create {checkpoint_path}")
        raise FileNotFoundError(f"DreamDojo checkpoint not found: {checkpoint_path}")

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt, images, video, fps
        runtime_root = self._runtime_root()
        checkpoint_path = self._checkpoint_path()
        output_dir = Path(kwargs.get("output_dir") or kwargs.get("save_dir") or Path.cwd() / "dreamdojo")
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = Path(output_path) if output_path is not None else output_dir / "dreamdojo_run.json"
        metadata_path = metadata_path.expanduser().resolve()

        code = f"""
from pathlib import Path
import sys
runtime_root = Path({str(runtime_root)!r})
src_root = Path({str(self._repo_src_root())!r})
sys.path.insert(0, str(src_root))
# ``dreamdojo_runtime`` is a package rooted at ``runtime_root``.  Python needs
# its parent on sys.path in order to import the package by name.
sys.path.insert(0, str(runtime_root.parent))
import dreamdojo_runtime  # noqa: F401
from cosmos_predict2.action_conditioned import inference
from cosmos_predict2.action_conditioned_config import ActionConditionedInferenceArguments, ActionConditionedSetupArguments
setup = ActionConditionedSetupArguments(
    output_dir={str(output_dir)!r},
    checkpoints_dir={str(Path(self.checkpoints_dir).expanduser().resolve())!r},
    save_dir={str(output_dir)!r},
    dataset_path={str(Path(self.dataset_path).expanduser())!r},
    experiment={self.experiment!r},
    checkpoint_interval={self.checkpoint_interval!r},
)
infer_args = ActionConditionedInferenceArguments()
infer_args.end = {int(kwargs.get("num_samples", 1))!r}
inference(setup, infer_args, Path({str(checkpoint_path)!r}))
"""
        python = str(kwargs.get("python_executable") or sys.executable)
        subprocess.run([python, "-c", code], cwd=str(runtime_root), check=True)
        payload = {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "run_dir": str(output_dir),
            "runtime": "worldfoundry.dreamdojo.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "checkpoint_path": str(checkpoint_path),
            "dataset_path": self.dataset_path,
            "interactions": list(interactions),
        }
        write_json(metadata_path, payload)
        return {**payload, "artifact_path": str(metadata_path)}
