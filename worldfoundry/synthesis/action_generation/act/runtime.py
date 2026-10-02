# Inference-only ACT source retained in-tree.
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path

from .policy import build_policy_class, load_dataset_stats


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def select_act_checkpoint(*, checkpoint_path: str | Path | None, checkpoint_dir: str | Path | None, checkpoints: Sequence[Mapping[str, Any]]) -> Path:
    """Select an ACT checkpoint file from explicit options or profile metadata.

    Args:
        checkpoint_path: Explicit policy checkpoint file or checkpoint directory.
        checkpoint_dir: Explicit directory containing policy_best.ckpt or policy_last.ckpt.
        checkpoints: Profile checkpoint records.
    """
    def expand(value: str | Path) -> Path:
        path = resolve_worldfoundry_path(value)
        if not path.is_absolute():
            path = project_root() / path
        return path.resolve()

    candidates: list[Path] = []
    if checkpoint_path:
        path = expand(checkpoint_path)
        if path.is_dir():
            candidates.extend([path / "policy_best.ckpt", path / "policy_last.ckpt"])
        else:
            candidates.append(path)
    if checkpoint_dir:
        root = expand(checkpoint_dir)
        candidates.extend([root / "policy_best.ckpt", root / "policy_last.ckpt"])
    for item in checkpoints:
        local_path = item.get("local_path") or item.get("path") or item.get("checkpoint_path")
        local_dir = item.get("local_dir")
        if local_path:
            candidates.append(expand(str(local_path)))
        if local_dir:
            root = expand(str(local_dir))
            candidates.extend([root / "policy_best.ckpt", root / "policy_last.ckpt"])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No local ACT policy checkpoint was found.")


@dataclass(frozen=True)
class ACTRuntimeConfig:
    """Runtime settings for in-tree ACT inference.

    Args:
        checkpoint_path: Local ACT policy checkpoint file.
        device: Torch device string.
        camera_names: Ordered camera names consumed by the ACT policy.
        state_dim: Proprio/action dimension.
        chunk_size: Number of actions predicted per policy query.
        temporal_agg: Whether to use ACT temporal aggregation.
    """
    checkpoint_path: Path
    device: str = "cuda"
    camera_names: tuple[str, ...] = ("head_cam", "left_cam", "right_cam")
    state_dim: int = 14
    chunk_size: int = 100
    temporal_agg: bool = False
    backbone: str = "resnet18"
    dilation: bool = False
    position_embedding: str = "sine"
    enc_layers: int = 4
    dec_layers: int = 7
    dim_feedforward: int = 3200
    hidden_dim: int = 512
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = False
    masks: bool = False
    require_dataset_stats: bool = True


class ACTRuntime:
    """Lazy in-tree ACT runtime backed by the local ACT policy implementation."""

    def __init__(self, config: ACTRuntimeConfig) -> None:
        self.config = config
        self.policy: Any | None = None
        self.stats: Mapping[str, Any] | None = None
        self.t = 0
        self.all_actions: Any | None = None
        self.all_time_actions: Any | None = None

    def load(self) -> None:
        """Load ACT architecture, weights, and dataset stats.

        Args:
            None.
        """
        if self.policy is not None:
            return
        import torch

        checkpoint = self.config.checkpoint_path.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ACT checkpoint file does not exist: {checkpoint}")
        args = SimpleNamespace(**self._args_dict(checkpoint.parent))
        policy_cls = build_policy_class()
        self.policy = policy_cls(args).to(torch.device(self.config.device))
        self.policy.eval()
        from worldfoundry.core.model_loading.checkpoints.file import load_torch_checkpoint

        payload = load_torch_checkpoint(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(payload, Mapping):
            for wrapper in ("model_state_dict", "state_dict", "model", "policy"):
                nested = payload.get(wrapper)
                if isinstance(nested, Mapping):
                    payload = nested
                    break
        if not isinstance(payload, Mapping) or not payload:
            raise TypeError("ACT checkpoint must contain a non-empty tensor state dictionary")
        if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
            raise TypeError("ACT checkpoint state dictionary may contain only named tensors")
        self.policy.load_state_dict(dict(payload), strict=True)
        self.stats = load_dataset_stats(checkpoint.parent)
        if self.config.require_dataset_stats and self.stats is None:
            raise FileNotFoundError(
                f"ACT requires dataset_stats.pkl beside the policy checkpoint: {checkpoint.parent}"
            )
        if self.config.temporal_agg:
            self.all_time_actions = torch.zeros(
                [3000, 3000 + self.config.chunk_size, self.config.state_dim],
                device=torch.device(self.config.device),
            )

    def predict_action(
        self,
        *,
        observation: Mapping[str, Any],
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one ACT policy query and write a WorldFoundry action trace.

        Args:
            observation: ACT observation with qpos plus camera image tensors/arrays.
            output_path: Destination action trace path.
            extra_metadata: Additional metadata copied into the trace.
        """
        self.load()
        assert self.policy is not None

        import numpy as np
        import torch

        started = time.monotonic()
        qpos_numpy = np.asarray(observation["qpos"], dtype=np.float32)
        qpos = torch.from_numpy(self._pre_process(qpos_numpy)).float().to(self.config.device).unsqueeze(0)
        images = [np.asarray(observation[name], dtype=np.float32) for name in self.config.camera_names]
        curr_image = torch.from_numpy(np.stack(images, axis=0)).float().to(self.config.device).unsqueeze(0)

        with torch.no_grad():
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
            if self.config.temporal_agg:
                assert self.all_time_actions is not None
                self.all_time_actions[[self.t], self.t : self.t + self.config.chunk_size] = self.all_actions
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                weights = torch.from_numpy(exp_weights).to(self.config.device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * weights).sum(dim=0, keepdim=True)
            else:
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        action = self._post_process(raw_action.cpu().numpy())
        self.t += 1
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "worldfoundry-act-action-trace",
            "status": "success",
            "model_id": "act",
            "backend": "worldfoundry.act.in_tree_runtime.ACTRuntime.predict_action",
            "backend_quality": "official_architecture",
            "artifact_kind": "action_trace",
            "checkpoint_path": str(self.config.checkpoint_path),
            "camera_names": list(self.config.camera_names),
            "state_dim": self.config.state_dim,
            "chunk_size": self.config.chunk_size,
            "temporal_agg": self.config.temporal_agg,
            "query_frequency": self.query_frequency,
            "action_shape": list(action.shape),
            "action": _jsonable(action),
            "duration_seconds": round(time.monotonic() - started, 3),
            "metadata": _jsonable(dict(extra_metadata or {})),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_sha256 = file_sha256(target)
        return {
            "status": "success",
            "model_id": "act",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": artifact_sha256,
            "backend": payload["backend"],
            "backend_quality": payload["backend_quality"],
            "duration_seconds": payload["duration_seconds"],
        }

    @property
    def query_frequency(self) -> int:
        return 1 if self.config.temporal_agg else self.config.chunk_size

    def _args_dict(self, checkpoint_dir: Path) -> dict[str, Any]:
        return {
            "ckpt_dir": str(checkpoint_dir),
            "device": self.config.device,
            "camera_names": list(self.config.camera_names),
            "state_dim": self.config.state_dim,
            "chunk_size": self.config.chunk_size,
            "temporal_agg": self.config.temporal_agg,
            "backbone": self.config.backbone,
            "dilation": self.config.dilation,
            "position_embedding": self.config.position_embedding,
            "enc_layers": self.config.enc_layers,
            "dec_layers": self.config.dec_layers,
            "dim_feedforward": self.config.dim_feedforward,
            "hidden_dim": self.config.hidden_dim,
            "dropout": self.config.dropout,
            "nheads": self.config.nheads,
            "pre_norm": self.config.pre_norm,
            "masks": self.config.masks,
        }

    def _pre_process(self, qpos):
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def _post_process(self, action):
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action
