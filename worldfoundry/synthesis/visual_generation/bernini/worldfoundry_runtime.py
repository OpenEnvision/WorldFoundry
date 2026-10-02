"""Import-light WorldFoundry runtime for the in-tree Bernini implementation."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    hfd_root_path,
    resolve_local_hf_model_path,
)

UPSTREAM_REVISION = "82d573e9a553dccb1405dde4cb0f26466f9df6c5"


@dataclass(frozen=True)
class BerniniVariant:
    model_id: str
    family: str
    checkpoint_repo_id: str
    checkpoint_env: str


VARIANTS: dict[str, BerniniVariant] = {
    "bernini": BerniniVariant(
        model_id="bernini",
        family="planner_renderer",
        checkpoint_repo_id="ByteDance/Bernini-Diffusers",
        checkpoint_env="WORLDFOUNDRY_BERNINI_ROOT",
    ),
    "bernini-r-14b": BerniniVariant(
        model_id="bernini-r-14b",
        family="renderer",
        checkpoint_repo_id="ByteDance/Bernini-R-Diffusers",
        checkpoint_env="WORLDFOUNDRY_BERNINI_R_ROOT",
    ),
    "bernini-r-1.3b": BerniniVariant(
        model_id="bernini-r-1.3b",
        family="renderer",
        checkpoint_repo_id="ByteDance/Bernini-R-1.3B-Diffusers",
        checkpoint_env="WORLDFOUNDRY_BERNINI_R_1P3B_ROOT",
    ),
}

TASK_ALIASES = {
    "text-to-image": "t2i",
    "image-to-image": "i2i",
    "image-editing": "i2i",
    "text-to-video": "t2v",
    "video-to-video": "v2v",
    "video-editing": "v2v",
    "reference-to-video": "r2v",
    "image-to-video": "r2v",
    "reference-video-to-video": "rv2v",
    "reference-guided-video-editing": "rv2v",
}

FULL_TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "t2i": {
        "num_frames": 1,
        "max_image_size": 842,
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
        "guidance_mode": "vae_txt_vit_wapg",
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_img": 1.0,
        "omega_vid": 1.0,
        "omega_scale": 1.0,
        "vit_denoising_step": 5,
        "vit_txt_cfg": 1.2,
        "vit_img_cfg": 1.0,
    },
    "i2i": {
        "num_frames": 1,
        "max_image_size": 842,
        "height": 512,
        "width": 512,
        "num_inference_steps": 40,
        "guidance_mode": "vae_txt_vit_wapg",
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_img": 1.25,
        "omega_vid": 1.25,
        "omega_scale": 0.75,
        "vit_denoising_step": 5,
        "vit_txt_cfg": 1.2,
        "vit_img_cfg": 1.0,
    },
    "t2v": {
        "num_frames": 81,
        "max_image_size": 842,
        "height": 480,
        "width": 848,
        "num_inference_steps": 50,
        "guidance_mode": "vae_txt_vit_wapg",
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_img": 1.0,
        "omega_vid": 1.0,
        "omega_scale": 1.0,
        "vit_denoising_step": 5,
        "vit_txt_cfg": 1.2,
        "vit_img_cfg": 1.0,
    },
    "v2v": {
        "num_frames": 81,
        "max_image_size": 848,
        "height": 0,
        "width": 0,
        "num_inference_steps": 40,
        "guidance_mode": "vae_txt_vit_wapg",
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_img": 1.25,
        "omega_vid": 1.25,
        "omega_scale": 0.75,
        "vit_denoising_step": 5,
        "vit_txt_cfg": 1.2,
        "vit_img_cfg": 1.0,
    },
    "r2v": {
        "num_frames": 81,
        "max_image_size": 842,
        "height": 480,
        "width": 848,
        "num_inference_steps": 40,
        "guidance_mode": "vae_txt_vit_wapg",
        "omega_txt": 4.0,
        "omega_tgt": 1.5,
        "omega_img": 4.5,
        "omega_vid": 1.25,
        "omega_scale": 0.8,
        "vit_denoising_step": 5,
        "vit_txt_cfg": 1.2,
        "vit_img_cfg": 1.0,
    },
    "rv2v": {
        "num_frames": 81,
        "max_image_size": 848,
        "height": 0,
        "width": 0,
        "num_inference_steps": 40,
        "guidance_mode": "rv2v_wapg",
        "omega_txt": 4.0,
        "omega_tgt": 1.5,
        "omega_img": 3.0,
        "omega_vid": 0.75,
        "omega_scale": 0.75,
        "vit_denoising_step": 5,
        "vit_txt_cfg": 1.2,
        "vit_img_cfg": 1.0,
    },
}

RENDERER_TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "t2i": dict(num_frames=1, guidance_mode="t2v_apg"),
    "i2i": dict(num_frames=1, guidance_mode="v2v"),
    "t2v": dict(num_frames=81, guidance_mode="t2v_apg"),
    "v2v": dict(num_frames=81, guidance_mode="v2v_apg"),
    "r2v": dict(num_frames=81, guidance_mode="r2v_apg"),
    "rv2v": dict(num_frames=81, guidance_mode="rv2v"),
}

COMMON_DEFAULTS = {
    "neg_prompt": (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    "max_image_size": 848,
    "height": 480,
    "width": 848,
    "num_inference_steps": 40,
    "omega_vid": 1.25,
    "omega_img": 4.5,
    "omega_txt": 4.0,
    "omega_scale": 0.8,
    "flow_shift": 5.0,
    "seed": 42,
    "fps": 16,
    "eta": 0.5,
    "norm_threshold": (50.0, 50.0, 50.0),
    "momentum": 0.0,
}

MODEL_CONFIG_DEFAULTS = {
    "use_unipc": True,
    "use_src_id_rotary_emb": True,
    "interpolate_src_id": True,
    "max_trained_src_id": 5,
}
MODEL_CONFIG_KEYS = frozenset(MODEL_CONFIG_DEFAULTS)

# Moving the largest model stage still needs working space for activations and
# CUDA kernels.  This is deliberately a reserve, not a claimed exact peak.
CUDA_WORKSPACE_RESERVE_BYTES = 8 * 1024**3


def checkpoint_integrity_errors(checkpoint_path: Path) -> list[str]:
    """Return actionable errors for interrupted or incomplete checkpoints.

    Bernini checkpoints are large enough that an interrupted download is a
    normal operational condition.  A root ``config.json`` alone must not make
    such a directory appear runnable: loading tens of GiB before discovering a
    missing final shard is both slow and needlessly consumes host memory.
    """

    errors: list[str] = []
    if not (checkpoint_path / "config.json").is_file():
        errors.append("checkpoint root config.json is missing")

    partials = [path for path in checkpoint_path.rglob("*.aria2") if ".cache" not in path.parts]
    if partials:
        examples = ", ".join(str(path.relative_to(checkpoint_path)) for path in partials[:3])
        errors.append(
            f"checkpoint download is incomplete ({len(partials)} aria2 partial file(s), e.g. {examples})"
        )

    # hfd records the repository manifest before starting aria2.  When it is
    # present, use it to catch missing shards even if the index/config files
    # themselves have not arrived yet.
    metadata_path = checkpoint_path / ".hfd" / "repo_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = None
    if isinstance(metadata, Mapping):
        expected: list[str] = []
        for sibling in metadata.get("siblings") or ():
            relative = sibling.get("rfilename") if isinstance(sibling, Mapping) else None
            if not isinstance(relative, str) or relative.startswith("assets/"):
                continue
            if relative in {".gitattributes", "README.md"} or relative.endswith("/.gitattributes"):
                continue
            expected.append(relative)
        missing = [relative for relative in expected if not (checkpoint_path / relative).is_file()]
        if missing:
            examples = ", ".join(missing[:4])
            errors.append(
                f"checkpoint is missing {len(missing)} repository file(s), e.g. {examples}"
            )

    # The current hfd helper records exact byte sizes in a tab-separated
    # manifest. Presence checks alone are insufficient: a truncated shard can
    # outlive a lost/stale .aria2 sidecar and otherwise look runnable until the
    # heavyweight model loader reaches that shard.
    manifest_path = checkpoint_path / ".hfd" / "manifest"
    try:
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        manifest_lines = []
    malformed_manifest: list[str] = []
    wrong_sizes: list[tuple[str, int, int]] = []
    for line_number, line in enumerate(manifest_lines, start=1):
        if not line.strip():
            continue
        try:
            size_text, relative = line.split("\t", 1)
            expected_size = int(size_text)
        except (ValueError, TypeError):
            malformed_manifest.append(f"line {line_number}")
            continue
        relative_path = Path(relative)
        if expected_size < 0 or relative_path.is_absolute() or ".." in relative_path.parts:
            malformed_manifest.append(f"line {line_number}")
            continue
        if relative.startswith("assets/") or relative in {".gitattributes", "README.md"}:
            continue
        if relative.endswith("/.gitattributes"):
            continue
        local_path = checkpoint_path / relative_path
        if not local_path.is_file():
            continue  # Missing paths are reported above and by checkpoint indexes.
        actual_size = local_path.stat().st_size
        if actual_size != expected_size:
            wrong_sizes.append((relative, actual_size, expected_size))
    if malformed_manifest:
        examples = ", ".join(malformed_manifest[:3])
        errors.append(f"invalid hfd byte-size manifest ({examples})")
    if wrong_sizes:
        examples = ", ".join(
            f"{relative}={actual}/{expected} bytes"
            for relative, actual, expected in wrong_sizes[:3]
        )
        errors.append(
            f"checkpoint has {len(wrong_sizes)} wrong-sized repository file(s), e.g. {examples}"
        )

    for index_path in checkpoint_path.rglob("*.safetensors.index.json"):
        if ".cache" in index_path.parts:
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid checkpoint index {index_path.relative_to(checkpoint_path)}: {exc}")
            continue
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            errors.append(
                f"invalid checkpoint index {index_path.relative_to(checkpoint_path)}: weight_map is missing"
            )
            continue
        shard_names = {name for name in weight_map.values() if isinstance(name, str)}
        missing_shards = [name for name in sorted(shard_names) if not (index_path.parent / name).is_file()]
        if missing_shards:
            examples = ", ".join(missing_shards[:4])
            errors.append(
                f"{index_path.relative_to(checkpoint_path)} references {len(missing_shards)} "
                f"missing shard(s), e.g. {examples}"
            )

    return errors


def _safetensors_group_sizes(index_path: Path) -> dict[str, int]:
    """Return tensor bytes grouped by the first two state-dict components."""

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        return {}

    tensors_by_file: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        if isinstance(name, str) and isinstance(filename, str):
            tensors_by_file.setdefault(filename, set()).add(name)

    sizes: Counter[str] = Counter()
    for filename, tensor_names in tensors_by_file.items():
        shard = index_path.parent / filename
        try:
            with shard.open("rb") as handle:
                header_length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_length))
        except (OSError, json.JSONDecodeError, struct.error):
            return {}
        for name in tensor_names:
            tensor = header.get(name)
            offsets = tensor.get("data_offsets") if isinstance(tensor, Mapping) else None
            if not isinstance(offsets, list) or len(offsets) != 2:
                return {}
            parts = name.split(".")
            group = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            sizes[group] += int(offsets[1]) - int(offsets[0])
    return dict(sizes)


def checkpoint_resource_estimate(checkpoint_path: Path) -> dict[str, int | None]:
    """Estimate per-rank host weights and the largest CUDA-resident stage."""

    shards = [
        path
        for path in checkpoint_path.rglob("*.safetensors")
        if ".cache" not in path.parts
    ]
    host_weight_bytes = sum(path.stat().st_size for path in shards)
    stage_sizes: Counter[str] = Counter()
    largest_shard_bytes = 0
    for path in shards:
        relative = path.relative_to(checkpoint_path)
        stage = relative.parts[0] if len(relative.parts) > 1 else "root"
        size = path.stat().st_size
        stage_sizes[stage] += size
        largest_shard_bytes = max(largest_shard_bytes, size)
    expert_bytes = stage_sizes["transformer"] + stage_sizes["transformer_2"]
    streaming_host_weight_bytes = host_weight_bytes - expert_bytes + largest_shard_bytes

    group_sizes: Counter[str] = Counter()
    indexed_totals: list[int] = []
    for index_path in checkpoint_path.rglob("*.safetensors.index.json"):
        if ".cache" in index_path.parts:
            continue
        groups = _safetensors_group_sizes(index_path)
        group_sizes.update(groups)
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            total_size = int((payload.get("metadata") or {}).get("total_size") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            total_size = 0
        if total_size:
            indexed_totals.append(total_size)

    # A combined Bernini index contains several stages which are moved to CUDA
    # sequentially (planner, T5, then the two diffusion experts).  Treating the
    # index's aggregate total as one resident stage incorrectly reports 168 GiB
    # for a checkpoint whose largest stage is about 53 GiB.  Fall back to the
    # aggregate only when tensor group information could not be read.
    peak_cuda_weight_bytes = (
        max(group_sizes.values())
        if group_sizes
        else max(indexed_totals, default=0)
    )
    return {
        "host_weight_bytes_per_rank": host_weight_bytes or None,
        "streaming_host_weight_bytes_per_rank": streaming_host_weight_bytes or None,
        "largest_checkpoint_shard_bytes": largest_shard_bytes or None,
        "peak_cuda_weight_bytes": peak_cuda_weight_bytes or None,
    }


def _cgroup_memory_available_bytes() -> int | None:
    """Read reclaim-aware cgroup memory headroom for v1 or v2 containers.

    Both cgroup versions charge file cache to ``memory.current``/``usage``.
    Checkpoint scans can therefore make raw headroom look nearly exhausted even
    though inactive file pages are reclaimable by model parameters.  Count only
    ``inactive_file`` as reclaimable; active cache remains a conservative safety
    margin for the multi-rank Bernini preflight.
    """

    candidates = (
        (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.stat"),
            ("inactive_file",),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.stat"),
            ("total_inactive_file", "inactive_file"),
        ),
    )
    for current_path, limit_path, stat_path, inactive_keys in candidates:
        try:
            current_text = current_path.read_text(encoding="utf-8").strip()
            limit_text = limit_path.read_text(encoding="utf-8").strip()
            if limit_text == "max":
                return None
            current, limit = int(current_text), int(limit_text)
        except (OSError, ValueError):
            continue
        # Very large v1 values represent an unlimited cgroup.
        if limit >= 1 << 60:
            return None
        inactive_file = 0
        try:
            stat_values = {
                key: int(value)
                for line in stat_path.read_text(encoding="utf-8").splitlines()
                if len((parts := line.split())) == 2
                for key, value in (parts,)
            }
            inactive_file = next(
                (stat_values[key] for key in inactive_keys if key in stat_values),
                0,
            )
        except (OSError, ValueError):
            inactive_file = 0
        return min(limit, max(limit - current, 0) + max(inactive_file, 0))
    return None


def normalize_task_type(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    normalized = TASK_ALIASES.get(normalized, normalized)
    if normalized in {"mv2v", "ads2v"}:
        return "v2v" if normalized == "mv2v" else "rv2v"
    if normalized not in {"t2i", "i2i", "t2v", "v2v", "r2v", "rv2v"}:
        raise ValueError(f"Unsupported Bernini task type {value!r}.")
    return normalized


def infer_task_type(
    *,
    explicit: str | None,
    image: Any,
    images: Sequence[Any] | None,
    video: Any,
    num_frames: int | None,
    output_schema: Mapping[str, Any] | None = None,
) -> str:
    if explicit:
        return normalize_task_type(explicit)
    output_keys = {str(key).lower() for key in (output_schema or {})}
    image_output = (num_frames == 1) or any("image" in key for key in output_keys)
    refs = bool(images) or image is not None
    if image_output:
        return "i2i" if refs else "t2i"
    if video is not None:
        return "rv2v" if refs else "v2v"
    return "r2v" if refs else "t2v"


class BerniniRuntime:
    """Lazy, checkpoint-backed wrapper around the pinned Bernini pipelines."""

    def __init__(
        self,
        *,
        model_id: str,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        model_config: Mapping[str, Any] | None = None,
    ) -> None:
        if model_id not in VARIANTS:
            raise ValueError(f"Unknown Bernini model id: {model_id}")
        self.variant = VARIANTS[model_id]
        self.device = device
        self.model_config = dict(MODEL_CONFIG_DEFAULTS)
        unknown_config = set(model_config or ()) - MODEL_CONFIG_KEYS
        if unknown_config:
            raise ValueError(f"Unsupported Bernini model config keys: {sorted(unknown_config)}")
        self.model_config.update(dict(model_config or {}))
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self._pipeline: Any = None

    def _resolve_checkpoint(self, explicit: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        env_value = os.environ.get(self.variant.checkpoint_env)
        if env_value:
            candidates.append(Path(env_value).expanduser())
        staged_name = self.variant.checkpoint_repo_id.replace("/", "--")
        candidates.extend(
            (
                hfd_root_path(staged_name),
                checkpoint_root_path(staged_name),
                checkpoint_root_path(self.variant.checkpoint_repo_id.rsplit("/", 1)[-1]),
                Path(self.variant.checkpoint_repo_id),
            )
        )
        for candidate in candidates:
            try:
                snapshot = resolve_local_hf_model_path(
                    candidate,
                    required_files=("config.json",),
                )
                if snapshot.is_dir() and (snapshot / "config.json").is_file():
                    return snapshot.resolve()
            except (FileNotFoundError, OSError):
                continue
        return None

    def requirement_report(self) -> dict[str, Any]:
        missing = []
        for module in (
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "safetensors",
            "einops",
            "numpy",
            "PIL",
            "tqdm",
            "ftfy",
            "veomni",
            "decord",
            "imageio",
            "imageio_ffmpeg",
        ):
            if importlib.util.find_spec(module) is None:
                missing.append(f"missing Python dependency: {module}")
        if self.checkpoint_path is None:
            missing.append(
                f"checkpoint not found; set {self.variant.checkpoint_env} or stage "
                f"{self.variant.checkpoint_repo_id} under WORLDFOUNDRY_HFD_ROOT"
            )
        else:
            missing.extend(checkpoint_integrity_errors(self.checkpoint_path))
        return {
            "ready": not missing,
            "missing": missing,
            "model_id": self.variant.model_id,
            "family": self.variant.family,
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "checkpoint_repo_id": self.variant.checkpoint_repo_id,
            "upstream_revision": UPSTREAM_REVISION,
        }

    def resource_report(self, *, nproc_per_node: int) -> dict[str, Any]:
        """Describe checkpoint residency and reject predictable host/GPU OOMs."""

        if self.checkpoint_path is None:
            return {"ready": False, "blockers": ["checkpoint is unavailable"]}
        estimate = checkpoint_resource_estimate(self.checkpoint_path)
        blockers: list[str] = []
        streaming_experts = self.variant.model_id == "bernini-r-14b"
        host_per_rank = estimate[
            "streaming_host_weight_bytes_per_rank"
            if streaming_experts
            else "host_weight_bytes_per_rank"
        ]
        host_available = _cgroup_memory_available_bytes()
        if host_per_rank and host_available is not None:
            host_required = int(host_per_rank) * nproc_per_node
            if host_required > host_available:
                blockers.append(
                    "Bernini Ulysses ranks replicate checkpoint weights in host memory: "
                    f"estimated {host_required / 1024**3:.1f} GiB for {nproc_per_node} ranks, "
                    f"but the cgroup has {host_available / 1024**3:.1f} GiB available"
                )

        gpu_free: list[int] = []
        peak_cuda_weight = estimate["peak_cuda_weight_bytes"]
        try:
            import torch

            if torch.cuda.is_available():
                if nproc_per_node > torch.cuda.device_count():
                    blockers.append(
                        f"requested {nproc_per_node} CUDA ranks but only {torch.cuda.device_count()} devices are visible"
                    )
                else:
                    if nproc_per_node == 1 and str(self.device).startswith("cuda:"):
                        indices = [int(str(self.device).split(":", 1)[1])]
                    elif nproc_per_node == 1:
                        indices = [torch.cuda.current_device()]
                    else:
                        indices = list(range(nproc_per_node))
                    gpu_free = [int(torch.cuda.mem_get_info(index)[0]) for index in indices]
                    if peak_cuda_weight:
                        required = int(peak_cuda_weight) + CUDA_WORKSPACE_RESERVE_BYTES
                        insufficient = [
                            (index, free)
                            for index, free in zip(indices, gpu_free)
                            if free < required
                        ]
                        if insufficient:
                            rendered = ", ".join(
                                f"cuda:{index}={free / 1024**3:.1f} GiB free"
                                for index, free in insufficient
                            )
                            blockers.append(
                                "largest Bernini stage needs at least "
                                f"{required / 1024**3:.1f} GiB including CUDA workspace; {rendered}"
                            )
        except (ImportError, RuntimeError, ValueError):
            # Dependency/device readiness is reported separately. Resource
            # discovery must not make plan-only inspection unusable.
            pass

        return {
            "ready": not blockers,
            "blockers": blockers,
            **estimate,
            "host_available_bytes": host_available,
            "cuda_free_bytes": gpu_free,
            "cuda_workspace_reserve_bytes": CUDA_WORKSPACE_RESERVE_BYTES,
            "nproc_per_node": nproc_per_node,
            "expert_loading": "streamed" if streaming_experts else "resident",
        }

    def _require_resources(self, *, nproc_per_node: int) -> None:
        report = self.resource_report(nproc_per_node=nproc_per_node)
        if not report["ready"]:
            raise RuntimeError("; ".join(report["blockers"]))

    def _load_pipeline(self, *, ulysses_size: int = 1) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        report = self.requirement_report()
        if not report["ready"]:
            raise RuntimeError("; ".join(report["missing"]))
        if not str(self.device).startswith("cuda"):
            raise RuntimeError("Bernini inference requires a CUDA device.")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            import torch

            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if ulysses_size != world_size:
                raise ValueError(
                    "A WorldFoundry Bernini request uses one Ulysses group; "
                    "ulysses_size must equal the torchrun world size."
                )
            from worldfoundry.core.distributed.runtime.torch_process_group import init as init_process_group

            init_process_group()
            self.device = f"cuda:{torch.cuda.current_device() if torch.cuda.is_available() else local_rank}"
            from .inference.parallel import init_parallel_state

            init_parallel_state(ulysses_size=ulysses_size)

        from .inference.pipeline import BerniniPipeline, BerniniRendererPipeline

        pipeline_cls = BerniniPipeline if self.variant.family == "planner_renderer" else BerniniRendererPipeline
        load_options = dict(self.model_config)
        if self.variant.family == "renderer":
            # Public Bernini-R repos already contain transformer/transformer_2.
            load_options["load_ckpt_weights"] = False
        if self.variant.model_id == "bernini-r-14b":
            # The two 14B experts cannot coexist in this cgroup. Load the
            # active expert directly onto CUDA and replace it at the scheduler
            # boundary instead of materializing both in host memory.
            load_options["lazy_expert_loading"] = True
            load_options["expert_device"] = self.device
        self._pipeline = pipeline_cls.from_pretrained(
            str(self.checkpoint_path),
            device=self.device,
            **load_options,
        )
        return self._pipeline

    @staticmethod
    def _plan_path(output_path: Path) -> Path:
        return output_path.with_suffix(output_path.suffix + ".runtime_plan.json")

    def runtime_plan(
        self,
        *,
        task_type: str,
        output_path: Path,
        nproc_per_node: int = 1,
        ulysses_size: int = 1,
    ) -> dict[str, Any]:
        return {
            **self.requirement_report(),
            "task_type": task_type,
            "output_path": str(output_path),
            "nproc_per_node": nproc_per_node,
            "ulysses_size": ulysses_size,
            "backend_quality": "official_in_tree_runtime",
            "resources": self.resource_report(nproc_per_node=nproc_per_node),
        }

    def _generate_distributed(
        self,
        *,
        prompt: str,
        output_path: Path,
        task_type: str,
        image: Any,
        images: Sequence[Any] | None,
        video: Any,
        nproc_per_node: int,
        ulysses_size: int,
        overrides: Mapping[str, Any],
    ) -> None:
        if nproc_per_node < 2:
            raise ValueError("Distributed Bernini inference requires nproc_per_node >= 2.")
        if ulysses_size != nproc_per_node:
            raise ValueError(
                "Single-request Bernini inference requires ulysses_size == nproc_per_node; "
                "data-parallel task batches are not represented by this pipeline call."
            )
        payload = {
            "model_id": self.variant.model_id,
            "checkpoint_path": str(self.checkpoint_path),
            "model_config": self.model_config,
            "prompt": prompt,
            "output_path": str(output_path),
            "task_type": task_type,
            "image": image,
            "images": list(images or ()),
            "video": video,
            "ulysses_size": ulysses_size,
            "overrides": dict(overrides),
        }
        with tempfile.TemporaryDirectory(prefix="worldfoundry_bernini_") as temp_dir:
            payload_path = Path(temp_dir) / "request.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("NCCL_NET_PLUGIN", "none")
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            from worldfoundry.core.execution.process import run_torchrun_module

            completed = run_torchrun_module(
                "worldfoundry.synthesis.visual_generation.bernini.distributed_infer",
                nproc_per_node=nproc_per_node,
                args=("--payload", str(payload_path)),
                env=env,
                python_executable=sys.executable,
            )
            if completed.returncode:
                detail = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(f"Bernini distributed inference failed: {detail[-8000:]}")

    def generate(
        self,
        *,
        prompt: str,
        output_path: str | Path,
        task_type: str,
        image: Any = None,
        images: Sequence[Any] | None = None,
        video: Any = None,
        plan_only: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        task_type = normalize_task_type(task_type)
        nproc_per_node = int(overrides.pop("nproc_per_node", 1))
        ulysses_size = int(overrides.pop("ulysses_size", nproc_per_node if nproc_per_node > 1 else 1))
        requested_output = Path(output_path).expanduser().resolve()
        output = requested_output.with_suffix(".png") if task_type in {"t2i", "i2i"} else requested_output.with_suffix(".mp4")
        report = self.requirement_report()
        if plan_only or not report["ready"]:
            plan_path = self._plan_path(output)
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(
                    self.runtime_plan(
                        task_type=task_type,
                        output_path=output,
                        nproc_per_node=nproc_per_node,
                        ulysses_size=ulysses_size,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return {
                "status": "planned" if plan_only and report["ready"] else "blocked",
                "backend_quality": "official_in_tree_runtime_plan",
                "blocked_reasons": report["missing"],
                "blocked_reason": "; ".join(report["missing"]) if report["missing"] else None,
                "plan_path": str(plan_path),
                "artifact_path": str(output),
                "artifact_kind": "generated_image" if task_type in {"t2i", "i2i"} else "generated_video",
            }

        defaults = dict(COMMON_DEFAULTS)
        defaults.update((FULL_TASK_DEFAULTS if self.variant.family == "planner_renderer" else RENDERER_TASK_DEFAULTS)[task_type])
        defaults.update({key: value for key, value in overrides.items() if value is not None})
        output.parent.mkdir(parents=True, exist_ok=True)
        distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        # torchrun workers inherit a launch that was already checked by the
        # parent; checking before rank-local CUDA selection would inspect GPU 0
        # on every worker.
        if not distributed:
            self._require_resources(nproc_per_node=nproc_per_node)
        if nproc_per_node > 1 and not distributed:
            self._generate_distributed(
                prompt=prompt,
                output_path=output,
                task_type=task_type,
                image=image,
                images=images,
                video=video,
                nproc_per_node=nproc_per_node,
                ulysses_size=ulysses_size,
                overrides=defaults,
            )
        else:
            pipeline = self._load_pipeline(ulysses_size=ulysses_size)
            write_output = True
            if distributed:
                from .inference.parallel import get_parallel_state

                write_output = get_parallel_state().ulysses_rank == 0
            # The official CLI selects a task-specific prefix for both the
            # planner and renderer pipelines. Keep an explicit override intact.
            from .inference.system_prompts import get_system_prompt_for_task

            defaults.setdefault("system_prompt", get_system_prompt_for_task(task_type))
            if self.variant.family == "planner_renderer":
                pipeline(
                    task_type,
                    prompt,
                    image=image,
                    images=list(images or ()) or None,
                    video=video,
                    output_path=str(output),
                    write_output=write_output,
                    **defaults,
                )
            else:
                if task_type == "r2v" and image is not None and not images:
                    images = [image]
                    image = None
                pipeline(
                    prompt,
                    image=image,
                    images=list(images or ()) or None,
                    video=video,
                    output_path=str(output),
                    write_output=write_output,
                    **defaults,
                )
            if distributed:
                import torch.distributed as dist

                dist.barrier()
        if not output.is_file():
            raise RuntimeError(f"Bernini completed without producing the expected artifact: {output}")
        return {
            "status": "succeeded",
            "runtime": "bernini_in_tree",
            "backend_quality": "official_in_tree_runtime",
            "artifact_path": str(output),
            "artifact_kind": "generated_image" if task_type in {"t2i", "i2i"} else "generated_video",
            "artifact_sha256": file_sha256(output),
            "metadata": {
                "model_id": self.variant.model_id,
                "task_type": task_type,
                "checkpoint_path": str(self.checkpoint_path),
                "upstream_revision": UPSTREAM_REVISION,
            },
        }


__all__ = [
    "BerniniRuntime",
    "BerniniVariant",
    "FULL_TASK_DEFAULTS",
    "MODEL_CONFIG_KEYS",
    "RENDERER_TASK_DEFAULTS",
    "UPSTREAM_REVISION",
    "VARIANTS",
    "checkpoint_integrity_errors",
    "infer_task_type",
    "normalize_task_type",
]
