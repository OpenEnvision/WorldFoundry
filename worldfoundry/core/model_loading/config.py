"""Model checkpoint path resolution and device/dtype placement config.

:class:`ModelConfig` (aliased as :data:`LoaderModelConfig`) describes
where weights come from — an explicit path or a Hugging Face /
ModelScope repo — and how they move through offload / onload /
preparing / computation devices. :meth:`ModelConfig.download_if_necessary`
materializes the source and, under USP, downloads only on rank zero.

This is placement and download policy, not the file deserializer
(:mod:`worldfoundry.core.model_loading.checkpoints.file`).
"""

import glob
import os
from dataclasses import dataclass
from typing import cast

import torch

from worldfoundry.core.io.paths import local_model_root_path

# ──────────────────────────────────────────────────────────────────────────
# Placement + download policy — not the file deserializer (see .file)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Resolved model source, download policy, and VRAM placement settings.

    Args:
        path: Existing checkpoint path or paths. When set, no model-hub lookup
            is needed.
        model_id: Hugging Face or ModelScope repository identifier.
        origin_file_pattern: Optional allow-pattern within the model repository.
        download_source: ``"huggingface"`` or ``"modelscope"``; defaults from
            ``WORLDFOUNDRY_DOWNLOAD_SOURCE`` and then to Hugging Face.
        local_model_path: Local repository cache root. Defaults through
            ``WORLDFOUNDRY_MODEL_DIR``.
        skip_download: Reuse local files without contacting the model hub.
        offload_device: Device used while weights are inactive.
        offload_dtype: Dtype used while weights are inactive.
        onload_device: First-stage prefetch device.
        onload_dtype: First-stage prefetch dtype.
        preparing_device: Second-stage prefetch device.
        preparing_dtype: Second-stage prefetch dtype.
        computation_device: Device used for forward execution.
        computation_dtype: Dtype used for forward execution.
        clear_parameters: Compatibility flag for loaders that release source
            tensors after assignment.
        state_dict: Optional already-loaded weights, bypassing file loading.
    """

    path: str | list[str] | None = None
    model_id: str | None = None
    origin_file_pattern: str | None = None
    download_source: str | None = None
    local_model_path: str | None = None
    skip_download: bool | None = None
    offload_device: str | torch.device | None = None
    offload_dtype: torch.dtype | None = None
    onload_device: str | torch.device | None = None
    onload_dtype: torch.dtype | None = None
    preparing_device: str | torch.device | None = None
    preparing_dtype: torch.dtype | None = None
    computation_device: str | torch.device | None = None
    computation_dtype: torch.dtype | None = None
    clear_parameters: bool = False
    state_dict: dict[str, torch.Tensor] | None = None

    def check_input(self):
        """Require either a concrete path or a model-hub identifier."""
        if self.path is None and self.model_id is None:
            raise ValueError(
                """No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`. `skip_download=True` only supports the first one."""
            )

    def parse_original_file_pattern(self):
        """Normalize the repository allow-pattern to a glob-compatible value."""
        if self.origin_file_pattern in [None, "", "./"]:
            return "*"
        elif self.origin_file_pattern.endswith("/"):
            return self.origin_file_pattern + "*"
        else:
            return self.origin_file_pattern

    def parse_download_source(self):
        """Resolve the configured model hub, including the environment override."""
        if self.download_source is None:
            if os.environ.get("WORLDFOUNDRY_DOWNLOAD_SOURCE") is not None:
                return os.environ["WORLDFOUNDRY_DOWNLOAD_SOURCE"]
            else:
                return "huggingface"
        else:
            return self.download_source

    def parse_skip_download(self) -> bool:
        """Resolve offline behavior from the field or environment.

        Accepts the common boolean spellings; an unparseable value raises
        instead of silently falling through to "download anyway", which would
        hang offline clusters whose users set e.g. ``=1`` expecting a skip.
        """
        if self.skip_download is not None:
            return self.skip_download
        raw = os.environ.get("WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD")
        if raw is None:
            return False
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"", "0", "false", "no", "n", "off"}:
            return False
        raise ValueError(
            f"WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD must be a boolean value (true/false, 1/0, yes/no, on/off); got {raw!r}"
        )

    def download(self):
        """Download missing repository files into ``local_model_path``."""
        origin_file_pattern = self.parse_original_file_pattern()
        # download_if_necessary() validates model_id and resolves local_model_path first.
        model_id = cast(str, self.model_id)
        local_dir = os.path.join(cast(str, self.local_model_path), model_id)
        downloaded_files = glob.glob(origin_file_pattern, root_dir=local_dir)
        download_source = self.parse_download_source()
        if download_source.lower() == "modelscope":
            snapshot_download = __import__("modelscope", fromlist=["snapshot_download"]).snapshot_download

            snapshot_download(
                model_id,
                local_dir=local_dir,
                allow_file_pattern=origin_file_pattern,
                ignore_file_pattern=downloaded_files,
                local_files_only=False,
            )
        elif download_source.lower() == "huggingface":
            hf_snapshot_download = __import__("huggingface_hub", fromlist=["snapshot_download"]).snapshot_download

            hf_snapshot_download(
                model_id,
                local_dir=local_dir,
                allow_patterns=origin_file_pattern,
                ignore_patterns=downloaded_files,
                local_files_only=False,
            )
        else:
            raise ValueError("`download_source` should be `modelscope` or `huggingface`.")

    def require_downloading(self, use_usp: bool = False):
        """Return whether this rank should contact the configured model hub."""
        if self.path is not None:
            return False
        skip_download = self.parse_skip_download()
        if use_usp:
            import torch.distributed as dist

            skip_download = skip_download or dist.get_rank() != 0
        return not skip_download

    def reset_local_model_path(self):
        """Apply the canonical WorldFoundry model directory when none is set.

        An explicitly configured ``local_model_path`` always wins;
        ``WORLDFOUNDRY_MODEL_DIR`` is consumed by ``local_model_root_path()``
        and only fills the default (explicit argument > environment > default).
        """
        if self.local_model_path is None:
            self.local_model_path = str(local_model_root_path())

    def download_if_necessary(self, use_usp: bool = False):
        """Materialize the configured source and replace ``path`` with local files.

        In USP execution only rank zero downloads; all ranks synchronize before
        the final local path is resolved.
        """
        self.check_input()
        self.reset_local_model_path()
        if self.require_downloading(use_usp=use_usp):
            self.download()
        if use_usp:
            import torch.distributed as dist

            # ``device_ids`` expects the *local* CUDA device index, not the
            # global rank (they only coincide on a single node). Bind the
            # barrier to this rank's current device; on CPU-only process
            # groups the argument is unsupported, so omit it.
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        if self.path is None:
            # check_input() guarantees model_id when no path was given.
            local_dir = os.path.join(cast(str, self.local_model_path), cast(str, self.model_id))
            if self.origin_file_pattern in [None, "", "./"]:
                self.path = local_dir
            else:
                self.path = glob.glob(os.path.join(local_dir, cast(str, self.origin_file_pattern)))
        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]

    def vram_config(self):
        """Return placement fields in the mapping expected by VRAM wrappers."""
        return {
            "offload_device": self.offload_device,
            "offload_dtype": self.offload_dtype,
            "onload_device": self.onload_device,
            "onload_dtype": self.onload_dtype,
            "preparing_device": self.preparing_device,
            "preparing_dtype": self.preparing_dtype,
            "computation_device": self.computation_device,
            "computation_dtype": self.computation_dtype,
        }


# ──────────────────────────────────────────────────────────────────────────
# Compatibility alias — prefer LoaderModelConfig in new code
# ──────────────────────────────────────────────────────────────────────────

# Prefer this unambiguous name in new code. ``ModelConfig`` remains for
# compatibility with existing loader configs and serialized targets.
LoaderModelConfig = ModelConfig


__all__ = ["LoaderModelConfig", "ModelConfig"]
