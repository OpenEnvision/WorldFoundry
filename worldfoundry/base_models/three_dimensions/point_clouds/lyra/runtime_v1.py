from __future__ import annotations

import copy
import importlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import (
    _candidate_lyra1_weights_subdir,
    default_local_lyra1_checkpoint_root,
    ensure_repo_on_path,
    maybe_set_cuda_device,
    resolve_lyra1_repo_root,
    resolve_path,
)


class Lyra1Representation:
    """Programmatic wrapper around Lyra-1 `sample.py` reconstruction."""

    def __init__(
        self,
        repo_root: str,
        static_ckpt_path: str,
        dynamic_ckpt_path: str,
        static_config_path: str,
        dynamic_config_path: str,
        inference_config_path: str,
        device: str = "cuda",
    ):
        self.repo_root = repo_root
        self.static_ckpt_path = static_ckpt_path
        self.dynamic_ckpt_path = dynamic_ckpt_path
        self.static_config_path = static_config_path
        self.dynamic_config_path = dynamic_config_path
        self.inference_config_path = inference_config_path
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        device=None,
        static_ckpt_path: Optional[str] = None,
        dynamic_ckpt_path: Optional[str] = None,
        static_config_path: Optional[str] = None,
        dynamic_config_path: Optional[str] = None,
        inference_config_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ):
        repo_root = resolve_lyra1_repo_root(pretrained_model_path)
        checkpoint_root = checkpoint_dir or default_local_lyra1_checkpoint_root()
        if checkpoint_root is None:
            raise FileNotFoundError("Lyra-1 reconstruction requires local Lyra checkpoints.")
        checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        weights_root = _candidate_lyra1_weights_subdir(checkpoint_root) or checkpoint_root / "Lyra"
        return cls(
            repo_root=repo_root,
            static_ckpt_path=resolve_path(
                static_ckpt_path or str(weights_root / "lyra_static.pt"),
                repo_root,
            ),
            dynamic_ckpt_path=resolve_path(
                dynamic_ckpt_path or str(weights_root / "lyra_dynamic.pt"),
                repo_root,
            ),
            static_config_path=resolve_path(
                static_config_path or "configs/demo/lyra_static.yaml",
                repo_root,
            ),
            dynamic_config_path=resolve_path(
                dynamic_config_path or "configs/demo/lyra_dynamic.yaml",
                repo_root,
            ),
            inference_config_path=resolve_path(
                inference_config_path or "configs/inference/default.yaml",
                repo_root,
            ),
            device=device or "cuda",
        )

    def get_representation(self, data: Dict[str, object]) -> Dict[str, object]:
        mode = str(data.get("mode", "static")).lower()
        if mode not in {"static", "dynamic"}:
            raise ValueError(f"Unsupported Lyra-1 mode: {mode}")

        generated_root = Path(str(data["generated_root"])).expanduser().resolve()
        if not generated_root.exists():
            raise FileNotFoundError(f"Lyra-1 generated root not found: {generated_root}")

        ckpt_path = Path(
            str(data.get("ckpt_path") or self._default_ckpt_path(mode))
        ).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Lyra-1 reconstruction checkpoint not found: {ckpt_path}")

        demo_config_path = Path(
            str(data.get("demo_config_path") or self._default_demo_config(mode))
        ).expanduser().resolve()
        if not demo_config_path.exists():
            raise FileNotFoundError(f"Lyra-1 demo config not found: {demo_config_path}")

        inference_config_path = Path(
            str(data.get("inference_config_path") or self.inference_config_path)
        ).expanduser().resolve()
        if not inference_config_path.exists():
            raise FileNotFoundError(f"Lyra-1 inference config not found: {inference_config_path}")

        output_dir = Path(
            str(data.get("output_dir") or generated_root.parent / f"{generated_root.name}_reconstruction")
        ).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        ensure_repo_on_path(self.repo_root)
        maybe_set_cuda_device(self.device)
        try:
            sample_module = importlib.import_module("sample")
            dataset_registry_module = importlib.import_module("src.models.eval_inputs.registry")
            misc_module = importlib.import_module("src.models.utils.misc")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Lyra-1 runtime dependencies are missing. "
                "Install the official Lyra-1 requirements before calling get_representation(). "
                f"Original error: {error}"
            ) from error

        dataset_registry = dataset_registry_module.dataset_registry
        load_and_merge_configs = misc_module.load_and_merge_configs

        runtime_dataset_name = str(
            data.get("dataset_name")
            or f"worldbench_lyra1_{mode}_{uuid.uuid4().hex[:8]}"
        )
        template_key = (
            "lyra_static_template_generated" if mode == "static" else "lyra_dynamic_template_generated"
        )
        runtime_entry = copy.deepcopy(dataset_registry[template_key])
        runtime_entry["kwargs"]["root_path"] = str(generated_root)
        dataset_registry[runtime_dataset_name] = runtime_entry

        try:
            config = load_and_merge_configs(
                [str(inference_config_path), str(demo_config_path)]
            )
            config.dataset_name = runtime_dataset_name
            config.out_dir_inference = str(output_dir)
            config.ckpt_path = str(ckpt_path)
            config.config_path = self._resolve_training_config_paths(
                config.config_path,
                data.get("training_config_paths"),
            )
            for field_name in [
                "static_view_indices_fixed",
                "target_index_subsample",
                "set_manual_time_idx",
                "target_index_manual",
                "target_index_manual_start_idx",
                "target_index_manual_num_idx",
                "target_index_manual_stride",
                "num_test_images",
                "do_eval",
                "use_depth",
                "out_fps",
                "save_grid",
                "num_grid_samples",
                "save_gt_input",
                "save_video_input",
                "save_gt_depth",
                "save_rgb_decoding",
                "save_gaussians",
                "save_gaussians_orig",
                "skip_existing",
                "ckpt_name",
            ]:
                if field_name in data and data[field_name] is not None:
                    setattr(config, field_name, data[field_name])

            previous_cwd = Path.cwd()
            try:
                os.chdir(self.repo_root)
                try:
                    sample_module.main(config)
                except ModuleNotFoundError as error:
                    raise ModuleNotFoundError(
                        "Lyra-1 reconstruction runtime dependencies are missing. "
                        "Install the official Lyra-1 environment before calling get_representation(). "
                        f"Original error: {error}"
                    ) from error
            finally:
                os.chdir(previous_cwd)
        finally:
            dataset_registry.pop(runtime_dataset_name, None)

        base_output_dir, render_roots = self._expected_output_dirs(
            out_dir_inference=output_dir,
            ckpt_path=ckpt_path,
            ckpt_name=getattr(config, "ckpt_name", None),
            dataset_name=runtime_dataset_name,
            target_index_manual=getattr(config, "target_index_manual", None),
        )
        render_results = [self._collect_render_result(root) for root in render_roots]
        result: Dict[str, Any] = {
            "mode": mode,
            "generated_root": str(generated_root),
            "output_dir": str(base_output_dir),
            "render_roots": [str(root) for root in render_roots],
            "render_results": render_results,
            "dataset_name": runtime_dataset_name,
            "checkpoint_path": str(ckpt_path),
            "demo_config_path": str(demo_config_path),
        }
        if len(render_results) == 1:
            result.update(render_results[0])
        return result

    def _default_ckpt_path(self, mode: str) -> str:
        return self.static_ckpt_path if mode == "static" else self.dynamic_ckpt_path

    def _default_demo_config(self, mode: str) -> str:
        return self.static_config_path if mode == "static" else self.dynamic_config_path

    def _resolve_training_config_paths(self, config_value, override_value):
        source = override_value if override_value is not None else config_value
        if isinstance(source, str):
            return resolve_path(source, self.repo_root)
        if source is not None and not isinstance(source, dict):
            try:
                return [resolve_path(path_value, self.repo_root) for path_value in list(source)]
            except TypeError:
                return source
        return source

    @staticmethod
    def _extract_checkpoint_name(ckpt_path: Path, ckpt_name: Optional[str]) -> Optional[str]:
        if ckpt_name:
            return str(ckpt_name)
        match = re.search(r"(checkpoint-\d+)", str(ckpt_path))
        if match:
            return match.group(1)
        return None

    def _expected_output_dirs(
        self,
        out_dir_inference: Path,
        ckpt_path: Path,
        ckpt_name: Optional[str],
        dataset_name: str,
        target_index_manual,
    ):
        base_output_dir = Path(out_dir_inference)
        checkpoint_name = self._extract_checkpoint_name(ckpt_path, ckpt_name)
        if checkpoint_name:
            base_output_dir = base_output_dir / checkpoint_name
        base_output_dir = base_output_dir / dataset_name

        if isinstance(target_index_manual, int):
            render_roots = [base_output_dir / str(target_index_manual)]
        elif target_index_manual is not None and not isinstance(target_index_manual, (str, bytes)):
            try:
                render_roots = [base_output_dir / str(index) for index in list(target_index_manual)]
            except TypeError:
                render_roots = [base_output_dir]
        else:
            render_roots = [base_output_dir]
        return base_output_dir, render_roots

    @staticmethod
    def _collect_render_result(root: Path) -> Dict[str, Any]:
        return {
            "output_dir": str(root),
            "main_render_dir": str(root / "main_gaussians_renderings"),
            "full_output_dir": str(root / "full_output"),
            "raw_output_dir": str(root / "raw"),
            "grid_output_dir": str(root / "grid"),
            "meta_dir": str(root / "meta"),
            "gaussians_dir": str(root / "gaussians"),
            "gaussians_orig_dir": str(root / "gaussians_orig"),
            "main_render_videos": [str(path.resolve()) for path in sorted((root / "main_gaussians_renderings").glob("*.mp4"))],
            "full_output_videos": [str(path.resolve()) for path in sorted((root / "full_output").glob("*.mp4"))],
            "grid_videos": [str(path.resolve()) for path in sorted((root / "grid").glob("*.mp4"))],
            "gaussian_ply_paths": [str(path.resolve()) for path in sorted((root / "gaussians").glob("*.ply"))],
            "gaussian_orig_ply_paths": [str(path.resolve()) for path in sorted((root / "gaussians_orig").glob("*.ply"))],
        }
