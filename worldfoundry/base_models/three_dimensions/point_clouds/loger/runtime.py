import inspect
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download

from worldfoundry.base_models.three_dimensions.point_clouds.loger.pi3 import Pi3
from worldfoundry.base_models.three_dimensions.point_clouds.loger.utils.geometry import (
    depth_edge,
)
from worldfoundry.core.model_loading import load_torch_checkpoint


class LoGeRRepresentation:
    """
    Representation class for the LoGeR model.

    Supports windowed inference with optional TTT (Test-Time Training),
    SWA (Sliding Window Attention) adapters, and Sim3/SE3 alignment.

    Expected input via get_representation():
        data["images"]  : torch.Tensor of shape (B, N, C, H, W), values in [0, 1]

    Optional inference controls in data (override config defaults if provided):
        window_size     : int   sliding window size (-1 = full sequence)
        overlap_size    : int   overlap between consecutive windows
        sim3            : bool  enable Sim3 scale alignment across windows
        se3             : bool  enable SE3 (no scale) alignment across windows
        reset_every     : int   reset TTT state every N windows (0 = never)
        turn_off_ttt    : bool  disable TTT even if layers exist
        turn_off_swa    : bool  disable SWA even if layers exist
        sim3_scale_mode : str   one of median / trimmed_mean / median_all / ...
        num_iterations  : int   number of TTT decode iterations per window
        conf_threshold  : float confidence sigmoid threshold for mask (default 0.1)
        edge_rtol       : float relative tolerance for depth-edge filter (default 0.03)
    """
    def __init__(
        self,
        model: Optional[Pi3] = None,
        device: Optional[str] = None,
        inference_defaults: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_type = "loger"
        # Inference-time forward() defaults parsed from original_config.yaml.
        # Keys here serve as fallbacks; values in data dict always take priority.
        self.inference_defaults: Dict[str, Any] = inference_defaults or {}

        if self.model is not None:
            self.model = self.model.to(self.device).eval()

        if self.device == "cuda" and torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        **kwargs,
    ) -> "LoGeRRepresentation":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Resolve paths ──────────────────────────────────────────
        if not os.path.exists(pretrained_model_path):
            from huggingface_hub import hf_hub_download

            ckpt_filename   = f"{subfolder}/latest.pt"            if subfolder else "latest.pt"
            config_filename = f"{subfolder}/original_config.yaml" if subfolder else "original_config.yaml"

            # 先查本地缓存，没有再联网
            try:
                ckpt_file = hf_hub_download(
                    repo_id=pretrained_model_path,
                    filename=ckpt_filename,
                    local_files_only=True,
                )
            except Exception:
                ckpt_file = hf_hub_download(
                    repo_id=pretrained_model_path,
                    filename=ckpt_filename,
                )
            print(f"Checkpoint: {ckpt_file}")
            model_root = os.path.dirname(ckpt_file)

            if config_path is None:
                try:
                    config_path = hf_hub_download(
                        repo_id=pretrained_model_path,
                        filename=config_filename,
                        local_files_only=True,
                    )
                except Exception:
                    try:
                        config_path = hf_hub_download(
                            repo_id=pretrained_model_path,
                            filename=config_filename,
                        )
                    except Exception:
                        pass

        elif os.path.isfile(pretrained_model_path):
            ckpt_file = pretrained_model_path
            model_root = os.path.dirname(ckpt_file)
            if config_path is None:
                candidate = os.path.join(model_root, "original_config.yaml")
                config_path = candidate if os.path.exists(candidate) else None

        else:
            model_root = pretrained_model_path
            if subfolder and not os.path.isfile(os.path.join(model_root, "latest.pt")):
                model_root = os.path.join(model_root, subfolder)
            ckpt_file = os.path.join(model_root, "latest.pt")
            if not os.path.exists(ckpt_file):
                pts = sorted(f for f in os.listdir(model_root) if f.endswith(".pt"))
                if not pts:
                    raise FileNotFoundError(f"No .pt checkpoint found in {model_root}")
                ckpt_file = os.path.join(model_root, pts[0])
            if config_path is None:
                candidate = os.path.join(model_root, "original_config.yaml")
                config_path = candidate if os.path.exists(candidate) else None

        # ── Parse config ───────────────────────────────────────────
        model_kwargs: Dict[str, Any] = {}
        inference_defaults: Dict[str, Any] = {}

        if config_path:
            model_kwargs, inference_defaults = cls._parse_config(config_path)

        model_kwargs.update(kwargs)

        # ── Instantiate and load ───────────────────────────────────
        model = Pi3(**model_kwargs)

        print(f"Loading checkpoint from {ckpt_file} ...")
        checkpoint = load_torch_checkpoint(ckpt_file, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully.")

        return cls(model=model, device=device, inference_defaults=inference_defaults)


    @staticmethod
    def _to_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        """Move a tensor to CPU and convert to float32 numpy array."""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
        return np.asarray(tensor, dtype=np.float32)

    @staticmethod
    def _parse_config(config_path: str):
        """
        Parse original_config.yaml and return:
            (model_kwargs, inference_defaults)

        model_kwargs      : valid Pi3.__init__ parameters
        inference_defaults: forward() parameters (se3, window_size, overlap_size, ...)
        """
        model_kwargs: Dict[str, Any] = {}
        inference_defaults: Dict[str, Any] = {}

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # ── Model init kwargs ──────────────────────────────────
            model_cfg = config.get("model", {})
            pi3_sig   = inspect.signature(Pi3.__init__)
            valid_init_keys = {
                name
                for name, param in pi3_sig.parameters.items()
                if name not in {"self", "args", "kwargs"}
                and param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            def _maybe_parse_sequence(value):
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        try:
                            parsed = yaml.safe_load(stripped)
                            if isinstance(parsed, (list, tuple)):
                                return list(parsed)
                        except Exception:
                            pass
                return value

            for key in sorted(valid_init_keys):
                if key in model_cfg:
                    value = model_cfg[key]
                    if key in {"ttt_insert_after", "attn_insert_after"}:
                        value = _maybe_parse_sequence(value)
                    model_kwargs[key] = value

            # ── Inference / forward() defaults ────────────────────
            # Priority: model section > training_settings section > top-level
            training_cfg = config.get("training_settings", {})

            def _get(key, default):
                """Look up key in model_cfg, then training_cfg, then top-level config."""
                if key in model_cfg:
                    return model_cfg[key]
                if key in training_cfg:
                    return training_cfg[key]
                return config.get(key, default)

            inference_defaults = {
                "se3":             bool(_get("se3",             True)),
                "sim3":            bool(_get("sim3",            False)),
                "window_size":     int(_get("window_size",      32)),
                "overlap_size":    int(_get("overlap_size",     3)),
                "reset_every":     int(_get("reset_every",      0)),
                "num_iterations":  int(_get("num_iterations",   1)),
                "sim3_scale_mode": str(_get("sim3_scale_mode",  "median")),
                "turn_off_ttt":    bool(_get("turn_off_ttt",    False)),
                "turn_off_swa":    bool(_get("turn_off_swa",    False)),
            }

        except Exception as exc:
            print(f"Warning: could not parse config {config_path}: {exc}. "
                  "Using default Pi3 init parameters.")

        return model_kwargs, inference_defaults

    @torch.no_grad()
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run LoGeR inference and return all scene representation outputs.

        Args:
            data : dict with at minimum:
                "images" : torch.Tensor (B, N, C, H, W) in [0, 1]

        Returns:
            dict with numpy arrays:
                points        : (B, N, H, W, 3)  global 3-D point cloud
                local_points  : (B, N, H, W, 3)  camera-frame point cloud
                camera_poses  : (B, N, 4, 4)      camera-to-world SE3
                conf          : (B, N, H, W, 1)   raw confidence logits
                masks         : (B, N, H, W)       binary quality mask
                depth_map     : (B, N, H, W)       z-depth (local frame)

            Plus any extra keys forwarded directly from the model output, e.g.:
                avg_gate_scale, attn_gate_scale,
                chunk_sim3_scales, chunk_sim3_poses / chunk_se3_poses,
                alignment_mode
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() first.")

        imgs = data["images"]
        if not isinstance(imgs, torch.Tensor):
            raise TypeError(
                f"data['images'] must be a torch.Tensor, got {type(imgs)}"
            )
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)   # add batch dim if missing
        imgs = imgs.to(self.device)

        # ── Build forward kwargs ───────────────────────────────────
        # Priority (highest → lowest):
        #   1. values explicitly set in `data`
        #   2. self.inference_defaults  (parsed from original_config.yaml)
        #   3. hard-coded fallbacks below
        def _get(key, fallback):
            if key in data:
                return data[key]
            if key in self.inference_defaults:
                return self.inference_defaults[key]
            return fallback

        forward_kwargs = dict(
            window_size      = int(_get("window_size",      32)),
            overlap_size     = int(_get("overlap_size",      3)),
            num_iterations   = int(_get("num_iterations",    1)),
            sim3             = bool(_get("sim3",             False)),
            se3              = bool(_get("se3",              False)),
            reset_every      = int(_get("reset_every",       0)),
            turn_off_ttt     = bool(_get("turn_off_ttt",     False)),
            turn_off_swa     = bool(_get("turn_off_swa",     False)),
            sim3_scale_mode  = str(_get("sim3_scale_mode",   "median")),
        )

        conf_threshold = float(_get("conf_threshold", 0.1))
        edge_rtol      = float(_get("edge_rtol",      0.03))

        # ── Forward pass ───────────────────────────────────────────
        autocast_enabled = (self.device == "cuda")
        with torch.amp.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
            raw = self.model(imgs, **forward_kwargs)

        # ── Core geometry outputs ──────────────────────────────────
        results: Dict[str, Any] = {}
        results["points"]       = self._to_numpy(raw.get("points"))
        results["local_points"] = self._to_numpy(raw.get("local_points"))
        results["camera_poses"] = self._to_numpy(raw.get("camera_poses"))
        results["conf"]         = self._to_numpy(raw.get("conf"))

        # ── Quality mask: sigmoid(conf) > threshold AND non-depth-edge ─
        conf_tensor = raw.get("conf")
        if conf_tensor is not None:
            conf_prob = torch.sigmoid(conf_tensor[..., 0])
            masks     = conf_prob > conf_threshold
            lp = raw.get("local_points")
            if lp is not None:
                non_edge = ~depth_edge(lp[..., 2], rtol=edge_rtol)
                masks    = torch.logical_and(masks, non_edge)
            results["masks"]     = masks.cpu().numpy()
            results["depth_map"] = self._to_numpy(
                lp[..., 2] if lp is not None else None
            )
        else:
            # use_conf=False model: accept everything
            B, N, C, H, W = imgs.shape
            results["masks"]     = np.ones((B, N, H, W), dtype=bool)
            lp = raw.get("local_points")
            results["depth_map"] = self._to_numpy(lp[..., 2]) if lp is not None else None

        # ── Optional / diagnostic outputs ─────────────────────────
        _scalar_keys = ("avg_gate_scale", "attn_gate_scale", "alignment_mode")
        _tensor_keys = (
            "chunk_sim3_scales", "chunk_sim3_poses", "chunk_se3_poses",
            "metric", "local_camera_poses", "camera_qvec",
            "overlap_prev_cam", "overlap_next_cam",
            "overlap_prev_pcd", "overlap_next_pcd", "overlap_next_conf",
        )

        for key in _scalar_keys:
            if key in raw and raw[key] is not None:
                val = raw[key]
                results[key] = val.item() if isinstance(val, torch.Tensor) else val

        for key in _tensor_keys:
            if key in raw and raw[key] is not None:
                results[key] = self._to_numpy(raw[key])

        return results
