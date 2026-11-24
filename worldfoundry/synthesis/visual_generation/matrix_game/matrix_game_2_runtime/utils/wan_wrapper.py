import inspect
import types
from collections import OrderedDict

import torch
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.action_21 import _video_vae
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_21 import (
    WanModel,  # noqa: F401 - Keeps the module's import-time setup.
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_action_21 import (
    CausalWanModel,
)
from worldfoundry.core.nn.diffusion.schedulers import SchedulerInterface

from .scheduler import FlowMatchScheduler


class WanVAEWrapper(torch.nn.Module):  # todo
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = (
            _video_vae(
                pretrained_path="skyreels_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]

        decode_function = self.model.cached_decode if use_cache else self.model.decode

        output = []
        for u in latent:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_config="",
        timestep_shift=5.0,
        is_causal=True,
    ):
        super().__init__()
        print(model_config)
        self.model = self._build_model_from_config(CausalWanModel, model_config)
        self.eval()
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 15 * 880  # 32760  # [1, 15, 16, 60, 104]
        self._model_dtype_cache: torch.dtype | None = None
        self._f64_schedule_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self._sigma_lookup_cache: dict[tuple[torch.device, float], torch.Tensor] = {}
        self._block_graph_enabled: bool = False
        self._block_graph_warmup: int = 1
        self._block_graph_max_runners: int = 4
        self._block_graph_runners: OrderedDict[tuple, object] = OrderedDict()
        self._attention_backend: str = "auto"
        self._exact_norms: bool = True
        self._sinusoid_freqs_cache: dict[torch.device, torch.Tensor] = {}
        self._master_t_freqs: torch.Tensor | None = None
        self._master_action_freqs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._shared_action_freqs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._kv_roll_scratch: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._static_crossattn_cache: list[dict] | None = None
        self._last_xkv_refs: tuple[torch.Tensor, ...] | None = None
        self._last_xkv_versions: tuple[int, ...] | None = None
        self._static_crossattn_dirty: bool = True
        self.post_init()

    def _apply(self, fn):
        self._model_dtype_cache = None
        if getattr(self, "_block_graph_runners", None):
            self.invalidate_block_cuda_graphs()
        return super()._apply(fn)

    def set_attention_backend(self, backend: str | None) -> None:
        """Select self-attention backend for fast block execution (``auto``, ``flash_attention_3``, ``torch``)."""
        resolved = "auto" if backend is None else str(backend).strip().casefold()
        if resolved in {"fa3", "flash3", "flash_attn_3", "flash_attention_3"}:
            resolved = "flash_attention_3"
        elif resolved in {"sdpa", "torch"}:
            resolved = "torch"
        elif resolved != "auto":
            raise ValueError(f"Unsupported Matrix-Game-2 attention backend: {backend!r}")
        if resolved != self._attention_backend:
            self._attention_backend = resolved
            self._exact_norms = resolved != "flash_attention_3"
            if getattr(self, "_block_graph_runners", None):
                self.invalidate_block_cuda_graphs()

    def _resolved_attention_backend(self, device: torch.device) -> str:
        from .fast_kernels import hopper_fa3_usable

        if self._attention_backend == "flash_attention_3":
            return "flash_attention_3" if hopper_fa3_usable(device) else "torch"
        # Default "auto" and "torch" preserve >60 dB / PSNR=inf lossless parity in BF16.
        return "torch"

    @staticmethod
    def _canonical_contiguous(t: torch.Tensor) -> torch.Tensor:
        expected = []
        acc = 1
        for dim in reversed(t.shape):
            expected.append(acc)
            acc *= dim
        expected.reverse()
        if tuple(t.stride()) == tuple(expected):
            return t
        return t.clone(memory_format=torch.contiguous_format)

    def _get_sinusoid_freqs(self, device: torch.device) -> torch.Tensor:
        cached = self._sinusoid_freqs_cache.get(device)
        if cached is None:
            half_dim = int(self.model.freq_dim) // 2
            cached = torch.pow(
                10000.0,
                -torch.arange(half_dim, device=device, dtype=torch.float64).div(half_dim),
            )
            self._sinusoid_freqs_cache[device] = cached
        return cached

    def _ensure_crossattn_cache_initialized(
        self,
        visual_context: torch.Tensor,
        crossattn_cache: list[dict],
    ) -> None:
        """Populate MG2 image cross-attention K/V cache once per session."""
        if crossattn_cache and crossattn_cache[0].get("is_init", False):
            return
        context = self.model.img_emb(visual_context)
        batch = int(context.shape[0])
        for block, slot in zip(self.model.blocks, crossattn_cache, strict=False):
            ca = block.cross_attn
            n, d = int(ca.num_heads), int(ca.head_dim)
            k = ca.norm_k(ca.k(context)).view(batch, -1, n, d)
            v = ca.v(context).view(batch, -1, n, d)
            with torch.inference_mode(False):
                k_buf = torch.empty(k.shape, dtype=k.dtype, device=k.device)
                v_buf = torch.empty(v.shape, dtype=v.dtype, device=v.device)
                k_buf.copy_(k)
                v_buf.copy_(v)
            slot["k"] = k_buf
            slot["v"] = v_buf
            slot["is_init"] = True

    def enable_block_cuda_graph(self, *, warmup: int = 1) -> None:
        """Enable virtual-window steady-state CUDA Graph capture for causal block steps."""
        self._block_graph_enabled = True
        self._block_graph_warmup = int(warmup)
        self._ensure_master_rope_tables()

    def disable_block_cuda_graph(self) -> None:
        """Disable block CUDA Graph capture, release graphs, and restore master RoPE tables."""
        self._block_graph_enabled = False
        self.invalidate_block_cuda_graphs()
        self._restore_master_rope_tables()

    def invalidate_block_cuda_graphs(self) -> None:
        for runner in self._block_graph_runners.values():
            runner.invalidate()
        self._block_graph_runners.clear()
        self._kv_roll_scratch.clear()
        self._static_crossattn_cache = None
        self._last_xkv_refs = None
        self._last_xkv_versions = None
        self._static_crossattn_dirty = True

    def block_graph_report(self) -> dict | None:
        if not self._block_graph_enabled and not self._block_graph_runners:
            return None
        captures = replays = eager = failures = 0
        for runner in self._block_graph_runners.values():
            rep = runner.report()
            captures += int(rep.get("capture", 0))
            replays += int(rep.get("replay", 0))
            eager += int(rep.get("eager", 0))
            failures += int(rep.get("capture_failed", 0))
        return {
            "enabled": self._block_graph_enabled,
            "attention_backend": self._attention_backend,
            "runner_count": len(self._block_graph_runners),
            "capture": captures,
            "replay": replays,
            "eager": eager,
            "capture_failed": failures,
        }

    def _ensure_master_rope_tables(self) -> None:
        device = next(self.model.parameters()).device
        if hasattr(self.model, "freqs"):
            if self.model.freqs.device != device:
                self.model.freqs = self.model.freqs.to(device)
            c = self.model.freqs.shape[1]
            t_dim = c - 2 * (c // 3)
            if self._master_t_freqs is None:
                self._master_t_freqs = self.model.freqs[:, :t_dim].clone()
            elif self._master_t_freqs.device != device:
                self._master_t_freqs = self._master_t_freqs.to(device)
        blocks = getattr(self.model, "blocks", ())
        for block in blocks:
            am = getattr(block, "action_model", None)
            if am is None or not hasattr(am, "freqs_cos"):
                continue
            if self._master_action_freqs is None:
                cos_gpu = am.freqs_cos.to(device=device).contiguous()
                sin_gpu = am.freqs_sin.to(device=device).contiguous()
                self._master_action_freqs = (cos_gpu.clone(), sin_gpu.clone())
                self._shared_action_freqs = (cos_gpu, sin_gpu)
            elif self._master_action_freqs[0].device != device:
                master_cos = self._master_action_freqs[0].to(device=device).contiguous()
                master_sin = self._master_action_freqs[1].to(device=device).contiguous()
                self._master_action_freqs = (master_cos, master_sin)
                self._shared_action_freqs = (master_cos.clone(), master_sin.clone())
            am.freqs_cos, am.freqs_sin = self._shared_action_freqs

    def _restore_master_rope_tables(self) -> None:
        if self._master_t_freqs is not None and hasattr(self.model, "freqs"):
            t_dim = self._master_t_freqs.shape[1]
            self.model.freqs[:, :t_dim].copy_(self._master_t_freqs)
        if self._master_action_freqs is not None and self._shared_action_freqs is not None:
            self._shared_action_freqs[0].copy_(self._master_action_freqs[0])
            self._shared_action_freqs[1].copy_(self._master_action_freqs[1])

    def _sync_virtual_rope_window(self, actual_start: int, virtual_start: int, num_frames: int) -> None:
        if self._master_t_freqs is not None:
            t_dim = self._master_t_freqs.shape[1]
            self.model.freqs[virtual_start : virtual_start + num_frames, :t_dim].copy_(
                self._master_t_freqs[actual_start : actual_start + num_frames]
            )
        if self._master_action_freqs is not None and self._shared_action_freqs is not None:
            self._shared_action_freqs[0][virtual_start : virtual_start + num_frames].copy_(
                self._master_action_freqs[0][actual_start : actual_start + num_frames]
            )
            self._shared_action_freqs[1][virtual_start : virtual_start + num_frames].copy_(
                self._master_action_freqs[1][actual_start : actual_start + num_frames]
            )

    def _roll_kv_caches_if_needed(
        self,
        *,
        kv_cache: list[dict],
        kv_cache_mouse: list[dict] | None,
        kv_cache_keyboard: list[dict] | None,
        current_start: int,
        block_tokens: int,
        block_frames: int,
        frame_seqlen: int,
    ) -> bool:
        """Perform one-time GPU KV rollover on step 0 of steady-state blocks."""
        current_end = current_start + block_tokens
        cached_global = kv_cache[0].get("_host_global_end_index")
        cached_local = kv_cache[0].get("_host_local_end_index")
        if cached_global is None:
            cached_global = int(kv_cache[0]["global_end_index"].item())
            cached_local = int(kv_cache[0]["local_end_index"].item())
        kv_size = kv_cache[0]["k"].shape[1]
        if not (current_end > cached_global and block_tokens + cached_local > kv_size):
            return False

        evicted = block_tokens + cached_local - kv_size
        rolled = cached_local - evicted
        if rolled > 0:
            scratch_key = (
                "main",
                tuple(kv_cache[0]["k"].shape[0:1]),
                rolled,
                kv_cache[0]["k"].dtype,
                kv_cache[0]["k"].device,
            )
            scratch = self._kv_roll_scratch.get(scratch_key)
            if scratch is None:
                k0 = kv_cache[0]["k"]
                with torch.inference_mode(False):
                    scratch = (
                        torch.empty((k0.shape[0], rolled, k0.shape[2], k0.shape[3]), dtype=k0.dtype, device=k0.device),
                        torch.empty((k0.shape[0], rolled, k0.shape[2], k0.shape[3]), dtype=k0.dtype, device=k0.device),
                    )
                self._kv_roll_scratch[scratch_key] = scratch
            k_tmp, v_tmp = scratch
            for slot in kv_cache:
                k_tmp.copy_(slot["k"][:, evicted : evicted + rolled])
                v_tmp.copy_(slot["v"][:, evicted : evicted + rolled])
                slot["k"][:, :rolled].copy_(k_tmp)
                slot["v"][:, :rolled].copy_(v_tmp)

        action_local_size = self.model.local_attn_size
        action_rolled = action_local_size - block_frames
        if action_rolled > 0:
            if kv_cache_mouse:
                m0 = kv_cache_mouse[0]["k"]
                m_key = ("mouse", m0.shape[0], action_rolled, m0.dtype, m0.device)
                m_scratch = self._kv_roll_scratch.get(m_key)
                if m_scratch is None:
                    with torch.inference_mode(False):
                        m_scratch = (
                            torch.empty(
                                (m0.shape[0], action_rolled, m0.shape[2], m0.shape[3]),
                                dtype=m0.dtype,
                                device=m0.device,
                            ),
                            torch.empty(
                                (m0.shape[0], action_rolled, m0.shape[2], m0.shape[3]),
                                dtype=m0.dtype,
                                device=m0.device,
                            ),
                        )
                    self._kv_roll_scratch[m_key] = m_scratch
                mk_tmp, mv_tmp = m_scratch
                for block, slot in zip(self.model.blocks, kv_cache_mouse, strict=False):
                    action_model = getattr(block, "action_model", None)
                    if action_model is None or not getattr(action_model, "enable_mouse", True):
                        continue
                    mk_tmp.copy_(slot["k"][:, block_frames : block_frames + action_rolled])
                    mv_tmp.copy_(slot["v"][:, block_frames : block_frames + action_rolled])
                    slot["k"][:, :action_rolled].copy_(mk_tmp)
                    slot["v"][:, :action_rolled].copy_(mv_tmp)
            if kv_cache_keyboard:
                kb0 = kv_cache_keyboard[0]["k"]
                kb_key = ("keyboard", kb0.shape[0], action_rolled, kb0.dtype, kb0.device)
                kb_scratch = self._kv_roll_scratch.get(kb_key)
                if kb_scratch is None:
                    with torch.inference_mode(False):
                        kb_scratch = (
                            torch.empty(
                                (kb0.shape[0], action_rolled, kb0.shape[2], kb0.shape[3]),
                                dtype=kb0.dtype,
                                device=kb0.device,
                            ),
                            torch.empty(
                                (kb0.shape[0], action_rolled, kb0.shape[2], kb0.shape[3]),
                                dtype=kb0.dtype,
                                device=kb0.device,
                            ),
                        )
                    self._kv_roll_scratch[kb_key] = kb_scratch
                kk_tmp, kv_tmp = kb_scratch
                for block, slot in zip(self.model.blocks, kv_cache_keyboard, strict=False):
                    if getattr(block, "action_model", None) is None:
                        continue
                    kk_tmp.copy_(slot["k"][:, block_frames : block_frames + action_rolled])
                    kv_tmp.copy_(slot["v"][:, block_frames : block_frames + action_rolled])
                    slot["k"][:, :action_rolled].copy_(kk_tmp)
                    slot["v"][:, :action_rolled].copy_(kv_tmp)
        del frame_seqlen
        return True

    @staticmethod
    def _set_cache_positions(
        kv_cache: list[dict],
        kv_cache_mouse: list[dict] | None,
        kv_cache_keyboard: list[dict] | None,
        *,
        main_global: int,
        main_local: int,
        action_global: int,
        action_local: int,
        sync_gpu_tensors: bool = False,
    ) -> None:
        for slot in kv_cache:
            has_host = "_host_global_end_index" in slot
            if has_host:
                slot["_host_global_end_index"] = main_global
                slot["_host_local_end_index"] = main_local
            if sync_gpu_tensors or not has_host:
                slot["global_end_index"].fill_(main_global)
                slot["local_end_index"].fill_(main_local)
        if kv_cache_mouse:
            for slot in kv_cache_mouse:
                has_host = "_host_global_end_index" in slot
                if has_host:
                    slot["_host_global_end_index"] = action_global
                    slot["_host_local_end_index"] = action_local
                if sync_gpu_tensors or not has_host:
                    slot["global_end_index"].fill_(action_global)
                    slot["local_end_index"].fill_(action_local)
        if kv_cache_keyboard:
            for slot in kv_cache_keyboard:
                has_host = "_host_global_end_index" in slot
                if has_host:
                    slot["_host_global_end_index"] = action_global
                    slot["_host_local_end_index"] = action_local
                if sync_gpu_tensors or not has_host:
                    slot["global_end_index"].fill_(action_global)
                    slot["local_end_index"].fill_(action_local)

    @staticmethod
    def _build_model_from_config(model_cls, model_config):
        if hasattr(model_config, "items"):
            signature = inspect.signature(model_cls.__init__)
            accepted = {
                name
                for name, param in signature.parameters.items()
                if name != "self" and param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}
            }
            config = {key: value for key, value in model_config.items() if key in accepted}
            return model_cls(**config)
        return model_cls.from_config(model_config)

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _get_f64_schedule(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._f64_schedule_cache.get(device)
        if cached is None:
            cached = (
                self.scheduler.sigmas.double().to(device),
                self.scheduler.timesteps.double().to(device),
            )
            self._f64_schedule_cache[device] = cached
        return cached

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Convert flow matching velocity prediction to x0 prediction using cached GPU schedule."""
        original_dtype = flow_pred.dtype
        sigmas, timesteps = self._get_f64_schedule(flow_pred.device)
        t_f64 = timestep.double().to(flow_pred.device)
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - t_f64.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt.double() - sigma_t * flow_pred.double()
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = (
            x.double().to(x0_pred.device) for x in [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        kv_cache_mouse: list[dict] | None = None,
        kv_cache_keyboard: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        assert noisy_image_or_video.shape[1] == 16
        # [B, F] -> [B]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep
        logits = None
        # ``dtype`` used to be exposed by older Diffusers ``ModelMixin``
        # versions.  Matrix-Game-2's vendored CausalWanModel no longer has
        # that convenience attribute with the unified runtime's Diffusers,
        # so derive it from the loaded parameters instead.
        model_dtype = self._model_dtype_cache
        if model_dtype is None:
            model_dtype = next(self.model.parameters()).dtype
            self._model_dtype_cache = model_dtype

        if kv_cache is not None:
            use_graph = False
            action_windows_size = 3
            if (
                self._block_graph_enabled
                and noisy_image_or_video.is_cuda
                and not self.model.training
                and not torch.is_grad_enabled()
                and input_timestep.ndim == 2
                and current_start is not None
                and crossattn_cache is not None
                and crossattn_cache[0].get("is_init", False)
                and int(getattr(self.model, "sink_size", 0)) == 0
            ):
                ph, pw = self.model.patch_size[1], self.model.patch_size[2]
                frame_seqlen = (noisy_image_or_video.shape[-2] // ph) * (noisy_image_or_video.shape[-1] // pw)
                block_frames = int(noisy_image_or_video.shape[2])
                block_tokens = block_frames * frame_seqlen
                start_frame = int(current_start) // frame_seqlen
                local_attn_size = int(getattr(self.model, "local_attn_size", -1))
                for blk in getattr(self.model, "blocks", ()):
                    am = getattr(blk, "action_model", None)
                    if am is not None:
                        action_windows_size = int(getattr(am, "windows_size", 3))
                        break
                virtual_start_frame_cand = local_attn_size - block_frames
                kv_cache_size = int(kv_cache[0]["k"].shape[1])
                cached_local_host = kv_cache[0].get("_host_local_end_index")
                cached_local = (
                    int(cached_local_host)
                    if cached_local_host is not None
                    else int(kv_cache[0]["local_end_index"].item())
                )
                if (
                    local_attn_size > block_frames
                    and virtual_start_frame_cand >= action_windows_size
                    and kv_cache_size == local_attn_size * frame_seqlen
                    and start_frame >= virtual_start_frame_cand
                    and cached_local >= virtual_start_frame_cand * frame_seqlen
                    and "keyboard_cond" in conditional_dict
                    and conditional_dict["keyboard_cond"].shape[1] >= 4 * start_frame + 1 + 4 * (block_frames - 1)
                ):
                    use_graph = True

            if use_graph:
                self._ensure_master_rope_tables()
                virtual_start_frame = local_attn_size - block_frames
                virtual_current_start = virtual_start_frame * frame_seqlen
                virtual_end_tokens = local_attn_size * frame_seqlen
                actual_end_tokens = int(current_start) + block_tokens
                actual_end_frame = start_frame + block_frames

                self._roll_kv_caches_if_needed(
                    kv_cache=kv_cache,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    current_start=int(current_start),
                    block_tokens=block_tokens,
                    block_frames=block_frames,
                    frame_seqlen=frame_seqlen,
                )
                self._set_cache_positions(
                    kv_cache,
                    kv_cache_mouse,
                    kv_cache_keyboard,
                    main_global=virtual_end_tokens,
                    main_local=virtual_end_tokens,
                    action_global=local_attn_size,
                    action_local=local_attn_size,
                    sync_gpu_tensors=False,
                )
                self._sync_virtual_rope_window(start_frame, virtual_start_frame, block_frames)

                # Extract the exact (4 * (local_attn_size - 1) + 1)-frame window consumed by
                # ActionModule at virtual_start_frame in canonical contiguous layout so batch-dim=1
                # parent strides do not vary across blocks and trigger recaptures.
                win_start = 4 * (start_frame - virtual_start_frame)
                win_end = 4 * start_frame + 1 + 4 * (block_frames - 1)
                kb_win = self._canonical_contiguous(conditional_dict["keyboard_cond"][:, win_start:win_end])
                has_mouse = "mouse_cond" in conditional_dict and conditional_dict["mouse_cond"] is not None
                mouse_win = (
                    self._canonical_contiguous(conditional_dict["mouse_cond"][:, win_start:win_end])
                    if has_mouse
                    else kb_win.new_empty(0)
                )
                cond_concat = self._canonical_contiguous(conditional_dict["cond_concat"])

                # Copy cross-attention K/V into persistent static buffers so graph pointers
                # remain stable even when a new session re-initializes crossattn_cache.
                xkv_refs = (
                    crossattn_cache[0]["k"],
                    crossattn_cache[0]["v"],
                    crossattn_cache[-1]["k"],
                    crossattn_cache[-1]["v"],
                )
                xkv_vers = tuple(None if t.is_inference() else t._version for t in xkv_refs)
                xkv_same = (
                    not self._static_crossattn_dirty
                    and self._last_xkv_refs is not None
                    and self._last_xkv_versions == xkv_vers
                    and all(a is b for a, b in zip(self._last_xkv_refs, xkv_refs, strict=True))
                )
                if (
                    self._static_crossattn_cache is None
                    or len(self._static_crossattn_cache) != len(crossattn_cache)
                    or self._static_crossattn_cache[0]["k"].shape != crossattn_cache[0]["k"].shape
                    or self._static_crossattn_cache[0]["k"].dtype != crossattn_cache[0]["k"].dtype
                    or self._static_crossattn_cache[0]["k"].device != crossattn_cache[0]["k"].device
                ):
                    with torch.inference_mode(False):
                        static_ca = []
                        for slot in crossattn_cache:
                            kb = torch.empty(slot["k"].shape, dtype=slot["k"].dtype, device=slot["k"].device)
                            vb = torch.empty(slot["v"].shape, dtype=slot["v"].dtype, device=slot["v"].device)
                            kb.copy_(slot["k"])
                            vb.copy_(slot["v"])
                            static_ca.append({"k": kb, "v": vb, "is_init": True})
                    self._static_crossattn_cache = static_ca
                    self._last_xkv_refs = xkv_refs
                    self._last_xkv_versions = xkv_vers
                    self._static_crossattn_dirty = False
                elif not xkv_same:
                    for dst, src in zip(self._static_crossattn_cache, crossattn_cache, strict=False):
                        dst["k"].copy_(src["k"])
                        dst["v"].copy_(src["v"])
                    self._last_xkv_refs = xkv_refs
                    self._last_xkv_versions = xkv_vers
                    self._static_crossattn_dirty = False
                graph_crossattn_cache = self._static_crossattn_cache

                resolved_attn = self._resolved_attention_backend(noisy_image_or_video.device)
                exact_norms = self._exact_norms
                runner_key = (
                    tuple((slot["k"].data_ptr(), slot["v"].data_ptr()) for slot in kv_cache),
                    tuple((slot["k"].data_ptr(), slot["v"].data_ptr()) for slot in graph_crossattn_cache),
                    None
                    if not kv_cache_mouse
                    else tuple((slot["k"].data_ptr(), slot["v"].data_ptr()) for slot in kv_cache_mouse),
                    None
                    if not kv_cache_keyboard
                    else tuple((slot["k"].data_ptr(), slot["v"].data_ptr()) for slot in kv_cache_keyboard),
                    tuple(noisy_image_or_video.shape),
                    has_mouse,
                    virtual_current_start,
                    resolved_attn,
                    exact_norms,
                )
                runner = self._block_graph_runners.get(runner_key)
                if runner is not None:
                    self._block_graph_runners.move_to_end(runner_key)
                else:
                    from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

                    from .fast_kernels import mg2_fast_steady_forward

                    model = self.model
                    device = noisy_image_or_video.device
                    sinusoid_freqs = self._get_sinusoid_freqs(device)
                    if model.freqs.device != device:
                        model.freqs = model.freqs.to(device)
                    bound_attn = resolved_attn
                    bound_exact = exact_norms
                    bound_vstart = virtual_current_start
                    bound_vend = virtual_end_tokens
                    bound_astart = virtual_start_frame
                    bound_aend = local_attn_size

                    def _steady_block_forward(x_in, t_in, cc_in, kb_in, m_in):
                        return mg2_fast_steady_forward(
                            model,
                            x_in,
                            t_in,
                            cc_in,
                            kb_in,
                            m_in if has_mouse else None,
                            sinusoid_freqs=sinusoid_freqs,
                            kv_cache=kv_cache,
                            kv_cache_mouse=kv_cache_mouse,
                            kv_cache_keyboard=kv_cache_keyboard,
                            crossattn_cache=graph_crossattn_cache,
                            local_start_index=bound_vstart,
                            local_end_index=bound_vend,
                            action_local_start=bound_astart,
                            action_local_end=bound_aend,
                            start_frame=bound_astart,
                            attention_backend=bound_attn,
                            exact_norms=bound_exact,
                        )

                    runner = InferenceCUDAGraphRunner(
                        _steady_block_forward,
                        warmup=self._block_graph_warmup,
                        max_graphs=4,
                        clone_outputs=True,
                        extra_key=f"mg2_steady_block:{hash(runner_key)}",
                    )
                    self._block_graph_runners[runner_key] = runner
                    while len(self._block_graph_runners) > self._block_graph_max_runners:
                        _, evicted_runner = self._block_graph_runners.popitem(last=False)
                        evicted_runner.invalidate()

                flow_pred = runner(
                    self._canonical_contiguous(noisy_image_or_video.to(model_dtype)),
                    self._canonical_contiguous(input_timestep.to(dtype=torch.float32)),
                    cond_concat,
                    kb_win,
                    mouse_win,
                )
                self._set_cache_positions(
                    kv_cache,
                    kv_cache_mouse,
                    kv_cache_keyboard,
                    main_global=actual_end_tokens,
                    main_local=kv_cache_size,
                    action_global=actual_end_frame,
                    action_local=local_attn_size,
                    sync_gpu_tensors=False,
                )
            else:
                if crossattn_cache is not None and not crossattn_cache[0].get("is_init", False):
                    self._static_crossattn_dirty = True
                if self._master_t_freqs is not None:
                    self._restore_master_rope_tables()
                # Fast fused path for warmup chunks (e.g., Chunk 0) when CUDA acceleration is active,
                # eliminating the 50 ms FA3-on-tiny-action-sequence bottleneck on initial chunks too.
                use_fast_warmup = (
                    self._block_graph_enabled
                    and noisy_image_or_video.is_cuda
                    and not self.model.training
                    and not torch.is_grad_enabled()
                    and input_timestep.ndim == 2
                    and current_start is not None
                    and crossattn_cache is not None
                    and int(getattr(self.model, "sink_size", 0)) == 0
                    and "keyboard_cond" in conditional_dict
                    and "cond_concat" in conditional_dict
                    and "visual_context" in conditional_dict
                )
                if use_fast_warmup:
                    ph, pw = self.model.patch_size[1], self.model.patch_size[2]
                    frame_seqlen = (noisy_image_or_video.shape[-2] // ph) * (noisy_image_or_video.shape[-1] // pw)
                    block_frames = int(noisy_image_or_video.shape[2])
                    block_tokens = block_frames * frame_seqlen
                    start_frame = int(current_start) // frame_seqlen
                    kv_cache_size = int(kv_cache[0]["k"].shape[1])
                    if int(current_start) + block_tokens <= kv_cache_size:
                        from .fast_kernels import mg2_fast_steady_forward

                        self._ensure_crossattn_cache_initialized(conditional_dict["visual_context"], crossattn_cache)
                        device = noisy_image_or_video.device
                        if self.model.freqs.device != device:
                            self.model.freqs = self.model.freqs.to(device)
                        has_mouse = "mouse_cond" in conditional_dict and conditional_dict["mouse_cond"] is not None
                        flow_pred = mg2_fast_steady_forward(
                            self.model,
                            noisy_image_or_video.to(model_dtype),
                            input_timestep.to(dtype=torch.float32),
                            conditional_dict["cond_concat"],
                            conditional_dict["keyboard_cond"],
                            conditional_dict["mouse_cond"] if has_mouse else None,
                            sinusoid_freqs=self._get_sinusoid_freqs(device),
                            kv_cache=kv_cache,
                            kv_cache_mouse=kv_cache_mouse,
                            kv_cache_keyboard=kv_cache_keyboard,
                            crossattn_cache=crossattn_cache,
                            local_start_index=int(current_start),
                            local_end_index=int(current_start) + block_tokens,
                            action_local_start=start_frame,
                            action_local_end=start_frame + block_frames,
                            start_frame=start_frame,
                            attention_backend=self._resolved_attention_backend(device),
                            exact_norms=self._exact_norms,
                        )
                        self._set_cache_positions(
                            kv_cache,
                            kv_cache_mouse,
                            kv_cache_keyboard,
                            main_global=int(current_start) + block_tokens,
                            main_local=int(current_start) + block_tokens,
                            action_global=start_frame + block_frames,
                            action_local=start_frame + block_frames,
                            sync_gpu_tensors=False,
                        )
                    else:
                        use_fast_warmup = False
                if not use_fast_warmup:
                    flow_pred = self.model(
                        noisy_image_or_video.to(model_dtype),  # .permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        **conditional_dict,
                        # seq_len=self.seq_len,
                        kv_cache=kv_cache,
                        kv_cache_mouse=kv_cache_mouse,
                        kv_cache_keyboard=kv_cache_keyboard,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                        cache_start=cache_start,
                    )  # .permute(0, 2, 1, 3, 4)

        else:
            flow_pred = self.model(
                noisy_image_or_video.to(model_dtype),  # .permute(0, 2, 1, 3, 4),
                t=input_timestep,
                **conditional_dict,
            )
            # .permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=rearrange(flow_pred, "b c f h w -> (b f) c h w"),  # .flatten(0, 1),
            xt=rearrange(noisy_image_or_video, "b c f h w -> (b f) c h w"),  # .flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        )  # .unflatten(0, flow_pred.shape[:2])
        pred_x0 = rearrange(pred_x0, "(b f) c h w -> b c f h w", b=flow_pred.shape[0])
        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
