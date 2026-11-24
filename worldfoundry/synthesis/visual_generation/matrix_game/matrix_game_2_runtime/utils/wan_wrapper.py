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
            if self._master_t_freqs is None or self._master_t_freqs.device != device:
                self._master_t_freqs = self.model.freqs[:, :t_dim].clone()
        blocks = getattr(self.model, "blocks", ())
        for block in blocks:
            am = getattr(block, "action_model", None)
            if am is None or not hasattr(am, "freqs_cos"):
                continue
            if self._master_action_freqs is None or self._master_action_freqs[0].device != device:
                cos_gpu = am.freqs_cos.to(device=device).contiguous()
                sin_gpu = am.freqs_sin.to(device=device).contiguous()
                self._master_action_freqs = (cos_gpu.clone(), sin_gpu.clone())
                self._shared_action_freqs = (cos_gpu, sin_gpu)
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
                    m_scratch = (
                        torch.empty(
                            (m0.shape[0], action_rolled, m0.shape[2], m0.shape[3]), dtype=m0.dtype, device=m0.device
                        ),
                        torch.empty(
                            (m0.shape[0], action_rolled, m0.shape[2], m0.shape[3]), dtype=m0.dtype, device=m0.device
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
            if "_host_global_end_index" in slot:
                slot["_host_global_end_index"] = main_global
                slot["_host_local_end_index"] = main_local
            if sync_gpu_tensors:
                slot["global_end_index"].fill_(main_global)
                slot["local_end_index"].fill_(main_local)
        if kv_cache_mouse:
            for slot in kv_cache_mouse:
                if "_host_global_end_index" in slot:
                    slot["_host_global_end_index"] = action_global
                    slot["_host_local_end_index"] = action_local
                if sync_gpu_tensors:
                    slot["global_end_index"].fill_(action_global)
                    slot["local_end_index"].fill_(action_local)
        if kv_cache_keyboard:
            for slot in kv_cache_keyboard:
                if "_host_global_end_index" in slot:
                    slot["_host_global_end_index"] = action_global
                    slot["_host_local_end_index"] = action_local
                if sync_gpu_tensors:
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
                cached_local = int(kv_cache[0].get("_host_local_end_index", 0))
                if (
                    local_attn_size > block_frames
                    and virtual_start_frame_cand >= action_windows_size
                    and kv_cache_size == local_attn_size * frame_seqlen
                    and start_frame >= virtual_start_frame_cand
                    and cached_local == kv_cache_size
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
                # ActionModule at virtual_start_frame. Clone into canonical contiguous layout so
                # batch-dim=1 parent strides do not vary across blocks and trigger recaptures.
                win_start = 4 * (start_frame - virtual_start_frame)
                win_end = 4 * start_frame + 1 + 4 * (block_frames - 1)
                kb_win = conditional_dict["keyboard_cond"][:, win_start:win_end].clone(
                    memory_format=torch.contiguous_format
                )
                has_mouse = "mouse_cond" in conditional_dict and conditional_dict["mouse_cond"] is not None
                mouse_win = (
                    conditional_dict["mouse_cond"][:, win_start:win_end].clone(memory_format=torch.contiguous_format)
                    if has_mouse
                    else kb_win.new_empty(0)
                )
                cond_concat = conditional_dict["cond_concat"].clone(memory_format=torch.contiguous_format)
                visual_context = conditional_dict["visual_context"].clone(memory_format=torch.contiguous_format)

                # Copy cross-attention K/V into persistent static buffers so graph pointers
                # remain stable even when a new session re-initializes crossattn_cache.
                # Check tensor object identity AND mutation version (both first and last layer)
                # plus explicit dirty flag so in-place reuse or Python id() recycling can never
                # serve stale cross-attention features across sessions.
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
                    self._static_crossattn_cache = [
                        {"k": slot["k"].clone(), "v": slot["v"].clone(), "is_init": True} for slot in crossattn_cache
                    ]
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
                )
                runner = self._block_graph_runners.get(runner_key)
                if runner is not None:
                    self._block_graph_runners.move_to_end(runner_key)
                else:
                    from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

                    model = self.model
                    device = noisy_image_or_video.device
                    half_dim = int(model.freq_dim) // 2
                    sinusoid_freqs = torch.pow(
                        10000.0,
                        -torch.arange(half_dim, device=device, dtype=torch.float64).div(half_dim),
                    )
                    if model.freqs.device != device:
                        model.freqs = model.freqs.to(device)
                    static_grid_sizes = torch.tensor(
                        [
                            block_frames,
                            noisy_image_or_video.shape[-2] // ph,
                            noisy_image_or_video.shape[-1] // pw,
                        ],
                        dtype=torch.long,
                    )
                    static_seq_lens = torch.tensor(
                        [block_tokens] * int(noisy_image_or_video.shape[0]), dtype=torch.long
                    )

                    def _steady_block_forward(x_in, t_in, cc_in, vc_in, kb_in, m_in):
                        x = torch.cat([x_in, cc_in], dim=1)
                        x = model.patch_embedding(x)
                        x = x.flatten(2).transpose(1, 2)

                        pos = t_in.flatten().to(dtype=torch.float64)
                        sinusoid = torch.outer(pos, sinusoid_freqs)
                        t_emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).type_as(x)
                        e = model.time_embedding(t_emb)
                        e0 = model.time_projection(e).unflatten(1, (6, model.dim)).unflatten(dim=0, sizes=t_in.shape)
                        context = model.img_emb(vc_in)

                        block_kwargs = {
                            "e": e0,
                            "seq_lens": static_seq_lens,
                            "grid_sizes": static_grid_sizes,
                            "freqs": model.freqs,
                            "context": context,
                            "mouse_cond": m_in if has_mouse else None,
                            "context_lens": None,
                            "keyboard_cond": kb_in,
                            "block_mask": model.block_mask,
                            "block_mask_mouse": model.block_mask_mouse,
                            "block_mask_keyboard": model.block_mask_keyboard,
                            "use_rope_keyboard": model.use_rope_keyboard,
                            "num_frame_per_block": model.num_frame_per_block,
                            "current_start": virtual_current_start,
                            "cache_start": virtual_current_start,
                        }
                        for block_index, block in enumerate(model.blocks):
                            block_kwargs["kv_cache"] = kv_cache[block_index]
                            block_kwargs["kv_cache_mouse"] = (
                                None if kv_cache_mouse is None else kv_cache_mouse[block_index]
                            )
                            block_kwargs["kv_cache_keyboard"] = (
                                None if kv_cache_keyboard is None else kv_cache_keyboard[block_index]
                            )
                            block_kwargs["crossattn_cache"] = graph_crossattn_cache[block_index]
                            x = block(x, **block_kwargs)

                        x = model.head(x, e.unflatten(dim=0, sizes=t_in.shape).unsqueeze(2))
                        return model.unpatchify(x, static_grid_sizes)

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
                    noisy_image_or_video.to(model_dtype).clone(memory_format=torch.contiguous_format),
                    input_timestep.to(dtype=torch.float32).clone(memory_format=torch.contiguous_format),
                    cond_concat,
                    visual_context,
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
