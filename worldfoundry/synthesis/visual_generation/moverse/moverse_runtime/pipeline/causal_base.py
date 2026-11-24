"""Shared cache and denoising utilities for MoVerse causal inference."""

import torch


class CausalPipelineBase:
    """Mixin providing shared cache utilities for causal inference.

    Subclasses must set before calling any base method:
        self.num_transformer_blocks (int)  — 30 for Wan
        self.frame_seq_length       (int)  — 1560 base, 3120 with height-concat
        self.num_frame_per_block    (int)
        self.independent_first_frame (bool)

    Methods are grouped into four sections:
        1. Attention patching      — _patch_attn_max_size
        2. Frame-block utilities   — _compute_num_blocks, _build_all_num_frames
        3. KV cache creation       — _compute_kv_cache_size, _make_kv_cache,
                                     _make_crossattn_cache
        4. KV cache reset          — _reset_kv_cache, _reset_crossattn_cache
        5. Conditioning slice      — _slice_cond
    """

    # ------------------------------------------------------------------
    # 1. Attention patching
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_attn_max_size(
        generator,
        frame_seq_length: int,
        num_max_frames: int = 21,
    ) -> None:
        """Patch max_attention_size on every CausalWanSelfAttention block.

        With height-concat conditioning (use_cond_latent=True) each frame
        produces 2× tokens (3120 instead of the default 1560 for the 60×52
        patch grid).  The hardcoded window in CausalWanSelfAttention is
        based on 1560 tokens/frame; without this patch early frames cannot
        attend to all their own keys and produce corrupted output.

        For global attention (local_attn_size == -1) the window is set to
        ``num_max_frames * frame_seq_length``.
        For local attention the window is ``local_attn_size * frame_seq_length``.

        Args:
            generator:       WanDiffusionWrapper whose model.blocks are patched.
            frame_seq_length: Actual tokens per frame (1560 or 3120).
            num_max_frames:  Maximum frames covered by the global window.
        """
        for block in generator.model.blocks:
            # CausalMMDiTDoubleBlock stores local_attn_size on the block itself
            # (not on block.self_attn which is a _SelfAttnProj without max_attention_size).
            # Skip those blocks — MMDiT inference does not use height-concat so no
            # patching is needed for them.
            if not hasattr(block.self_attn, "local_attn_size"):
                continue
            if block.self_attn.local_attn_size == -1:
                block.self_attn.max_attention_size = num_max_frames * frame_seq_length
            else:
                block.self_attn.max_attention_size = block.self_attn.local_attn_size * frame_seq_length

    # ------------------------------------------------------------------
    # 2. Frame-block utilities
    # ------------------------------------------------------------------

    def _compute_num_blocks(
        self,
        num_frames: int,
        initial_latent: torch.Tensor | None,
    ) -> tuple[int, int, int]:
        """Validate frame-count divisibility and return block/frame counts.

        For a standard t2v or i2v model (independent_first_frame=False, or
        True but initial_latent provided) all ``num_frames`` noise frames
        must be divisible by ``num_frame_per_block``.

        For an independent-first-frame t2v model running *without* image
        conditioning (independent_first_frame=True, initial_latent=None) the
        first frame is generated as a singleton block, so (num_frames - 1)
        must be divisible.

        Args:
            num_frames:     T dimension of the noise tensor.
            initial_latent: Optional context latent [B, T_ctx, C, H, W].

        Returns:
            num_blocks:        Number of AR generation blocks (excluding the
                               initial-latent context blocks).
            num_input_frames:  Frames in initial_latent (0 if None).
            num_output_frames: num_frames + num_input_frames.
        """
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        return num_blocks, num_input_frames, num_output_frames

    def _build_all_num_frames(
        self,
        num_blocks: int,
        initial_latent: torch.Tensor | None,
    ) -> list[int]:
        """Return the per-block frame-count list for the AR generation loop.

        For an independent-first-frame t2v model running without image
        conditioning, prepends a singleton block: [1, B, B, B, ...].
        Otherwise returns [B, B, B, ...] where B = num_frame_per_block.

        Args:
            num_blocks:     Number of AR generation blocks.
            initial_latent: Provided image/video conditioning, or None.

        Returns:
            List of integers, one entry per AR block.
        """
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        return all_num_frames

    # ------------------------------------------------------------------
    # 3. KV cache creation
    # ------------------------------------------------------------------

    def _compute_kv_cache_size(
        self,
        local_attn_size: int,
        num_output_frames: int,
    ) -> int:
        """Return the KV cache sequence-length dimension.

        For local/sliding-window attention (local_attn_size != -1) the cache
        only needs to hold ``local_attn_size`` frames worth of tokens.
        For global attention the cache holds all ``num_output_frames`` frames.
        With height-concat conditioning ``frame_seq_length`` is already doubled,
        so the capacity doubles automatically.

        Args:
            local_attn_size:   Frames in the local attention window (-1 = global).
            num_output_frames: Total frames to generate (used for global cache).

        Returns:
            Number of token slots in the KV cache (sequence-length dim).
        """
        if local_attn_size != -1:
            return local_attn_size * self.frame_seq_length
        return num_output_frames * self.frame_seq_length

    def _make_kv_cache(
        self,
        batch_size: int,
        kv_cache_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[dict]:
        """Allocate a single KV cache list with ``num_transformer_blocks`` entries.

        Each entry holds::

            {
                "k":                Tensor [batch_size, kv_cache_size, 12, 128],
                "v":                Tensor [batch_size, kv_cache_size, 12, 128],
                "global_end_index": LongTensor [1],  # total tokens written
                "local_end_index":  LongTensor [1],  # local window write pos
            }

        The 12 attention heads and 128 head-dim are Wan model constants.

        Args:
            batch_size:    Inference batch size.
            kv_cache_size: Sequence-length dimension of k/v tensors.
            dtype:         Floating-point dtype (typically bfloat16).
            device:        Target device.

        Returns:
            List of ``num_transformer_blocks`` cache dicts.
        """
        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        return kv_cache

    def _make_memrope_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        memrope_cfg: dict,
    ) -> list[dict]:
        """Allocate the MemRoPE Phase 2 split gen/cond three-tier KV cache.

        Each of ``num_transformer_blocks`` entries holds two half-caches ("gen", "cond"),
        each laid out as separate tier buffers (raw, un-roped keys; raw values)::

            {
                "sink_k/v":  [B, sink*fs, 12, 128]   # pinned earliest frames
                "local_k/v": [B, local*fs, 12, 128]  # rolling recent window
                "muL_k/v":   [B, fs, 12, 128]        # long-term EMA (if long_mem>0)
                "muS_k/v":   [B, fs, 12, 128]        # short-term EMA (if short_mem>0)
                "sink_count","local_count": int      # frames currently filled
                "muL_init","muS_init": bool          # EMA seeded yet
            }

        plus a shared ``"committed_frame"`` (last frame index pushed; -1 = empty). ``fs`` is
        the per-half token count (frame_seq_length // 2, e.g. 1560). See SUMMARY_MEMROPE.md.
        """

        def _half(cfg: dict, fs: int) -> dict:
            """One half/tier-set with ``fs`` tokens per frame (fs = full 3120 in fused mode,
            1560 per half in split mode)."""
            S, L = cfg["sink"], cfg["local"]
            hb = {
                "sink_k": torch.zeros([batch_size, S * fs, 12, 128], dtype=dtype, device=device),
                "sink_v": torch.zeros([batch_size, S * fs, 12, 128], dtype=dtype, device=device),
                "local_k": torch.zeros([batch_size, L * fs, 12, 128], dtype=dtype, device=device),
                "local_v": torch.zeros([batch_size, L * fs, 12, 128], dtype=dtype, device=device),
                "sink_count": 0,
                "local_count": 0,
                "muL_init": False,
                "muS_init": False,
            }
            if cfg["long_mem"] > 0:
                hb["muL_k"] = torch.zeros([batch_size, fs, 12, 128], dtype=dtype, device=device)
                hb["muL_v"] = torch.zeros([batch_size, fs, 12, 128], dtype=dtype, device=device)
            if cfg["short_mem"] > 0:
                hb["muS_k"] = torch.zeros([batch_size, fs, 12, 128], dtype=dtype, device=device)
                hb["muS_v"] = torch.zeros([batch_size, fs, 12, 128], dtype=dtype, device=device)
            return hb

        cache = []
        if memrope_cfg.get("fused", False):
            # FUSED: single [gen|cond] full-frame tier-set + preallocated roped-key / value
            # buffers (sized to the max assembled length = sink + 2 memory + local frames).
            ffs = self.frame_seq_length  # full frame (gen+cond), e.g. 3120
            g = memrope_cfg["gen"]
            max_frames = g["sink"] + g["long_mem"] + g["short_mem"] + g["local"]
            for _ in range(self.num_transformer_blocks):
                entry = _half(g, ffs)
                entry["committed_frame"] = -1
                entry["roped_keys_buf"] = torch.zeros(
                    [batch_size, max_frames * ffs, 12, 128], dtype=dtype, device=device
                )
                entry["vals_buf"] = torch.zeros([batch_size, max_frames * ffs, 12, 128], dtype=dtype, device=device)
                cache.append(entry)
        else:
            # SPLIT: independent gen/cond half-caches (1560 tokens per half-frame).
            fs = self.frame_seq_length // 2
            for _ in range(self.num_transformer_blocks):
                cache.append(
                    {
                        "gen": _half(memrope_cfg["gen"], fs),
                        "cond": _half(memrope_cfg["cond"], fs),
                        "committed_frame": -1,
                    }
                )
        return cache

    @staticmethod
    def _reset_memrope_kv_cache(cache: list[dict]) -> None:
        """Reset counters/EMA-init flags for reuse between inference runs (tensors kept).

        Handles both the fused (flat tier-set) and split (gen/cond sub-dicts) layouts.
        """

        def _reset_half(hb: dict) -> None:
            hb["sink_count"] = 0
            hb["local_count"] = 0
            hb["muL_init"] = False
            hb["muS_init"] = False

        for entry in cache:
            entry["committed_frame"] = -1
            if "gen" in entry:  # split layout
                _reset_half(entry["gen"])
                _reset_half(entry["cond"])
            else:  # fused layout (flat)
                _reset_half(entry)

    def _make_crossattn_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[dict]:
        """Allocate a single cross-attention cache list.

        Each of the ``num_transformer_blocks`` entries holds::

            {
                "k":       Tensor [batch_size, 512, 12, 128],
                "v":       Tensor [batch_size, 512, 12, 128],
                "is_init": False,   # set True after first forward pass populates k/v
            }

        The 512 dim is the fixed text-encoder sequence length for Wan (UMT5-XXL
        outputs at most 512 tokens).

        Args:
            batch_size: Inference batch size.
            dtype:      Floating-point dtype.
            device:     Target device.

        Returns:
            List of ``num_transformer_blocks`` cache dicts.
        """
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        return crossattn_cache

    # ------------------------------------------------------------------
    # 4. KV cache reset (for lazy re-use between inference calls)
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_kv_cache(
        kv_cache: list[dict],
        device: torch.device,
    ) -> None:
        """Reset global_end_index and local_end_index to 0 for every block.

        Called when an already-allocated cache is reused for a new inference
        run to avoid re-allocating large tensors.  Modifies the list in place.

        Args:
            kv_cache: List of per-block KV cache dicts.
            device:   Device for the replacement index tensors.
        """
        for block in kv_cache:
            block["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
            block["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)

    @staticmethod
    def _reset_crossattn_cache(crossattn_cache: list[dict]) -> None:
        """Reset is_init flags to False for every block.

        Forces cross-attention keys/values to be recomputed from the new text
        conditioning on the next forward pass.  Modifies the list in place.

        Args:
            crossattn_cache: List of per-block cross-attn cache dicts.
        """
        for block in crossattn_cache:
            block["is_init"] = False

    # ------------------------------------------------------------------
    # 5. Flow-matching Euler step helper
    # ------------------------------------------------------------------

    def _flow_step(
        self,
        flow_pred_flat: torch.Tensor,
        noisy_flat: torch.Tensor,
        t_flat: torch.Tensor,
        next_t_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic Euler ODE step for flow-matching denoising.

        Computes x_{t_next} = x_t + flow * (sigma_next - sigma_t) where sigmas
        are resolved from the scheduler's pre-built sigma table via nearest-
        neighbour lookup on the timestep axis.

        This is the deterministic alternative to the stochastic consistency step
        ``add_noise(pred_x0, noise, next_t)`` used in the default pipeline.  The
        two approaches differ as follows:

        * Consistency (default): stochastic — injects fresh noise at every step
          and re-anchors to the x0 prediction.
        * Flow-matching Euler (this method): deterministic — integrates the ODE
          trajectory directly using the predicted velocity (flow = noise - x0).

        Requires the shared Wan inference scheduler with
        ``sigmas`` and ``timesteps`` attributes populated by ``set_timesteps``.

        Args:
            flow_pred_flat: Flow velocity prediction [B*F, C, H, W] (= noise - x0).
            noisy_flat:     Current noisy latent x_t [B*F, C, H, W].
            t_flat:         Current timestep values [B*F], integer-valued.
            next_t_flat:    Target timestep values [B*F], integer-valued.

        Returns:
            x_{t_next} [B*F, C, H, W] with the same dtype as ``noisy_flat``.
        """
        return self.scheduler.flow_step(
            flow_pred_flat,
            noisy_flat,
            t_flat,
            next_t_flat,
        )

    # ------------------------------------------------------------------
    # 5b. Dual KV cache (MMDiT: separate x and c sub-buffers)
    # ------------------------------------------------------------------

    def _make_dual_kv_cache(
        self,
        batch_size: int,
        kv_cache_size: int,
        dtype: torch.dtype,
        device: torch.device,
        double_stream_layers,
    ) -> list[dict]:
        """Allocate a KV cache list with dual x/c sub-buffers at double-stream blocks.

        At regular (single-stream) blocks the cache dict has the standard layout::

            {"k": ..., "v": ..., "global_end_index": ..., "local_end_index": ...}

        At double-stream blocks the cache dict has::

            {
                "x_k": Tensor [batch_size, kv_cache_size, 12, 128],
                "x_v": Tensor [batch_size, kv_cache_size, 12, 128],
                "c_k": Tensor [batch_size, kv_cache_size, 12, 128],
                "c_v": Tensor [batch_size, kv_cache_size, 12, 128],
                "global_end_index": LongTensor [1],
                "local_end_index":  LongTensor [1],
            }

        x and c share the same index pointers because they advance in lockstep.

        Args:
            batch_size:          Inference batch size.
            kv_cache_size:       Sequence-length dimension of k/v tensors.
            dtype:               Floating-point dtype.
            device:              Target device.
            double_stream_layers: Iterable of block indices that are double-stream.

        Returns:
            List of ``num_transformer_blocks`` cache dicts.
        """
        double_set = set(double_stream_layers)
        kv_cache = []
        for i in range(self.num_transformer_blocks):
            if i in double_set:
                kv_cache.append(
                    {
                        "x_k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                        "x_v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                        "c_k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                        "c_v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    }
                )
            else:
                kv_cache.append(
                    {
                        "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                        "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    }
                )
        return kv_cache

    @staticmethod
    def _reset_dual_kv_cache(
        kv_cache: list[dict],
        device: torch.device,
    ) -> None:
        """Reset index pointers to 0 for all blocks in a dual KV cache.

        Handles both single-stream dicts (key 'k') and dual-stream dicts (key 'x_k')
        in the same list.  Modifies the list in place.

        Args:
            kv_cache: List of per-block KV cache dicts (mixed single/dual).
            device:   Device for the replacement index tensors.
        """
        for block in kv_cache:
            block["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
            block["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 6. KV cache x-token broadcast helper (skip_clean_rerun_copy_x_last_c_full)
    # ------------------------------------------------------------------

    def _copy_x_last_to_all_layers(self, current_num_frames: int) -> None:
        """After the exit denoising step: broadcast the last single-stream block's
        x-token KV entries to every layer in the KV cache, leaving c-token slots
        (height-concat mode) and c-stream sub-buffers (MMDiT mode) untouched.

        This is the ``skip_clean_rerun_copy_x_last_c_full`` alternative to the
        clean re-run forward pass.  Instead of running the generator at t=0 with
        the denoised prediction (which overwrites KV slots with clean features),
        we copy the already-written x-token entries from the last single-stream
        transformer block across all other layers.

        Token layout per frame
        ----------------------
        Height-concat (is_mmdit=False, use_cond_latent=True):
            frame_seq_length = 3120
            positions [0 : 1560] = x-tokens (noisy half)
            positions [1560 : 3120] = c-tokens (condition half) — left unchanged
            x_len = 1560  (= frame_seq_length // 2)

        MMDiT (is_mmdit=True):
            frame_seq_length = 1560
            All 1560 positions per frame = x-tokens
            double-stream blocks keep their own c_k/c_v — left unchanged
            x_len = 1560  (= frame_seq_length)

        Both modes: x_len == 1560.

        Reference block selection
        -------------------------
        The reference is the *last* entry in ``kv_cache1`` whose dict contains
        the key ``"k"`` (i.e., the last single-stream transformer block).

        For height-concat all 30 blocks are single-stream; for MMDiT some are
        double-stream (``"x_k"`` key).  The last single-stream block's features
        after the denoising exit step serve as the proxy for the clean context
        that would otherwise come from the clean re-run pass.

        Write targets
        -------------
        Single-stream (``"k"`` key):
            Overwrite k[:, x_start : x_start+x_len] and v[...] for each frame.
        Double-stream (``"x_k"`` key, MMDiT only):
            Overwrite x_k[:, x_start : x_start+x_len] and x_v[...].
            c_k and c_v are left intact.

        Args:
            current_num_frames: number of frames in the block just denoised.
        """
        x_len = 1560  # x-tokens per frame (Wan constant)

        # Find the last single-stream block as the reference
        ref_idx: int | None = None
        for i in range(len(self.kv_cache1) - 1, -1, -1):
            if "k" in self.kv_cache1[i]:
                ref_idx = i
                break
        if ref_idx is None:
            return  # All double-stream — nothing to copy (shouldn't happen)

        ref_cache = self.kv_cache1[ref_idx]
        local_end: int = ref_cache["local_end_index"].item()
        block_tokens: int = current_num_frames * self.frame_seq_length
        local_start: int = local_end - block_tokens
        # local_start is non-negative for global attention. Local-attention callers
        # must keep the requested block inside the active cache window.

        # Clone reference x-token slices (one per frame in the block)
        with torch.no_grad():
            ref_k_list: list[torch.Tensor] = []
            ref_v_list: list[torch.Tensor] = []
            for i in range(current_num_frames):
                xs = local_start + i * self.frame_seq_length
                ref_k_list.append(ref_cache["k"][:, xs : xs + x_len].clone())
                ref_v_list.append(ref_cache["v"][:, xs : xs + x_len].clone())

            # Broadcast to all layers
            for cache in self.kv_cache1:
                for i in range(current_num_frames):
                    xs = local_start + i * self.frame_seq_length
                    if "k" in cache:
                        # Single-stream (height-concat or MMDiT single-stream)
                        cache["k"][:, xs : xs + x_len].copy_(ref_k_list[i])
                        cache["v"][:, xs : xs + x_len].copy_(ref_v_list[i])
                    elif "x_k" in cache:
                        # MMDiT double-stream: x sub-buffers only
                        cache["x_k"][:, xs : xs + x_len].copy_(ref_k_list[i])
                        cache["x_v"][:, xs : xs + x_len].copy_(ref_v_list[i])
                        # c_k / c_v intentionally left unchanged

    def _copy_all_last_to_all_layers(self, current_num_frames: int) -> None:
        """After the exit denoising step: broadcast the last layer's KV entries to
        every layer in the KV cache, including c-token slots.

        This is the ``skip_clean_rerun_copy_x_last_c_last`` variant.

        Height-concat (is_mmdit=False, use_cond_latent=True):
            frame_seq_length = 3120.  ALL 3120 tokens per frame (x + c) are copied
            from the last single-stream block to every single-stream block.

        MMDiT (is_mmdit=True):
            x / single-stream part — identical to ``_copy_x_last_to_all_layers``:
                last single-stream block → all single-stream k/v + all double-stream x_k/x_v.
            c / double-stream part — new:
                last double-stream block's c_k/c_v → every double-stream block's c_k/c_v.

        Args:
            current_num_frames: number of frames in the block just denoised.
        """
        frame_len = self.frame_seq_length  # 3120 height-concat, 1560 MMDiT

        # ── Reference: last single-stream block ──────────────────────────────
        ref_single_idx: int | None = None
        for i in range(len(self.kv_cache1) - 1, -1, -1):
            if "k" in self.kv_cache1[i]:
                ref_single_idx = i
                break
        if ref_single_idx is None:
            return

        ref_single = self.kv_cache1[ref_single_idx]
        local_end: int = ref_single["local_end_index"].item()
        block_tokens: int = current_num_frames * frame_len
        local_start: int = local_end - block_tokens

        with torch.no_grad():
            # Clone full-frame token slices from last single-stream block
            ref_k_list: list[torch.Tensor] = []
            ref_v_list: list[torch.Tensor] = []
            for i in range(current_num_frames):
                xs = local_start + i * frame_len
                ref_k_list.append(ref_single["k"][:, xs : xs + frame_len].clone())
                ref_v_list.append(ref_single["v"][:, xs : xs + frame_len].clone())

            # ── Reference: last double-stream block (MMDiT c-stream) ─────────
            ref_ck_list: list[torch.Tensor] | None = None
            ref_cv_list: list[torch.Tensor] | None = None
            for i in range(len(self.kv_cache1) - 1, -1, -1):
                if "x_k" in self.kv_cache1[i]:
                    ref_double = self.kv_cache1[i]
                    ref_ck_list = []
                    ref_cv_list = []
                    for j in range(current_num_frames):
                        xs = local_start + j * frame_len
                        ref_ck_list.append(ref_double["c_k"][:, xs : xs + frame_len].clone())
                        ref_cv_list.append(ref_double["c_v"][:, xs : xs + frame_len].clone())
                    break

            # ── Broadcast ────────────────────────────────────────────────────
            for cache in self.kv_cache1:
                for i in range(current_num_frames):
                    xs = local_start + i * frame_len
                    if "k" in cache:
                        # Single-stream: copy full frame (x+c for height-concat, x for MMDiT)
                        cache["k"][:, xs : xs + frame_len].copy_(ref_k_list[i])
                        cache["v"][:, xs : xs + frame_len].copy_(ref_v_list[i])
                    elif "x_k" in cache:
                        # MMDiT double-stream: update x_k/x_v from single-stream ref
                        cache["x_k"][:, xs : xs + frame_len].copy_(ref_k_list[i])
                        cache["x_v"][:, xs : xs + frame_len].copy_(ref_v_list[i])
                        # Update c_k/c_v from double-stream ref
                        if ref_ck_list is not None:
                            cache["c_k"][:, xs : xs + frame_len].copy_(ref_ck_list[i])
                            cache["c_v"][:, xs : xs + frame_len].copy_(ref_cv_list[i])

    # ------------------------------------------------------------------
    # 7. Conditioning slice helper
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_cond(
        cond_latent: torch.Tensor | None,
        start: int,
        length: int,
    ) -> torch.Tensor | None:
        """Slice cond_latent along the temporal dimension, or return None.

        Args:
            cond_latent: Full conditioning tensor [B, T, C, H, W], or None.
            start:       Index of the first frame to extract.
            length:      Number of frames to extract.

        Returns:
            ``cond_latent[:, start : start + length]`` when cond_latent is not
            None, otherwise None.
        """
        if cond_latent is None:
            return None
        return cond_latent[:, start : start + length]
