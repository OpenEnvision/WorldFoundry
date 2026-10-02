from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
import torch

from einops import rearrange
from ..utils.wan_wrapper import WanDiffusionWrapper
from tqdm import tqdm


# The Matrix-Game-2 streaming VAE exposes 32 recurrent feature-cache slots.
# Keep this lightweight: importing ``ZERO_VAE_CACHE`` allocates roughly 2 GiB of
# CPU tensors before inference starts, even though inference immediately replaces
# every one of those tensors with ``None``.
VAE_CACHE_SLOTS = 32


@dataclass
class CausalInferenceSession:
    """Mutable state for one resident Matrix-Game-2 rollout.

    The diffusion, action-attention, cross-attention, and VAE caches are owned by
    the session and survive between calls to :meth:`generate_next_block`.
    ``current_start_frame`` is measured in latent frames, not decoded RGB frames.

    ``conditional_dict`` is intentionally retained by reference. Interactive
    actions update its preallocated mouse/keyboard timelines in place, matching
    the original Matrix-Game-2 conditioning algorithm.
    """

    conditional_dict: Dict[str, Any]
    mode: str
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    kv_cache: List[Dict[str, Any]]
    mouse_kv_cache: List[Dict[str, Any]]
    keyboard_kv_cache: List[Dict[str, Any]]
    crossattn_cache: List[Dict[str, Any]]
    vae_cache: List[Optional[torch.Tensor]]
    current_start_frame: int = 0
    initial_latent: Optional[torch.Tensor] = None
    last_model_ms: Optional[float] = None
    last_decode_ms: Optional[float] = None
    invalidated: bool = False


@dataclass(frozen=True)
class CausalInferenceBlock:
    """One increment produced by :meth:`generate_next_block`."""

    latent: torch.Tensor
    video: torch.Tensor
    start_frame: int
    end_frame: int
    model_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    total_ms: Optional[float] = None

def get_current_action(mode="universal"):

    CAM_VALUE = 0.1
    if mode == 'universal':
        print()
        print('-'*30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n (I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "i":  [CAM_VALUE, 0],
            "k":  [-CAM_VALUE, 0],
            "j":  [0, -CAM_VALUE],
            "l":  [0, CAM_VALUE],
            "u":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
            "q": [0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
                idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()
                if idx_mouse in CAMERA_VALUE_MAP.keys() and idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    elif mode == 'gta_drive':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "a":  [0, -CAM_VALUE],
            "d":  [0, CAM_VALUE],
            "q":  [0, 0]
        }
        KEYBOARD_IDX = { 
            "w": [1, 0], "s": [0, 1],
            "q": [0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                indexes = input('Please input the actions (split with ` `):\n(e.g. `W` for forward, `W A` for forward and left)\n').strip().lower().split(' ')
                idx_mouse = []
                idx_keyboard = []
                for i in indexes:
                    if i in CAMERA_VALUE_MAP.keys():
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX.keys():
                        idx_keyboard += [i]
                if len(idx_mouse) == 0:
                    idx_mouse += ['q']
                if len(idx_keyboard) == 0:
                    idx_keyboard += ['q']
                assert idx_mouse in [['a'], ['d'], ['q']] and idx_keyboard in [['q'], ['w'], ['s']]
                flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse[0]]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard[0]]).cuda()
    elif mode == 'templerun':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS\n (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")
        print('-'*30)
        KEYBOARD_IDX = { 
            "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
            "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
            "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
            "q": [1, 0, 0, 0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_keyboard = input('Please input the action: \n(e.g. `W` for forward, `Z` for turning left)\n').strip().lower()
                if idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    
    if mode != 'templerun':
        return {
            "mouse": mouse_cond,
            "keyboard": keyboard_cond
        }
    return {
        "keyboard": keyboard_cond
    }

def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode='universal'):
    
    new_cond = {}
    
    cond_concat_start = int(conditional_dict.get("_cond_concat_start_frame", 0))
    local_start_frame = current_start_frame - cond_concat_start
    if local_start_frame < 0:
        raise ValueError(
            "cond_concat starts after the requested global latent frame: "
            f"offset={cond_concat_start}, requested={current_start_frame}"
        )
    cond_concat = conditional_dict["cond_concat"][
        :, :, local_start_frame:local_start_frame + num_frame_per_block
    ]
    if cond_concat.shape[2] != num_frame_per_block:
        raise ValueError(
            "cond_concat does not cover the requested latent block: "
            f"offset={cond_concat_start}, requested={current_start_frame}, "
            f"available={conditional_dict['cond_concat'].shape[2]}"
        )
    new_cond["cond_concat"] = cond_concat
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace is not None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block-1)
        if mode != 'templerun':
            conditional_dict["mouse_cond"][:, -last_frame_num + final_frame: final_frame] = replace['mouse'][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["keyboard_cond"][:, -last_frame_num + final_frame: final_frame] = replace['keyboard'][None, None, :].repeat(1, last_frame_num, 1)
    if mode != 'templerun':
        new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace is not None:
        return new_cond, conditional_dict
    else:
        return new_cond

class CausalInferencePipeline(torch.nn.Module):
    """One resident session with optional decoder/context-refresh overlap.

    The caller serializes start, step, reset and close on a pipeline instance.
    With ``overlap_vae_decode=True``, CUDA decoding uses a pipeline-owned stream
    after denoising, while the caller stream refreshes clean generator caches.
    A device event joins outputs and recurrent caches before caller-side copies
    or the next chunk. Tensor stream records guard allocator lifetime; reset,
    replacement sessions and close drain only the owned decoder stream. CPU
    tensors use the explicitly reported synchronous fallback. The option stays
    disabled by default; stream use alone does not establish a latency benefit.
    """

    def __init__(
            self,
            args,
            device="cuda",
            generator=None,
            vae_decoder=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
            
        self.vae_decoder = vae_decoder
        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self._session: Optional[CausalInferenceSession] = None
        self._last_block: Optional[CausalInferenceBlock] = None
        # Calls on one pipeline are serialized by its resident session owner.
        # The optional stream belongs to this pipeline, never to a global pool.
        self.overlap_vae_decode = False
        self._decode_stream = None
        self._decode_stream_device = None
        self._decode_complete = None
        self._decode_overlap_runtime = {"calls": 0, "execution": "disabled"}

    @property
    def session(self) -> Optional[CausalInferenceSession]:
        """Return the active rollout state, if one has been started."""

        return self._session

    @property
    def last_block(self) -> Optional[CausalInferenceBlock]:
        """Return the most recent block and its model/decode timings."""

        return self._last_block

    def reset_session(self) -> None:
        """Drop the active rollout state without touching resident model weights."""

        self._drain_decode_stream()
        self._session = None
        self._last_block = None
        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.crossattn_cache = None

    def close(self) -> None:
        """Drain the owned decoder stream and release the active session."""
        self.reset_session()

    def _drain_decode_stream(self) -> None:
        stream = self._decode_stream
        try:
            if stream is not None:
                stream.synchronize()
        finally:
            self._decode_stream = None
            self._decode_stream_device = None
            self._decode_complete = None

    def _get_decode_stream(self, device):
        if self._decode_stream is not None and self._decode_stream_device != device:
            self._drain_decode_stream()
        if self._decode_stream is None:
            self._decode_stream = torch.cuda.Stream(device=device)
            self._decode_stream_device = device
        return self._decode_stream

    def _record_decode_overlap(self, execution: str) -> None:
        self._decode_overlap_runtime["execution"] = execution
        applied = getattr(self, "_worldfoundry_applied_optimizations", None)
        if applied is not None and self.overlap_vae_decode:
            applied.effective["overlap_vae_decode"] = execution

    @staticmethod
    def _record_cache_stream(value, stream) -> None:
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value.record_stream(stream)
        elif isinstance(value, Mapping):
            for child in value.values():
                CausalInferencePipeline._record_cache_stream(child, stream)
        elif isinstance(value, (list, tuple)):
            for child in value:
                CausalInferencePipeline._record_cache_stream(child, stream)

    def start_session(
        self,
        conditional_dict: Mapping[str, Any],
        *,
        initial_latent: Optional[torch.Tensor] = None,
        reference_tensor: Optional[torch.Tensor] = None,
        mode: str = "universal",
    ) -> CausalInferenceSession:
        """Allocate a rollout's caches and optionally seed clean latent context.

        With no ``initial_latent``, starting a session performs no model
        generation. If one is supplied, it is consumed once to seed the causal
        attention caches. Matrix-Game-2 image conditioning normally leaves it
        unset and generates its first block from ``cond_concat`` on the first
        call to :meth:`step_session`.
        """

        # An explicit reference wins so legacy inference keeps allocating caches
        # from the noise tensor exactly as before, even when context is supplied.
        reference = reference_tensor
        if reference is None:
            reference = initial_latent
        if reference is None:
            reference = conditional_dict.get("cond_concat")
        if reference is None or reference.ndim < 1:
            raise ValueError(
                "initial_latent, reference_tensor, or cond_concat is required"
            )

        batch_size = reference.shape[0]
        # A replacement session releases recurrent caches only after its owned
        # decoder stream finishes. It does not synchronize unrelated streams.
        self.reset_session()
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=reference.dtype,
            device=reference.device,
        )
        self._initialize_kv_cache_mouse_and_keyboard(
            batch_size=batch_size,
            dtype=reference.dtype,
            device=reference.device,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=reference.dtype,
            device=reference.device,
        )

        session = CausalInferenceSession(
            conditional_dict=(
                conditional_dict
                if isinstance(conditional_dict, dict)
                else dict(conditional_dict)
            ),
            mode=mode,
            batch_size=batch_size,
            dtype=reference.dtype,
            device=reference.device,
            kv_cache=self.kv_cache1,
            mouse_kv_cache=self.kv_cache_mouse,
            keyboard_kv_cache=self.kv_cache_keyboard,
            crossattn_cache=self.crossattn_cache,
            vae_cache=[None] * VAE_CACHE_SLOTS,
            initial_latent=initial_latent,
        )
        self._session = session
        if initial_latent is not None:
            self._seed_session(session, initial_latent)
        return session

    def _seed_session(
        self,
        session: CausalInferenceSession,
        initial_latent: torch.Tensor,
    ) -> None:
        """Populate a fresh session's caches with clean initial context once."""

        if session.current_start_frame != 0:
            raise RuntimeError("a Matrix-Game-2 session can only be seeded once")
        if initial_latent.ndim != 5 or initial_latent.shape[0] != session.batch_size:
            raise ValueError("initial_latent must have shape [B, C, F, H, W]")
        if initial_latent.shape[2] % self.num_frame_per_block != 0:
            raise ValueError(
                "initial_latent frames must be divisible by num_frame_per_block"
            )

        num_input_blocks = initial_latent.shape[2] // self.num_frame_per_block
        for _ in range(num_input_blocks):
            start_frame = session.current_start_frame
            current_ref_latents = initial_latent[
                :, :, start_frame:start_frame + self.num_frame_per_block
            ]
            timestep = torch.zeros(
                [session.batch_size, 1],
                device=session.device,
                dtype=torch.int64,
            )
            self.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=cond_current(
                    session.conditional_dict,
                    start_frame,
                    self.num_frame_per_block,
                    mode=session.mode,
                ),
                timestep=timestep,
                kv_cache=session.kv_cache,
                kv_cache_mouse=session.mouse_kv_cache,
                kv_cache_keyboard=session.keyboard_kv_cache,
                crossattn_cache=session.crossattn_cache,
                current_start=start_frame * self.frame_seq_length,
            )
            session.current_start_frame += self.num_frame_per_block

    def step_session(
        self,
        noise: torch.Tensor,
        conditional_dict: Optional[Mapping[str, Any]] = None,
        *,
        action: Optional[Mapping[str, torch.Tensor]] = None,
        mode: Optional[str] = None,
        profile: bool = False,
    ) -> torch.Tensor:
        """Generate and decode one block on the active resident session."""

        if self._session is None:
            raise RuntimeError("start_session() must be called before step_session()")
        if conditional_dict is not None:
            self._session.conditional_dict = (
                conditional_dict
                if isinstance(conditional_dict, dict)
                else dict(conditional_dict)
            )
        if mode is not None:
            self._session.mode = mode
        block = self.generate_next_block(
            self._session,
            noise,
            action=action,
            profile=profile,
        )
        return block.video

    def generate_next_block(
        self,
        session: CausalInferenceSession,
        noise: torch.Tensor,
        *,
        action: Optional[Mapping[str, torch.Tensor]] = None,
        profile: bool = False,
    ) -> CausalInferenceBlock:
        """Generate one block; the resident owner serializes calls and reset.

        Opt-in decode overlap shares only the read-only denoised latent between
        streams. The caller stream joins decoding before outputs are returned;
        cache reuse and caller-side copies follow ordinary CUDA stream ordering.
        Reset/close or a failed overlapped block drains only the owned stream.
        """

        if session.invalidated:
            raise RuntimeError("session was invalidated by a failed overlapped block; start a new session")

        expected_prefix = (session.batch_size, 16, self.num_frame_per_block)
        if noise.ndim != 5 or tuple(noise.shape[:3]) != expected_prefix:
            raise ValueError(
                "noise must have shape "
                f"[B, 16, {self.num_frame_per_block}, H, W]; "
                f"got {tuple(noise.shape)}"
            )
        if noise.device != session.device or noise.dtype != session.dtype:
            raise ValueError("noise device and dtype must match the session")
        if self.denoising_step_list.numel() == 0:
            raise RuntimeError("denoising_step_list must not be empty")

        start_frame = session.current_start_frame
        if action is None:
            current_condition = cond_current(
                session.conditional_dict,
                start_frame,
                self.num_frame_per_block,
                mode=session.mode,
            )
        else:
            current_condition, updated_condition = cond_current(
                session.conditional_dict,
                start_frame,
                self.num_frame_per_block,
                replace=action,
                mode=session.mode,
            )
            session.conditional_dict = updated_condition

        model_start = model_end = decode_end = None
        if profile and noise.is_cuda:
            timing_stream = torch.cuda.current_stream(session.device)
            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            model_start.record(timing_stream)

        noisy_input = noise
        denoised_pred: Optional[torch.Tensor] = None
        timestep: Optional[torch.Tensor] = None
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [session.batch_size, self.num_frame_per_block],
                device=session.device,
                dtype=torch.int64,
            ) * current_timestep
            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=current_condition,
                timestep=timestep,
                kv_cache=session.kv_cache,
                kv_cache_mouse=session.mouse_kv_cache,
                kv_cache_keyboard=session.keyboard_kv_cache,
                crossattn_cache=session.crossattn_cache,
                current_start=start_frame * self.frame_seq_length,
            )

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                flat_prediction = rearrange(
                    denoised_pred, "b c f h w -> (b f) c h w"
                )
                noisy_input = self.scheduler.add_noise(
                    flat_prediction,
                    torch.randn_like(flat_prediction),
                    next_timestep * torch.ones(
                        [session.batch_size * self.num_frame_per_block],
                        device=session.device,
                        dtype=torch.long,
                    ),
                )
                noisy_input = rearrange(
                    noisy_input,
                    "(b f) c h w -> b c f h w",
                    b=denoised_pred.shape[0],
                )

        assert denoised_pred is not None and timestep is not None
        context_timestep = torch.ones_like(timestep) * self.args.context_noise
        overlap = self.overlap_vae_decode and noise.is_cuda
        decode_start = block_end = None
        caller_stream = torch.cuda.current_stream(session.device) if overlap else None
        try:
            if overlap:
                decode_stream = self._get_decode_stream(session.device)
                denoise_ready = torch.cuda.Event()
                denoise_ready.record(caller_stream)
                decode_stream.wait_event(denoise_ready)
                denoised_pred.record_stream(decode_stream)
                with torch.cuda.stream(decode_stream):
                    if profile:
                        decode_start = torch.cuda.Event(enable_timing=True)
                        decode_start.record(decode_stream)
                    video, decoded_cache = self.vae_decoder(
                        denoised_pred.transpose(1, 2).half(), *session.vae_cache
                    )
                    if decode_end is None:
                        decode_end = torch.cuda.Event()
                    decode_end.record(decode_stream)
                    self._decode_complete = decode_end

            # The clean-context refresh owns only generator/action caches;
            # decoding owns only the independent recurrent VAE feature cache.
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=current_condition,
                timestep=context_timestep,
                kv_cache=session.kv_cache,
                kv_cache_mouse=session.mouse_kv_cache,
                kv_cache_keyboard=session.keyboard_kv_cache,
                crossattn_cache=session.crossattn_cache,
                current_start=start_frame * self.frame_seq_length,
            )
            if model_end is not None:
                model_end.record(timing_stream)

            if overlap:
                caller_stream.wait_event(decode_end)
                self._record_cache_stream((video, decoded_cache), caller_stream)
                self._decode_overlap_runtime["calls"] += 1
                self._record_decode_overlap("decode-stream-overlap-enqueued")
            else:
                video, decoded_cache = self.vae_decoder(
                    denoised_pred.transpose(1, 2).half(), *session.vae_cache
                )
                if self.overlap_vae_decode:
                    self._record_decode_overlap("synchronous (non-CUDA fallback)")

            model_ms = decode_ms = total_ms = None
            if profile and decode_end is not None:
                assert model_start is not None and model_end is not None
                if overlap:
                    block_end = torch.cuda.Event(enable_timing=True)
                    block_end.record(caller_stream)
                    block_end.synchronize()
                    model_ms = model_start.elapsed_time(model_end)
                    decode_ms = decode_start.elapsed_time(decode_end)
                    total_ms = model_start.elapsed_time(block_end)
                else:
                    decode_end.record(timing_stream)
                    decode_end.synchronize()
                    model_ms = model_start.elapsed_time(model_end)
                    decode_ms = model_end.elapsed_time(decode_end)
                    total_ms = model_ms + decode_ms
        except BaseException:
            if overlap:
                session.invalidated = True
                self._drain_decode_stream()
            raise

        session.vae_cache = decoded_cache
        session.current_start_frame += self.num_frame_per_block

        if profile and model_ms is not None and decode_ms is not None:
            print(f"model_time: {model_ms}", flush=True)
            print(f"decode_time: {decode_ms}", flush=True)
            fps = video.shape[1] * 1000 / total_ms
            print(f"  - FPS: {fps:.2f}")

        session.last_model_ms = model_ms
        session.last_decode_ms = decode_ms
        block = CausalInferenceBlock(
            latent=denoised_pred,
            video=video,
            start_frame=start_frame,
            end_frame=session.current_start_frame,
            model_ms=model_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
        )
        self._last_block = block
        return block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: Mapping[str, Any],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        mode: str = "universal",
        profile: bool = False,
    ) -> torch.Tensor:
        """Run the legacy finite rollout through the resident block API."""

        if noise.ndim != 5 or noise.shape[1] != 16:
            raise ValueError("noise must have shape [B, 16, F, H, W]")
        num_frames = noise.shape[2]
        if num_frames % self.num_frame_per_block != 0:
            raise ValueError("noise frames must be divisible by num_frame_per_block")

        session = self.start_session(
            conditional_dict,
            initial_latent=initial_latent,
            reference_tensor=noise,
            mode=mode,
        )
        videos: List[torch.Tensor] = []
        generated_latents: List[torch.Tensor] = []
        num_blocks = num_frames // self.num_frame_per_block
        for block_index in tqdm(range(num_blocks)):
            noise_start = block_index * self.num_frame_per_block
            block = self.generate_next_block(
                session,
                noise[:, :, noise_start:noise_start + self.num_frame_per_block],
                profile=profile,
            )
            videos.append(block.video)
            generated_latents.append(block.latent)

        if return_latents:
            latent_parts: List[torch.Tensor] = []
            if initial_latent is not None:
                latent_parts.append(initial_latent)
            latent_parts.extend(generated_latents)
            return torch.cat(latent_parts, dim=2)
        return videos

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "_host_global_end_index": 0,
                "_host_local_end_index": 0,
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "_host_global_end_index": 0,
                "_host_local_end_index": 0,
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "_host_global_end_index": 0,
                "_host_local_end_index": 0,
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
