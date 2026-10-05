import types
from pathlib import Path
from typing import List, Optional
import torch
from torch import nn
from worldfoundry.synthesis.visual_generation.echo_infinity.echo_infinity_runtime.utils.scheduler import SchedulerInterface, FlowMatchScheduler
from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import umt5_xxl
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import _video_vae
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.echo_infinity.model import WanModel, RegisterTokens, GanAttentionBlock
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.echo_infinity.causal import CausalWanModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.echo_infinity.infinity import CausalWanModel as CausalWanModelInfinity
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.echo_infinity.memory import CausalWanModelInfinityMemory

class WanTextEncoder(torch.nn.Module):

    def __init__(self, model_root: str | Path, device: str | torch.device | None = None) -> None:
        super().__init__()
        model_root = Path(model_root).expanduser().resolve()
        encoder_path = model_root / 'models_t5_umt5-xxl-enc-bf16.pth'
        tokenizer_path = model_root / 'google' / 'umt5-xxl'
        if not encoder_path.is_file():
            raise FileNotFoundError(f'Echo-Infinity T5 encoder weights not found: {encoder_path}')
        if not tokenizer_path.exists():
            raise FileNotFoundError(f'Echo-Infinity tokenizer directory not found: {tokenizer_path}')
        self.text_encoder = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.float32, device=torch.device('cpu')).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(torch.load(str(encoder_path), map_location='cpu', weights_only=False))
        target_device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = self.text_encoder.to(target_device)
        self.tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return {'prompt_embeds': context}

class WanVAEWrapper(torch.nn.Module):

    def __init__(self, model_root: str | Path):
        super().__init__()
        model_root = Path(model_root).expanduser().resolve()
        vae_path = model_root / 'Wan2.1_VAE.pth'
        if not vae_path.is_file():
            raise FileNotFoundError(f'Echo-Infinity VAE weights not found: {vae_path}')
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.model = _video_vae(pretrained_path=str(vae_path), z_dim=16).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = (pixel.device, pixel.dtype)
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool=False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, 'Batch size must be 1 when using cache'
        device, dtype = (latent.device, latent.dtype)
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]
        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode
        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel_chunk(self, latent: torch.Tensor, use_cache: bool=False, chunk_size: int=120) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, 'Batch size must be 1 when using cache'
        device, dtype = (latent.device, latent.dtype)
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]
        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode
        output = []
        for u in zs:
            num_frames = u.shape[1]
            if num_frames <= chunk_size:
                decoded = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                decoded = decoded.cpu()
            else:
                decoded_chunks = []
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    chunk = u[:, start_idx:end_idx, :, :]
                    self.model.clear_cache()
                    decoded_chunk = decode_function(chunk.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                    decoded_chunks.append(decoded_chunk.cpu())
                    del decoded_chunk
                    torch.cuda.empty_cache()
                decoded = torch.cat(decoded_chunks, dim=1)
                self.model.clear_cache()
            output.append(decoded)
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

class WanDiffusionWrapper(torch.nn.Module):

    def __init__(self, model_name='Wan2.1-T2V-1.3B', model_root: str | Path | None=None, wan_root: str | Path | None=None, timestep_shift=8.0, is_causal=False, local_attn_size=-1, sink_size=0, use_infinite_attention=False, dr_rope=False, tri_rope_cont=False, tri_rope_pmax=21, relative_rope=False, relative_rope_pmax=21):
        super().__init__()
        if model_root is None:
            if wan_root is None:
                raise ValueError('Echo-Infinity requires a local wan_root or model_root.')
            model_root = Path(wan_root).expanduser() / model_name
        model_root = Path(model_root).expanduser().resolve()
        if not model_root.exists():
            raise FileNotFoundError(f'Echo-Infinity Wan model root not found: {model_root}')
        model_root_str = str(model_root)
        if is_causal:
            if use_infinite_attention:
                if relative_rope:
                    self.model = CausalWanModelInfinityMemory.from_pretrained(model_root_str, local_attn_size=local_attn_size, sink_size=sink_size)
                    self.model.enable_infmem(relative_rope=True, relative_rope_pmax=relative_rope_pmax, num_frame_per_block_attr=3)
                else:
                    self.model = CausalWanModelInfinity.from_pretrained(model_root_str, local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                self.model = CausalWanModel.from_pretrained(model_root_str, local_attn_size=local_attn_size, sink_size=sink_size, dr_rope=dr_rope, tri_rope_cont=tri_rope_cont, tri_rope_pmax=tri_rope_pmax, relative_rope=relative_rope, relative_rope_pmax=relative_rope_pmax)
        else:
            self.model = WanModel.from_pretrained(model_root_str)
        self.model.eval()
        self.uniform_timestep = not is_causal
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000)
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        self._cls_pred_branch = nn.Sequential(nn.LayerNorm(atten_dim * 3 + time_embed_dim), nn.Linear(atten_dim * 3 + time_embed_dim, 1536), nn.SiLU(), nn.Linear(atten_dim, num_class))
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)
        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(lambda x: x.double().to(flow_pred.device), [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(lambda x: x.double().to(x0_pred.device), [x0_pred, xt, scheduler.sigmas, scheduler.timesteps])
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(self, noisy_image_or_video: torch.Tensor, conditional_dict: dict, timestep: torch.Tensor, kv_cache: Optional[List[dict]]=None, crossattn_cache: Optional[List[dict]]=None, current_start: Optional[int]=None, classify_mode: Optional[bool]=False, concat_time_embeddings: Optional[bool]=False, clean_x: Optional[torch.Tensor]=None, aug_t: Optional[torch.Tensor]=None, cache_start: Optional[int]=None, sink_recache_after_switch=False) -> torch.Tensor:
        prompt_embeds = conditional_dict['prompt_embeds']
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        logits = None
        if kv_cache is not None:
            flow_pred = self.model(noisy_image_or_video.permute(0, 2, 1, 3, 4), t=input_timestep, context=prompt_embeds, seq_len=self.seq_len, kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start=current_start, cache_start=cache_start, sink_recache_after_switch=sink_recache_after_switch).permute(0, 2, 1, 3, 4)
        elif clean_x is not None:
            flow_pred = self.model(noisy_image_or_video.permute(0, 2, 1, 3, 4), t=input_timestep, context=prompt_embeds, seq_len=self.seq_len, clean_x=clean_x.permute(0, 2, 1, 3, 4), aug_t=aug_t, sink_recache_after_switch=sink_recache_after_switch).permute(0, 2, 1, 3, 4)
        elif classify_mode:
            flow_pred, logits = self.model(noisy_image_or_video.permute(0, 2, 1, 3, 4), t=input_timestep, context=prompt_embeds, seq_len=self.seq_len, classify_mode=True, register_tokens=self._register_tokens, cls_pred_branch=self._cls_pred_branch, gan_ca_blocks=self._gan_ca_blocks, concat_time_embeddings=concat_time_embeddings, sink_recache_after_switch=sink_recache_after_switch)
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(noisy_image_or_video.permute(0, 2, 1, 3, 4), t=input_timestep, context=prompt_embeds, seq_len=self.seq_len, sink_recache_after_switch=sink_recache_after_switch).permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1), timestep=timestep.flatten(0, 1)).unflatten(0, flow_pred.shape[:2])
        if logits is not None:
            return (flow_pred, pred_x0, logits)
        return (flow_pred, pred_x0)

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()
