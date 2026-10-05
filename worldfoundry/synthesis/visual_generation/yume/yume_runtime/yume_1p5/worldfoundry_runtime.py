import os
import socket
import torch
import numpy as np
from diffusers.video_processor import VideoProcessor

from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan22 import (
    WAN_CONFIGS,
    SIZE_CONFIGS,
    MAX_AREA_CONFIGS
)
from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume_1p5 import Yume1p5TI2V


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma


class Yume1p5Runtime:

    def __init__(
        self,
        model,
        device,
        weight_dtype,
        owns_process_group=False,
    ) -> None:
        self.model = model
        self.weight_dtype = weight_dtype
        self.device = device
        self.owns_process_group = owns_process_group


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device,
        weight_dtype,
        fsdp
    ) -> "Yume1p5Runtime":
        
        torch.backends.cuda.matmul.allow_tf32 = True
        
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            raise FileNotFoundError(
                "Yume 1.5 requires a local checkpoint directory. "
                f"Runtime downloads are disabled for strict in-tree execution: {pretrained_model_path}"
            )
        

        # set device and distributed settings
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        import torch.distributed as dist
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(rank)
        owns_process_group = not dist.is_initialized()
        if owns_process_group:
            if int(os.environ.get("WORLD_SIZE", "1")) == 1:
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                os.environ.setdefault("LOCAL_RANK", "0")
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", _free_port())
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        try:
            cfg = WAN_CONFIGS["ti2v-5B"]
            model = Yume1p5TI2V(
                config=cfg,
                checkpoint_dir=model_root,
                device_id=rank,
                dit_fsdp=fsdp,
            )

            model.model.eval().requires_grad_(False).to(weight_dtype)
            if not fsdp:
                model.model.to(device)
        except BaseException:
            if owns_process_group and dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            raise

        return cls(
            model=model,
            device=device,
            weight_dtype=weight_dtype,
            owns_process_group=owns_process_group,
        )

    def cleanup(self):
        """Destroy the single-rank process group created by this runtime."""
        import torch.distributed as dist

        if self.owns_process_group and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            self.owns_process_group = False
    
    @torch.no_grad()
    def predict_per_interaction(
        self, 
        prompt,
        image,
        video,
        interaction_idx,
        interaction,
        interaction_caption,
        interaction_speed,
        interaction_distance,
        task_type,
        size,
        seed,
        max_area,
        current_latent_num,
        current_frame_num,
        num_euler_timesteps,
        history_latents=None
    ):

        # prepare input caption
        caption = "First-person perspective." + interaction_caption

        INTERACTION_SPEED_and_DISTANCE_2_CAPTION_DICT = {
            "movement": "Actual distance moved: {distance} at {speed} meters per second.",
            "rotation": "View rotation speed: {speed}."
        }

        if "camera_" not in interaction.lower():
            interaction_type = "movement"
            caption += INTERACTION_SPEED_and_DISTANCE_2_CAPTION_DICT[interaction_type].format(
                speed=interaction_speed,
                distance=interaction_distance
            )
        else:
            interaction_type = "rotation"
            caption += INTERACTION_SPEED_and_DISTANCE_2_CAPTION_DICT[interaction_type].format(
                speed=interaction_speed
            )


        if interaction_idx == 0: # first interaction
            prompt = prompt if prompt else ""
            caption = prompt + caption

            if task_type != "t2v":

                if task_type == "i2v":
                    video_tensor = torch.zeros(image.shape[0], 1+current_frame_num, size[0], size[1]) # (C, 33, H, W)
                    video_tensor[:, 0] = (image - 0.5) * 2
                    video = video_tensor.permute(1, 0, 2, 3) # (33, C, H, W)


                visual_content = video.squeeze().permute(1, 0, 2, 3).contiguous().to(self.device) # (C, F, H, W)

                visual_content_extended = torch.cat([visual_content[:, 0].unsqueeze(1).repeat(1, 16, 1, 1), visual_content[:,:33]], dim=1)
                history_current_frame_num = visual_content_extended.shape[1]

                visual_latents = torch.cat(
                    [
                        self.model.vae.encode([visual_content_extended.to(self.device)[:,:-32].to(self.device)])[0], \
                        self.model.vae.encode([visual_content_extended.to(self.device)[:,-32:].to(self.device)])[0]
                    ],
                    dim=1
                ) 
                history_latents = visual_latents[:, :-current_latent_num]

            else:
                history_current_frame_num = current_frame_num
                history_latents = None

        else: # continuation sampling
            history_current_frame_num = (history_latents.shape[1]-1)*4+1+32     

        if task_type != "t2v" or interaction_idx > 0:
            arg_c, arg_null, noise, mask2, img = self.model.generate(
                caption,
                frame_num=history_current_frame_num,
                max_area=max_area,
                current_latent_num=current_latent_num,
                img=history_latents,
                seed=seed
            )
        else:
            arg_c, arg_null, noise = self.model.generate(
                caption,
                frame_num=history_current_frame_num,
                size=(size[1], size[0]),
                max_area=max_area,
                current_latent_num=current_latent_num,
                seed=seed
            )

        if interaction_idx == 0: # first interaction
            latent = noise
        else: # continuation sampling
            history_zero_latents = torch.cat([history_latents, torch.zeros(48, current_latent_num, history_latents.shape[2], history_latents.shape[3]).to(self.device)], dim=1)
            noise = torch.randn_like(history_zero_latents)
            latent = noise.clone()


        sample_step_num = num_euler_timesteps
        sampling_sigmas = get_sampling_sigmas(sample_step_num, 7.0)


        if task_type != "t2v" or interaction_idx > 0:
            latent = torch.cat([img[0][:, :-current_latent_num, :, :], latent[:, -current_latent_num:, :, :]], dim=1)
    

        # update current latents
        with torch.autocast("cuda", dtype=self.weight_dtype):
            for i in range(sample_step_num):
                latent_model_input = [latent.squeeze(0)]

                if task_type != "t2v" or interaction_idx > 0:

                    timestep = [sampling_sigmas[i] * 1000]
                    timestep = torch.tensor(timestep).to(self.device)
                    temp_ts = (mask2[0][0][:-current_latent_num, ::2, ::2]).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(arg_c['seq_len'] - temp_ts.size(0)) * timestep
                    ])
                    timestep = temp_ts.unsqueeze(0)

                    noise_pred_cond = self.model.model(latent_model_input, t=timestep, **arg_c)[0]

                    if i + 1 == sample_step_num:
                        temp_x0 = latent[:, -current_latent_num:, :, :] + (0 - sampling_sigmas[i]) * noise_pred_cond[:, -current_latent_num:, :, :]
                    else:
                        temp_x0 = latent[:, -current_latent_num:, :, :] + (sampling_sigmas[i + 1] - sampling_sigmas[i]) * noise_pred_cond[:, -current_latent_num:, :, :]

                else:
                    timestep = [sampling_sigmas[i] * 1000]
                    timestep = torch.tensor(timestep).to(self.device)

                    noise_pred_cond = self.model.model(latent_model_input, t=timestep, flag=False, **arg_c)[0]

                    if i + 1 == sample_step_num:
                        latent = latent + (0 - sampling_sigmas[i]) * noise_pred_cond
                    else:
                        latent = latent + (sampling_sigmas[i + 1] - sampling_sigmas[i]) * noise_pred_cond

                if interaction_idx > 0:
                    latent = torch.cat([history_zero_latents[:, :-current_latent_num, :, :], temp_x0], dim=1)
                elif task_type != "t2v":
                    latent = torch.cat([visual_latents[:, :-current_latent_num, :, :], temp_x0], dim=1)


        if interaction_idx > 0:
            history_current_latents = torch.cat([history_latents, latent[:, -current_latent_num:, :, :]], dim=1)
        else:
            if task_type != "t2v":
                history_current_latents = torch.cat([visual_latents[:, :-current_latent_num, :, :], latent[:, -current_latent_num:, :, :]], dim=1)
            else:
                history_current_latents = latent

        with torch.autocast("cuda", dtype=torch.bfloat16):
            history_current_video = self.model.vae.decode([history_current_latents[:, -current_latent_num:, :, :].to(torch.float32)])[0]
            current_video = history_current_video[:, -current_frame_num:]

        return current_video, history_current_latents
            
    
    @torch.no_grad()
    def predict(
        self, 
        prompt, 
        image, 
        video, 
        interactions,
        interaction_captions, 
        interaction_speeds, 
        interaction_distances, 
        task_type, 
        size,
        seed,
        num_euler_timesteps
    ):
        # configs
        current_latent_num = 8 # time compression ratio is 4, so 8 latents corresponds to 32 frames
        current_frame_num = 32


        # inference per interaction
        output_video_list = []

        for interaction_idx, interaction_caption in enumerate(interaction_captions):
            output_video_per_interaction, history_latents = self.predict_per_interaction(
                prompt=prompt,
                image=image,
                video=video if interaction_idx == 0 else output_video_list[-1],
                interaction_idx=interaction_idx,
                interaction=interactions[interaction_idx],
                interaction_caption=interaction_caption, 
                interaction_speed=interaction_speeds[interaction_idx],
                interaction_distance=interaction_distances[interaction_idx], 
                task_type=task_type,
                size=SIZE_CONFIGS[size], 
                seed=seed,
                num_euler_timesteps=num_euler_timesteps,
                max_area=MAX_AREA_CONFIGS[size],
                current_latent_num=current_latent_num,
                current_frame_num=current_frame_num,
                history_latents=history_latents if interaction_idx > 0 else None
            )
            output_video_list.append(output_video_per_interaction)


        # postprocess output video
        vae_spatial_scale_factor = 8
        video_processor = VideoProcessor(vae_scale_factor=vae_spatial_scale_factor)

        output_video = video_processor.postprocess_video(torch.cat(output_video_list, dim=1).unsqueeze(0), output_type="pil")[0]

        return output_video
