import torch
import torch.distributed as dist
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE as CanonicalWanVideoVAE
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    WanVideoVAE38 as CanonicalWanVideoVAE38,
)


class WanVideoVAE(CanonicalWanVideoVAE):
    def __init__(self, z_dim: int = 16) -> None:
        super().__init__(z_dim=z_dim)

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape

        device = data.device

        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0]).to(device)
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1]).to(device)

        mask = torch.min(h.view(H, 1), w.view(1, W))
        mask = mask.view(1, 1, 1, H, W)
        return mask

    def decode_dist(self, zs, world_size, cur_rank, split_dim):
        zs = zs.squeeze(0)
        splited_total_len = zs.shape[split_dim]
        splited_chunk_len = splited_total_len // world_size
        padding_size = 1

        if cur_rank == 0:
            if split_dim == 2:
                zs = zs[:, :, : splited_chunk_len + 2 * padding_size, :].contiguous()
            elif split_dim == 3:
                zs = zs[:, :, :, : splited_chunk_len + 2 * padding_size].contiguous()
        elif cur_rank == world_size - 1:
            if split_dim == 2:
                zs = zs[:, :, -(splited_chunk_len + 2 * padding_size) :, :].contiguous()
            elif split_dim == 3:
                zs = zs[:, :, :, -(splited_chunk_len + 2 * padding_size) :].contiguous()
        else:
            if split_dim == 2:
                zs = zs[
                    :,
                    :,
                    cur_rank * splited_chunk_len - padding_size : (cur_rank + 1) * splited_chunk_len + padding_size,
                    :,
                ].contiguous()
            elif split_dim == 3:
                zs = zs[
                    :,
                    :,
                    :,
                    cur_rank * splited_chunk_len - padding_size : (cur_rank + 1) * splited_chunk_len + padding_size,
                ].contiguous()

        images = self.model.decode(zs.unsqueeze(0), self.scale).float().clamp_(-1, 1)

        if cur_rank == 0:
            if split_dim == 2:
                images = images[:, :, :, : splited_chunk_len * 8, :].contiguous()
            elif split_dim == 3:
                images = images[:, :, :, :, : splited_chunk_len * 8].contiguous()
        elif cur_rank == world_size - 1:
            if split_dim == 2:
                images = images[:, :, :, -splited_chunk_len * 8 :, :].contiguous()
            elif split_dim == 3:
                images = images[:, :, :, :, -splited_chunk_len * 8 :].contiguous()
        else:
            if split_dim == 2:
                images = images[:, :, :, 8 * padding_size : -8 * padding_size, :].contiguous()
            elif split_dim == 3:
                images = images[:, :, :, :, 8 * padding_size : -8 * padding_size].contiguous()

        full_images = [torch.empty_like(images) for _ in range(world_size)]
        dist.all_gather(full_images, images)

        torch.cuda.synchronize()

        images = torch.cat(full_images, dim=split_dim + 1)

        return images

    def tiled_decode(self, hidden_states, device, tile_size, tile_stride, tile_batch_size=4):
        from collections import defaultdict

        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size

        stride_h = max(1, size_h - 4)
        stride_w = max(1, size_w - 4)

        tasks_by_shape = defaultdict(list)

        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue

                h_orig, w_orig = h + size_h, w + size_w
                actual_h_ = min(h_orig, H)
                actual_w_ = min(w_orig, W)

                shape = (actual_h_ - h, actual_w_ - w)
                tasks_by_shape[shape].append((h, h_orig, actual_h_, w, w_orig, actual_w_))

        batched_tasks = []
        for shape, tasks in tasks_by_shape.items():  # noqa: B007 - The loop variable is read outside the loop.
            for i in range(0, len(tasks), tile_batch_size):
                batched_tasks.append(tasks[i : i + tile_batch_size])

        data_device = "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros(
            (1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )

        for batch in tqdm(batched_tasks, desc="VAE decoding (Ultimate Batched)"):
            hidden_states_batch = torch.cat(
                [hidden_states[:, :, :, h:actual_h_, w:actual_w_] for (h, _, actual_h_, w, _, actual_w_) in batch],
                dim=0,
            ).to(computation_device)

            out_batch = self.model.decode(hidden_states_batch, self.scale)

            out_batch = out_batch.to(hidden_states.dtype)

            for b_idx, (h, h_orig, actual_h_, w, w_orig, actual_w_) in enumerate(batch):  # noqa: B007 - The loop variable is read outside the loop.
                out_tile = out_batch[b_idx : b_idx + 1]
                mask = self.build_mask(
                    out_tile,
                    is_bound=(h == 0, h_orig >= H, w == 0, w_orig >= W),
                    border_width=(
                        (size_h - stride_h) * self.upsampling_factor,
                        (size_w - stride_w) * self.upsampling_factor,
                    ),
                )
                out_tile = (out_tile * mask).to(data_device)
                mask_cpu = mask.to(data_device)

                target_h = h * self.upsampling_factor
                target_w = w * self.upsampling_factor

                values[:, :, :, target_h : target_h + out_tile.shape[3], target_w : target_w + out_tile.shape[4]] += (
                    out_tile
                )
                weight[:, :, :, target_h : target_h + mask_cpu.shape[3], target_w : target_w + mask_cpu.shape[4]] += (
                    mask_cpu
                )

        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def single_encode(self, video, device):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x

    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos


class WanVideoVAE38(CanonicalWanVideoVAE38, WanVideoVAE):
    """Wan 2.2 construction with the family's existing tile/device orchestration."""

    def __init__(self, z_dim: int = 48, dim: int = 160) -> None:
        super().__init__(z_dim=z_dim, dim=dim)
