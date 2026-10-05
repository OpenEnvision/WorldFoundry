# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spatial/temporal video tiling so one decode does not blow RAM.

:class:`TileProcessor` / :class:`ParallelHelper` shard huge frames,
run a callable per tile, then blend. Used by eval and long I2V — not
by the DiT forward (that uses latent chunks instead).
"""

import math
from collections import OrderedDict
from typing import List

import torch
from tqdm import tqdm

# Blend helpers are compiled lazily in ``VideoTiler.__init__`` so importing
# ``worldfoundry.core.io`` does not configure the persistent compiler cache.

# ──────────────────────────────────────────────────────────────────────────
# Tile shard / gather / overlap-blend — eval and long I2V, not DiT forward
# ──────────────────────────────────────────────────────────────────────────


class ParallelHelper:
    """Distribute video tiles by cost and reconstruct their global ordering.

    In non-distributed execution every tile remains local. With a process group,
    larger tiles are round-robin balanced across ranks and decoded frame tensors
    are gathered with explicit count, dtype, and shape metadata.
    """

    def __init__(self):
        """Stateless helper; all work is on the static methods."""

        pass

    @staticmethod
    def split_tile_list(
        tile_numel_dict: OrderedDict[int, int], parallel_group: torch.distributed.ProcessGroup = None
    ) -> List[int]:
        """
        Splits the given tile size into a list of sizes that each rank should handle.

        This method takes into account the number of ranks in a distributed setting.
        If the distributed environment is not initialized, it returns a list of
        integers from 0 to tile_size - 1, representing each tile index.

        If the distributed environment is initialized, it calculates the base tile size
        for each rank and distributes any remaining tiles among the ranks.

        Args:
            tile_numel_dict (OrderedDict[int, int]): Dict of index and numel of tiles.
            parallel_group (torch.distributed.ProcessGroup, optional):
                Distributed decoding group. Defaults to None.

        Returns:
            List[int]: A list of tile indices assigned to the current rank.
            List[int]: A list of global tile indices.
        """
        if not torch.distributed.is_initialized():
            return list(range(len(tile_numel_dict))), list(range(len(tile_numel_dict)))
        else:
            tile_idxs = list(OrderedDict(sorted(tile_numel_dict.items(), key=lambda x: x[1], reverse=True)).keys())
            world_size = torch.distributed.get_world_size(group=parallel_group)
            cur_rank = torch.distributed.get_rank(group=parallel_group)
            global_tile_idxs = []
            cur_rank_tile_idxs = []
            for rank in range(world_size):
                rank_tile_idxs = [tile_idxs[rank + world_size * i] for i in range(len(tile_idxs) // world_size)]
                if rank < len(tile_idxs) % world_size:
                    rank_tile_idxs.append(tile_idxs[len(tile_idxs) // world_size * world_size + rank])
                if rank == cur_rank:
                    cur_rank_tile_idxs = rank_tile_idxs
                global_tile_idxs = global_tile_idxs + rank_tile_idxs
            return cur_rank_tile_idxs, global_tile_idxs

    @staticmethod
    def _all_gather_fixed(
        tensor: torch.Tensor,
        *,
        world_size: int,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """Gather equal-shaped tensors into ``[world_size, ...]`` storage."""

        tensor = tensor.contiguous()
        if hasattr(torch.distributed, "all_gather_into_tensor"):
            output = tensor.new_empty((world_size * tensor.shape[0], *tensor.shape[1:]))
            torch.distributed.all_gather_into_tensor(output, tensor, group=parallel_group)
            return output.reshape(world_size, *tensor.shape)
        outputs = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(outputs, tensor, group=parallel_group)
        return torch.stack(outputs, dim=0)

    @staticmethod
    def gather_frames(
        frames: List[torch.Tensor], global_tile_idxs: List[int], parallel_group: torch.distributed.ProcessGroup = None
    ) -> List[torch.Tensor]:
        """
        Gathers frame data from all ranks in a distributed environment.

        This method collects frames from all ranks and combines them into a single list.
        If the distributed environment is not initialized, it simply returns the input frames.

        Args:
            frames (List[torch.Tensor]): A list of frames (tensors) from the current rank.
            global_tile_idxs (List[int]): A list of global tile indices.
            parallel_group (torch.distributed.ProcessGroup, optional):
                Distributed decoding group. Defaults to None.

        Returns:
            List[torch.Tensor]: A list of frames (tensors) from all ranks.
        """
        if not torch.distributed.is_initialized():
            return frames
        world_size = torch.distributed.get_world_size(group=parallel_group)
        if frames:
            device = frames[0].device
        elif "nccl" in str(torch.distributed.get_backend(parallel_group)).lower() and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        dtype_codes = {
            torch.float16: 1,
            torch.bfloat16: 2,
            torch.float32: 3,
            torch.float64: 4,
            torch.uint8: 5,
            torch.int32: 6,
            torch.int64: 7,
        }
        code_dtypes = {code: dtype for dtype, code in dtype_codes.items()}
        if frames:
            dtype = frames[0].dtype
            if dtype not in dtype_codes:
                raise TypeError(f"Unsupported distributed tile dtype: {dtype}")
            if any(frame.ndim != 5 or frame.dtype != dtype or frame.device != device for frame in frames):
                raise ValueError("Distributed tile frames must share one 5D shape contract, dtype, and device")
            local_dtype_code = dtype_codes[dtype]
        else:
            local_dtype_code = 0

        local_header = torch.tensor(
            (len(frames), local_dtype_code),
            dtype=torch.int64,
            device=device,
        ).view(1, 2)
        headers = ParallelHelper._all_gather_fixed(
            local_header,
            world_size=world_size,
            parallel_group=parallel_group,
        ).reshape(world_size, 2)
        counts = [int(value) for value in headers[:, 0].tolist()]
        total_count = sum(counts)
        if total_count == 0:
            if global_tile_idxs:
                raise ValueError("Tile index metadata is non-empty but no distributed rank produced a frame")
            return []
        nonzero_dtype_codes = {int(value) for value in headers[:, 1].tolist() if int(value)}
        if len(nonzero_dtype_codes) != 1:
            raise RuntimeError(f"Distributed tile ranks disagree on dtype: {sorted(nonzero_dtype_codes)}")
        dtype = code_dtypes[nonzero_dtype_codes.pop()]

        max_count = max(counts, default=0)
        local_shapes = torch.zeros((max_count, 5), dtype=torch.int64, device=device)
        for index, frame in enumerate(frames):
            local_shapes[index] = torch.tensor(frame.shape, dtype=torch.int64, device=device)
        gathered_shapes = ParallelHelper._all_gather_fixed(
            local_shapes,
            world_size=world_size,
            parallel_group=parallel_group,
        )
        all_rank_shapes = [
            [tuple(int(value) for value in gathered_shapes[rank, index].tolist()) for index in range(count)]
            for rank, count in enumerate(counts)
        ]
        total_sizes = [sum(math.prod(shape) for shape in shapes) for shapes in all_rank_shapes]
        max_total_size = max(total_sizes, default=0)

        flattened = torch.cat([frame.reshape(-1) for frame in frames]) if frames else torch.empty(0, dtype=dtype, device=device)
        padded = torch.zeros(max_total_size, dtype=dtype, device=device)
        if flattened.numel():
            padded[: flattened.numel()].copy_(flattened)
        gathered_data = ParallelHelper._all_gather_fixed(
            padded.view(1, -1),
            world_size=world_size,
            parallel_group=parallel_group,
        ).reshape(world_size, max_total_size)

        result_frames = []
        for rank, shapes in enumerate(all_rank_shapes):
            offset = 0
            for shape in shapes:
                size = math.prod(shape)
                result_frames.append(gathered_data[rank, offset : offset + size].view(shape))
                offset += size
        if len(global_tile_idxs) != len(result_frames):
            raise ValueError(
                "Distributed tile index count does not match gathered frames: "
                f"{len(global_tile_idxs)} != {len(result_frames)}"
            )
        result_frames_dict = OrderedDict((idx, frame) for idx, frame in zip(global_tile_idxs, result_frames))
        return list(OrderedDict(sorted(result_frames_dict.items())).values())

    @staticmethod
    def index_undot(index: int, loop_size: List[int]) -> List[int]:
        """
        Converts a single index into a list of indices, representing the position in a multi-dimensional space.

        This method takes an integer index and a list of loop sizes, and converts the index into a list of indices
        that correspond to the position in a multi-dimensional space.

        Args:
            index (int): The single index to be converted.
            loop_size (List[int]): A list of integers representing the size of each dimension in the multi-dimensional space.

        Returns:
            List[int]: A list of integers representing the position in the multi-dimensional space.
        """
        undotted_index = []
        for i in range(len(loop_size) - 1, -1, -1):
            undotted_index.append(index % loop_size[i])
            index = index // loop_size[i]
        undotted_index.reverse()
        assert len(undotted_index) == len(loop_size)
        return undotted_index

    @staticmethod
    def index_dot(index: List[int], loop_size: List[int]) -> int:
        """
        Converts a list of indices into a single index, representing the position in a multi-dimensional space.

        This method takes a list of indices and a list of loop sizes, and converts the list of indices into a single index
        that corresponds to the position in a multi-dimensional space.

        Args:
            index (List[int]): A list of integers representing the position in the multi-dimensional space.
            loop_size (List[int]): A list of integers representing the size of each dimension in the multi-dimensional space.

        Returns:
            int: A single integer representing the position in the multi-dimensional space.
        """
        assert len(index) == len(loop_size)
        dot_index = 0
        strides = [1]
        for i in range(len(loop_size) - 1, -1, -1):
            strides.append(strides[-1] * loop_size[i])
        strides.reverse()
        strides = strides[1:]
        assert len(index) == len(strides)
        for i in range(len(index)):
            dot_index += index[i] * strides[i]
        return dot_index


class TileProcessor:
    """Encode or decode large videos through overlapping spatiotemporal tiles.

    The processor derives latent tile sizes from codec downsample factors,
    schedules tiles locally or across ``parallel_group``, and linearly blends
    temporal/vertical/horizontal overlaps to suppress seams. ``encode_fn`` and
    ``decode_fn`` remain model-owned callables.
    """

    def __init__(
        self,
        encode_fn,
        decode_fn,
        tile_sample_min_height: int = 256,
        tile_sample_min_width: int = 256,
        tile_sample_min_length: int = 16,
        spatial_downsample_factor: int = 8,
        temporal_downsample_factor: int = 1,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        sr_ratio=1,
        first_frame_as_image: bool = False,
        parallel_group: torch.distributed.ProcessGroup = None,
    ):
        """
        Initializes an instance of the class.

        Args:
            encode_fn (function): The encoding function used for tile sampling.
            decode_fn (function): The decoding function used for tile reconstruction.
            tile_sample_min_size (int, optional): The minimum size of the sampled tiles. Defaults to 256.
            tile_sample_min_length (int, optional): The minimum length of the sampled tiles. Defaults to 16.
            spatial_downsample_factor (int, optional): The actual spataial downsample factor of given encode_fn. Defaults to 8.
            temporal_downsample_factor (int, optional): The actual temporal downsample factor of the latent space tiles. Defaults to 1.
            tile_overlap_factor (float, optional): The overlap factor between adjacent tiles. Defaults to 0.25.
            parallel_group (torch.distributed.ProcessGroup, optional): Distributed decoding group. Defaults to None.
        """
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_min_length = tile_sample_min_length
        self.tile_latent_min_height = tile_sample_min_height // spatial_downsample_factor
        self.tile_latent_min_width = tile_sample_min_width // spatial_downsample_factor

        self.tile_latent_min_length = tile_sample_min_length // temporal_downsample_factor
        if first_frame_as_image:
            self.tile_latent_min_length += 1

        self.spatial_tile_overlap_factor = spatial_tile_overlap_factor
        self.temporal_tile_overlap_factor = temporal_tile_overlap_factor
        self.sr_ratio = sr_ratio
        self.parallel_group = parallel_group
        from worldfoundry.core.execution.compile_cache import CompilePolicy, compile_callable_cached

        blend_policy = CompilePolicy(dynamic=False)
        self._blend_t_compiled = compile_callable_cached(
            self.blend_t,
            policy=blend_policy,
            namespace="video-tiling",
        )
        self._blend_v_compiled = compile_callable_cached(
            self.blend_v,
            policy=blend_policy,
            namespace="video-tiling",
        )
        self._blend_h_compiled = compile_callable_cached(
            self.blend_h,
            policy=blend_policy,
            namespace="video-tiling",
        )

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Linear-ramp overlap along time so adjacent temporal tiles do not seam."""

        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for t in range(blend_extent):
            b[:, :, t, :, :] = a[:, :, -blend_extent + t, :, :] * (1 - t / blend_extent) + b[:, :, t, :, :] * (
                t / blend_extent
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Linear-ramp overlap along height (vertical tile seams)."""

        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Linear-ramp overlap along width (horizontal tile seams)."""

        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.FloatTensor, verbose: bool = False):
        """Encode *x* tile-by-tile and blend overlaps in latent space."""

        overlap_height = int(self.tile_sample_min_height * (1 - self.spatial_tile_overlap_factor))
        overlap_width = int(self.tile_sample_min_width * (1 - self.spatial_tile_overlap_factor))
        overlap_length = int(self.tile_sample_min_length * (1 - self.temporal_tile_overlap_factor))
        blend_extent_h = int(self.tile_latent_min_height * self.spatial_tile_overlap_factor)
        blend_extent_w = int(self.tile_latent_min_width * self.spatial_tile_overlap_factor)
        blend_extent_t = int(self.tile_latent_min_length * self.temporal_tile_overlap_factor)
        height_limit = self.tile_latent_min_height - blend_extent_h
        width_limit = self.tile_latent_min_width - blend_extent_w
        frame_limit = self.tile_latent_min_length - blend_extent_t

        length_tile_size = (x.shape[2] + overlap_length - 1) // overlap_length
        height_tile_size = (x.shape[3] + overlap_height - 1) // overlap_height
        width_tile_size = (x.shape[4] + overlap_width - 1) // overlap_width
        total_tile_size = length_tile_size * height_tile_size * width_tile_size
        for_loop_size = [length_tile_size, height_tile_size, width_tile_size]

        tiles = []
        tile_numel_dict = OrderedDict()
        for tile_index in range(total_tile_size):
            undot_tile_index = ParallelHelper.index_undot(tile_index, for_loop_size)
            f_idx, i_idx, j_idx = undot_tile_index
            f = f_idx * overlap_length
            i = i_idx * overlap_height
            j = j_idx * overlap_width

            # Extract the tile from the latent representation and decode it
            tile = x[
                :,
                :,
                f : f + self.tile_sample_min_length,
                i : i + self.tile_sample_min_height,
                j : j + self.tile_sample_min_width,
            ]
            tiles.append(tile)
            tile_numel_dict[tile_index] = tile.numel()
        tile_index_list, global_tile_index_list = ParallelHelper.split_tile_list(
            tile_numel_dict, parallel_group=self.parallel_group
        )
        progress_bar = tqdm(
            total=len(tile_index_list),
            desc=f"[Rank {torch.distributed.get_rank(group=self.parallel_group)}] Encoding Tiles",
            disable=not verbose,
        )

        frames = []
        # Encode each tile based on the tile index list
        for tile_index in tile_index_list:
            tile = tiles[tile_index]
            encoded = self.encode_fn(tile)
            frames.append(encoded)
            progress_bar.update(1)

        # Gather all decoded frames from different ranks
        frames = ParallelHelper.gather_frames(frames, global_tile_index_list, parallel_group=self.parallel_group)
        assert len(frames) == total_tile_size
        progress_bar.close()

        result_frames = []
        # Blend the encoded tiles to create the final output
        for tile_index in range(total_tile_size):
            undot_tile_index = ParallelHelper.index_undot(tile_index, for_loop_size)
            f, i, j = undot_tile_index

            tile = frames[tile_index]
            # Blend with previous tiles if applicable
            if f > 0:
                idx = ParallelHelper.index_dot([f - 1, i, j], for_loop_size)
                tile = self.blend_t(frames[idx], tile, blend_extent_t)
            if i > 0:
                idx = ParallelHelper.index_dot([f, i - 1, j], for_loop_size)
                tile = self.blend_v(frames[idx], tile, blend_extent_h)
            if j > 0:
                idx = ParallelHelper.index_dot([f, i, j - 1], for_loop_size)
                tile = self.blend_h(frames[idx], tile, blend_extent_w)
            result_frames.append(tile[:, :, :frame_limit, :height_limit, :width_limit])

        assert len(result_frames) == total_tile_size

        concat_frames = []
        for f in range(length_tile_size):
            result_rows = []
            for i in range(height_tile_size):
                result_row = []
                for j in range(width_tile_size):
                    idx = ParallelHelper.index_dot([f, i, j], for_loop_size)
                    result_row.append(result_frames[idx])
                result_rows.append(torch.cat(result_row, dim=4))
            concat_frames.append(torch.cat(result_rows, dim=3))

        # Concatenate all result frames along the temporal dimension
        result = torch.cat(concat_frames, dim=2)
        return result

    def tiled_decode(self, z: torch.FloatTensor, verbose: bool = False):
        """Decode latent tiles and blend overlaps in pixel space."""

        overlap_height = int(self.tile_latent_min_height * (1 - self.spatial_tile_overlap_factor))
        overlap_width = int(self.tile_latent_min_width * (1 - self.spatial_tile_overlap_factor))
        overlap_length = int(self.tile_latent_min_length * (1 - self.temporal_tile_overlap_factor))

        real_tile_sample_min_height = int(self.tile_latent_min_height * self.spatial_downsample_factor * self.sr_ratio)
        real_tile_sample_min_width = int(self.tile_latent_min_width * self.spatial_downsample_factor * self.sr_ratio)
        real_tile_sample_min_length = int(self.tile_latent_min_length * self.temporal_downsample_factor)

        blend_extent_h = int(real_tile_sample_min_height * self.spatial_tile_overlap_factor)
        blend_extent_w = int(real_tile_sample_min_width * self.spatial_tile_overlap_factor)
        blend_extent_t = int(real_tile_sample_min_length * self.temporal_tile_overlap_factor)

        height_limit = real_tile_sample_min_height - blend_extent_h
        width_limit = real_tile_sample_min_width - blend_extent_w
        frame_limit = real_tile_sample_min_length - blend_extent_t

        length_tile_size = (z.shape[2] + overlap_length - 1) // overlap_length
        height_tile_size = (z.shape[3] + overlap_height - 1) // overlap_height
        width_tile_size = (z.shape[4] + overlap_width - 1) // overlap_width
        total_tile_size = length_tile_size * height_tile_size * width_tile_size
        for_loop_size = [length_tile_size, height_tile_size, width_tile_size]

        tiles = []
        tile_numel_dict = OrderedDict()
        for tile_index in range(total_tile_size):
            undot_tile_index = ParallelHelper.index_undot(tile_index, for_loop_size)
            f_idx, i_idx, j_idx = undot_tile_index
            f = f_idx * overlap_length
            i = i_idx * overlap_height
            j = j_idx * overlap_width

            # Extract the tile from the latent representation and decode it
            tile = z[
                :,
                :,
                f : f + self.tile_latent_min_length,
                i : i + self.tile_latent_min_height,
                j : j + self.tile_latent_min_width,
            ]
            tiles.append(tile)
            tile_numel_dict[tile_index] = tile.numel()
        tile_index_list, global_tile_index_list = ParallelHelper.split_tile_list(
            tile_numel_dict, parallel_group=self.parallel_group
        )
        progress_bar = tqdm(
            total=len(tile_index_list),
            desc=f"[Rank {torch.distributed.get_rank(group=self.parallel_group)}] Decoding Tiles",
            disable=not verbose,
        )

        frames = []
        # Decode each tile based on the tile index list
        for tile_index in tile_index_list:
            tile = tiles[tile_index]
            decoded = self.decode_fn(tile)
            frames.append(decoded)
            progress_bar.update(1)

        progress_bar.close()
        # Gather all decoded frames from different ranks
        frames = ParallelHelper.gather_frames(frames, global_tile_index_list, parallel_group=self.parallel_group)
        assert len(frames) == total_tile_size

        result_frames = []
        # Blend the decoded tiles to create the final output
        for tile_index in tile_index_list:
            undot_tile_index = ParallelHelper.index_undot(tile_index, for_loop_size)
            f, i, j = undot_tile_index

            tile = frames[tile_index].clone()
            # Blend with previous tiles if applicable
            if f > 0:
                idx = ParallelHelper.index_dot([f - 1, i, j], for_loop_size)
                tile = self._blend_t_compiled(frames[idx], tile, blend_extent_t)
            if i > 0:
                idx = ParallelHelper.index_dot([f, i - 1, j], for_loop_size)
                tile = self._blend_v_compiled(frames[idx], tile, blend_extent_h)
            if j > 0:
                idx = ParallelHelper.index_dot([f, i, j - 1], for_loop_size)
                tile = self._blend_h_compiled(frames[idx], tile, blend_extent_w)
            result_frames.append(tile[:, :, :frame_limit, :height_limit, :width_limit])

        # Gather and concatenate the final result frames
        result_frames = ParallelHelper.gather_frames(
            result_frames, global_tile_index_list, parallel_group=self.parallel_group
        )
        assert len(result_frames) == total_tile_size

        concat_frames = []
        for f in range(length_tile_size):
            result_rows = []
            for i in range(height_tile_size):
                result_row = []
                for j in range(width_tile_size):
                    idx = ParallelHelper.index_dot([f, i, j], for_loop_size)
                    result_row.append(result_frames[idx])
                result_rows.append(torch.cat(result_row, dim=4))
            concat_frames.append(torch.cat(result_rows, dim=3))

        # Concatenate all result frames along the temporal dimension
        result = torch.cat(concat_frames, dim=2)
        return result
