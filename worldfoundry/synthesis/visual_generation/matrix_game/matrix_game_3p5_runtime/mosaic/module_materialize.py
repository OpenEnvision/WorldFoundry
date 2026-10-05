"""GPU materialization for Matrix-Game 3.5 inference samples."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from .config import _normalize_mosaic_fuse_mode


class _MaterializeMixin:
    """Encode the inference sample and build its initial Mosaic memory."""

    def _get_inference_frustum_handler(self):
        cached = getattr(self, "_inference_frustum_handler", None)
        if cached is None:
            from ..frustum.frustum_handler import FrustumHandler

            cached = FrustumHandler(np.eye(3), latent_stride=16)
            self._inference_frustum_handler = cached
        return cached

    @staticmethod
    def _atomic_dump_latent(target_path, tensor):
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".tmp_",
            suffix=f".{os.getpid()}.pt",
            dir=target_dir,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(tensor.detach().to("cpu"), handle)
            os.replace(temporary_path, target_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _frames_to_vae_input(frames, *, device, dtype, pixel_range=None):
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)
        if frames.dtype == torch.uint8 or str(pixel_range or "").lower() in {
            "uint8",
            "uint8_or_255",
            "0_255",
        }:
            return frames.to(device=device, dtype=dtype).mul_(1.0 / 127.5).sub_(1.0)
        return frames.to(device=device, dtype=dtype)

    @staticmethod
    def _canonical_view_change_like_latents(latents, *, device=None, dtype=None):
        batch, _, frames, height, width = latents.shape
        output = torch.zeros(
            batch,
            frames,
            height,
            width,
            3,
            device=device or latents.device,
            dtype=dtype or latents.dtype,
        )
        output[..., 0] = 1.0
        return output

    def _materialize_subject_ref_latents(self, data, *, device, dtype):
        ref_images = data.get("subject_ref_images")
        if ref_images is None:
            return data
        if not torch.is_tensor(ref_images):
            ref_images = torch.as_tensor(ref_images)
        info = dict(data.get("subject_ref_latent_info") or {})
        if ref_images.ndim != 4 or int(ref_images.shape[0]) <= 0:
            data.pop("subject_ref_latents", None)
            info.update({"enabled": False, "status": "no_refs", "ref_count": 0})
        else:
            ref_count = int(ref_images.shape[0])
            codec_dtype = getattr(self, "vae_dtype", dtype)
            ref_videos = [
                ref_images[index].to(device=device, dtype=codec_dtype).unsqueeze(1)
                for index in range(ref_count)
            ]
            ref_latents = self.vae.encode(
                ref_videos,
                device=device,
                tiled=False,
            ).to(device=device, dtype=dtype)
            data["subject_ref_latents"] = ref_latents
            data["subject_ref_count"] = ref_count
            info.update(
                {
                    "enabled": True,
                    "status": "ok",
                    "ref_count": ref_count,
                    "ref_images_shape": tuple(ref_images.shape),
                    "ref_latents_shape": tuple(ref_latents.shape),
                    "dtype": str(ref_latents.dtype),
                    "device": str(ref_latents.device),
                }
            )
        data["subject_ref_latent_info"] = info
        return data

    def _encode_memory_cache_misses(
        self,
        memory_frames_by_id,
        *,
        device,
        dtype,
        pixel_range,
    ):
        prepared_by_length = {}
        replicate_frames = int(self.memory_vae_encode_input_frames)
        codec_dtype = getattr(self, "vae_dtype", dtype)
        for frame_id in sorted(int(value) for value in memory_frames_by_id):
            video = self._frames_to_vae_input(
                memory_frames_by_id[frame_id],
                device=device,
                dtype=codec_dtype,
                pixel_range=pixel_range,
            )
            if int(video.shape[1]) == 1 and replicate_frames > 1:
                video = video.repeat(1, replicate_frames, 1, 1)
            prepared_by_length.setdefault(int(video.shape[1]), []).append((frame_id, video))

        encoded = {}
        for rows in prepared_by_length.values():
            latents = self.vae.encode(
                [video for _, video in rows],
                device=device,
                tiled=False,
            )
            if int(latents.shape[2]) > 1:
                latents = latents[:, :, -1:].contiguous()
            for index, (frame_id, _) in enumerate(rows):
                encoded[frame_id] = latents[index].to(device=device, dtype=dtype)
        return encoded

    @staticmethod
    def _zero_mosaic_payload(data, noisy_latents, *, view_change):
        data["queried_latent"] = torch.zeros_like(noisy_latents)
        data["ql_rope_indice_pef"] = torch.zeros(
            noisy_latents.shape[2],
            noisy_latents.shape[-2],
            noisy_latents.shape[-1],
            2,
            dtype=torch.long,
            device=noisy_latents.device,
        )
        if view_change:
            data["ql_view_change_pef"] = _MaterializeMixin._canonical_view_change_like_latents(noisy_latents)
        data["needs_vae_materialization"] = False
        return data

    @torch.no_grad()
    def _materialize_data(self, data):
        """Encode one inference payload and fuse its initial memory latents."""
        if not data.get("needs_vae_materialization"):
            return data
        if int(data.get("cn_drop_count", 0) or 0) != 0:
            raise ValueError("Matrix-Game inference requires a single clean anchor (cn_drop_count must be 0).")

        device = self.device
        dtype = self.dtype
        codec_dtype = getattr(self, "vae_dtype", dtype)
        self._activate_components(["vae"])

        cn_frames = data["cn_frames"]
        cn_input = self._frames_to_vae_input(
            cn_frames,
            device=device,
            dtype=codec_dtype,
            pixel_range=data.get("frame_pixel_range"),
        )
        cn_latents = self.vae.encode(
            [cn_input],
            device=device,
            tiled=False,
        )[0].to(device=device, dtype=dtype)
        expected_steps = (int(cn_frames.shape[1]) - 1) // 4 + 1
        if int(cn_latents.shape[1]) != expected_steps:
            raise RuntimeError(
                f"VAE encoded {cn_frames.shape[1]} frames into "
                f"{cn_latents.shape[1]} latent steps; expected {expected_steps}."
            )
        clean_latents = cn_latents[:, :1].unsqueeze(0).contiguous()
        noisy_latents = cn_latents[:, 1:].unsqueeze(0).contiguous()
        data["clean_latents"] = clean_latents
        data["latents"] = noisy_latents
        data = self._materialize_subject_ref_latents(
            data,
            device=device,
            dtype=dtype,
        )

        fused_inputs = data["fused_query_inputs"]
        memory_total_frames = int(fused_inputs["memory_total_frames"])
        # JSON-backed datasets may deserialize frame ids as strings. Normalize
        # once so cache hits, misses, and write targets address the same slots.
        memory_frames_by_id = {
            int(frame_id): frames for frame_id, frames in (data.get("memory_frames_by_id", {}) or {}).items()
        }
        cached_paths = {
            int(frame_id): path for frame_id, path in (data.get("cached_memory_paths_by_id", {}) or {}).items()
        }
        target_paths = {
            int(frame_id): path for frame_id, path in (data.get("memory_cache_hash_by_id", {}) or {}).items()
        }
        memory_latents = torch.zeros(
            int(cn_latents.shape[0]),
            memory_total_frames,
            int(fused_inputs["H_lat"]),
            int(fused_inputs["W_lat"]),
            device=device,
            dtype=dtype,
        )

        for frame_id, cache_path in cached_paths.items():
            frame_id = int(frame_id)
            try:
                cached = torch.load(
                    cache_path,
                    map_location=device,
                    weights_only=True,
                )
            except Exception:
                if frame_id in memory_frames_by_id:
                    continue
                raise
            memory_latents[:, frame_id : frame_id + 1] = cached.to(
                device=device,
                dtype=dtype,
            )

        if memory_frames_by_id:
            encoded = self._encode_memory_cache_misses(
                memory_frames_by_id,
                device=device,
                dtype=dtype,
                pixel_range=data.get("frame_pixel_range"),
            )
            for frame_id, latent in encoded.items():
                memory_latents[:, frame_id] = latent[:, 0]
                target = target_paths.get(frame_id)
                if target:
                    try:
                        self._atomic_dump_latent(target, latent)
                    except OSError:
                        pass
        if memory_total_frames == 0:
            return self._zero_mosaic_payload(
                data,
                noisy_latents,
                view_change=bool(self.mosaic_view_change_prope),
            )

        fuse_mode = _normalize_mosaic_fuse_mode(self.mosaic_fuse_mode)
        if fuse_mode == "random":
            raise ValueError("Matrix-Game inference requires a deterministic fuse mode")
        handler = self._get_inference_frustum_handler()
        if "memory_K" in fused_inputs and "query_K" in fused_inputs:
            single_k = None
            memory_k = fused_inputs["memory_K"]
            query_k = fused_inputs["query_K"]
        else:
            single_k = fused_inputs["K"]
            memory_k = None
            query_k = None
        return_source_ids = bool(self.mosaic_return_source_frame_ids)
        fused = handler.fuse_candidates(
            query_extrinsics=fused_inputs["query_extrinsics"],
            candidate_frame_ids=fused_inputs["candidate_frame_ids"],
            latents=memory_latents.detach().float(),
            w2c=fused_inputs["w2c"],
            depths=fused_inputs["depths"],
            K=single_k,
            memory_K=memory_k,
            query_K=query_k,
            H_lat=int(fused_inputs["H_lat"]),
            W_lat=int(fused_inputs["W_lat"]),
            memory_total_frames=memory_total_frames,
            fuse_mode=fuse_mode,
            interpolation_mode="nearest",
            return_revgrid=True,
            return_view_change=bool(self.mosaic_view_change_prope),
            latent_merge_4frames=False,
            query_reference_frame=fused_inputs.get(
                "query_reference_frame",
                self.mosaic_query_reference_frame,
            ),
            fill_ratio_threshold=0.95,
            fuse_block_size=int(self.mosaic_fuse_block_size),
            return_source_frame_ids=return_source_ids,
            source_valid_masks=fused_inputs.get("source_valid_masks"),
        )
        data["queried_latent"] = fused[0].to(device=device, dtype=dtype)
        data["ql_rope_indice_pef"] = fused[1]
        offset = 2
        if self.mosaic_view_change_prope:
            data["ql_view_change_pef"] = fused[offset]
            offset += 1
        data["ql_source_frame_ids"] = fused[offset] if return_source_ids else None
        data["needs_vae_materialization"] = False
        return data


__all__ = ["_MaterializeMixin"]
