"""MagCache: skip a DiT forward and reuse the residual when the error stays small."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from torch import Tensor

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ConfigModel


class MagCacheConfig(ConfigModel):
    mag_ratios: list[float] | dict[str, list[float]]
    thresh: float = 0.12
    K: int = 2  # MagCache paper hyperparameter name
    retention_ratio: float = 0.2


def select_mag_ratios(
    mag_ratios: Sequence[float] | Mapping[str, Sequence[float]],
    mode: str | None = None,
) -> tuple[Sequence[float], str | None]:
    """Select mode-specific coefficients while keeping flat configs valid."""
    if not isinstance(mag_ratios, Mapping):
        return mag_ratios, None
    if mode is None:
        raise ValueError("mode is required when mag_ratios is a mapping")
    try:
        return mag_ratios[mode], mode
    except KeyError as exc:
        available = ", ".join(sorted(mag_ratios))
        raise ValueError(f"No MagCache ratios configured for mode {mode!r}; available modes: {available}") from exc


def nearest_interp(src_array: np.ndarray, target_length: int) -> np.ndarray:
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])
    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def prepare_mag_ratios(mag_ratios, num_steps: int) -> np.ndarray:
    mag_ratios = np.asarray(mag_ratios, dtype=np.float64)
    if len(mag_ratios) != num_steps * 2:
        mag_ratio_con = nearest_interp(mag_ratios[0::2], num_steps)
        mag_ratio_ucon = nearest_interp(mag_ratios[1::2], num_steps)
        mag_ratios = np.concatenate([mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
    return mag_ratios


class MagCache:
    """MagCache steps used by :class:`CacheDiT`."""

    def _reset_mag_state(self) -> None:
        if self._mag is None:
            return
        self._mag.update(
            {
                "cnt": 0,
                "accumulated_err": [0.0, 0.0],
                "accumulated_steps": [0, 0],
                "accumulated_ratio": [1.0, 1.0],
                "residual_cache": [None, None],
                "computed_forwards": 0,
                "skipped_forwards": 0,
                "last_generation_stats": None,
            }
        )

    def _mag_should_skip(self) -> tuple[bool, object]:
        st = self._mag
        assert st is not None
        skip = False
        residual = None
        if st["cnt"] >= int(st["num_steps"] * st["retention_ratio"]):
            lane = st["cnt"] % 2
            st["accumulated_ratio"][lane] *= st["mag_ratios"][st["cnt"]]
            st["accumulated_steps"][lane] += 1
            st["accumulated_err"][lane] += abs(1 - st["accumulated_ratio"][lane])
            residual = st["residual_cache"][lane]
            if (
                residual is not None
                and st["accumulated_err"][lane] < st["thresh"]
                and st["accumulated_steps"][lane] <= st["K"]
            ):
                skip = True
            else:
                st["accumulated_err"][lane] = 0.0
                st["accumulated_steps"][lane] = 0
                st["accumulated_ratio"][lane] = 1.0
        return skip, residual

    def _mag_advance(self) -> None:
        st = self._mag
        assert st is not None
        st["cnt"] += 2 if st["no_cfg"] else 1
        if st["cnt"] >= st["num_steps"]:
            st["last_generation_stats"] = {
                "computed_forwards": st["computed_forwards"],
                "skipped_forwards": st["skipped_forwards"],
                "total_forwards": st["computed_forwards"] + st["skipped_forwards"],
            }
            st["cnt"] = 0
            st["accumulated_ratio"] = [1.0, 1.0]
            st["accumulated_err"] = [0.0, 0.0]
            st["accumulated_steps"] = [0, 0]
            st["residual_cache"] = [None, None]
            st["computed_forwards"] = 0
            st["skipped_forwards"] = 0

    def _magcache_forward(
        self,
        x_video: Tensor | None,
        x_audio: Tensor | None,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        audio_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ):
        st = self._mag
        assert st is not None
        dit = self.module

        # Audio-only: MagCache does not skip (K5 parity); no counter advance.
        if x_video is None:
            return dit(
                x_video=None,
                x_audio=x_audio,
                text_embed=text_embed,
                pooled_text_embed=pooled_text_embed,
                time=time,
                visual_rope=visual_rope,
                audio_rope=audio_rope,
                text_rope=text_rope,
                attention_mask=attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        both = x_audio is not None and bool(getattr(dit, "is_multimodal", False))
        if both:
            return self._mag_forward_fused(
                x_video,
                x_audio,
                text_embed,
                pooled_text_embed,
                time,
                visual_rope,
                audio_rope,
                text_rope,
                attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )
        return self._mag_forward_video(
            x_video,
            text_embed,
            pooled_text_embed,
            time,
            visual_rope,
            text_rope,
            attention_mask,
            visual_token_type_ids=visual_token_type_ids,
        )

    def _mag_forward_fused(
        self,
        x_video: Tensor,
        x_audio: Tensor,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        audio_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ):
        """T2VA: skip text_blocks + visual_blocks; OutLayer via ``_time_only``."""
        dit = self.module
        st = self._mag
        assert st is not None
        attn_mask = dit._normalize_attn_mask(attention_mask) if hasattr(dit, "_normalize_attn_mask") else attention_mask

        te_v, pe_v = (
            (text_embed[0], pooled_text_embed[0]) if isinstance(text_embed, list) else (text_embed, pooled_text_embed)
        )
        te_a, pe_a = (
            (text_embed[1], pooled_text_embed[1]) if isinstance(text_embed, list) else (text_embed, pooled_text_embed)
        )
        if isinstance(text_rope, list):
            rope_v, rope_a = text_rope[0], text_rope[1]
        else:
            rope_v = rope_a = text_rope
        t_v, t_a = (time[0], time[1]) if isinstance(time, list) else (time, time)

        vis_embed, vis_shape, vis_rope = dit._embed_visual(
            x_video,
            visual_rope,
            visual_token_type_ids=visual_token_type_ids,
        )
        aud_embed, aud_rope = dit._embed_audio(x_audio, audio_rope)
        ori_v, ori_a = vis_embed, aud_embed

        skip, residual = self._mag_should_skip()
        if skip:
            st["skipped_forwards"] += 1
            video_tm = dit._time_only("video", pe_v, t_v)
            audio_tm = dit._time_only("audio", pe_a, t_a)
            self._mag_advance()
            return dit._project_fused(
                ori_v + residual[0],
                ori_a + residual[1],
                vis_shape,
                video_tm,
                audio_tm,
            )

        st["computed_forwards"] += 1
        video_te, video_tm = dit._encode_text("video", te_v, pe_v, t_v, rope_v, attn_mask)
        audio_te, audio_tm = dit._encode_text("audio", te_a, pe_a, t_a, rope_a, attn_mask)
        vis_embed, aud_embed = dit._run_visual_blocks_fused(
            vis_embed,
            aud_embed,
            video_te,
            audio_te,
            video_tm,
            audio_tm,
            vis_rope,
            aud_rope,
            attn_mask,
        )
        st["residual_cache"][st["cnt"] % 2] = (vis_embed - ori_v, aud_embed - ori_a)
        self._mag_advance()
        return dit._project_fused(vis_embed, aud_embed, vis_shape, video_tm, audio_tm)

    def _mag_forward_video(
        self,
        x_video: Tensor,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ):
        """T2V / multimodal video-only: skip text_blocks + visual_blocks on MagCache hit."""
        dit = self.module
        st = self._mag
        assert st is not None
        attn_mask = dit._normalize_attn_mask(attention_mask) if hasattr(dit, "_normalize_attn_mask") else attention_mask

        te_in = text_embed[0] if isinstance(text_embed, list) else text_embed
        pe_in = pooled_text_embed[0] if isinstance(pooled_text_embed, list) else pooled_text_embed
        rope_in = text_rope[0] if isinstance(text_rope, list) else text_rope
        t_in = time[0] if isinstance(time, list) else time
        prefix = "video" if getattr(dit, "is_multimodal", False) else None
        if getattr(dit, "is_multimodal", False) and isinstance(text_rope, list):
            rope_in = text_rope[0]

        vis_embed, vis_shape, vis_rope = dit._embed_visual(
            x_video,
            visual_rope,
            visual_token_type_ids=visual_token_type_ids,
        )
        ori = vis_embed

        skip, residual = self._mag_should_skip()
        if skip:
            st["skipped_forwards"] += 1
            tm = dit._time_only(prefix, pe_in, t_in)
            self._mag_advance()
            return dit._project_video(ori + residual, vis_shape, tm)

        st["computed_forwards"] += 1
        if prefix is not None:
            te, tm = dit._encode_text(prefix, te_in, pe_in, t_in, rope_in, attn_mask)
        else:
            te, tm = dit._encode_t2v(te_in, pe_in, t_in, rope_in, attn_mask)
        vis_embed, _ = dit._run_visual_blocks_single(
            vis_embed,
            None,
            te,
            tm,
            vis_rope,
            None,
            attn_mask,
        )
        st["residual_cache"][st["cnt"] % 2] = vis_embed - ori
        self._mag_advance()
        return dit._project_video(vis_embed, vis_shape, tm)
