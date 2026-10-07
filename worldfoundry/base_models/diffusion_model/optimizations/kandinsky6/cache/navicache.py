"""NaviCache: Kalman residual skip for the multimodal video+audio DiT."""

from __future__ import annotations

import torch
from torch import Tensor

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ConfigModel


class NaviCacheConfig(ConfigModel):
    thresh: float = 0.05
    align_steps: int = 10
    process_noise: float = 0.05
    measurement_noise: float = 0.05


class NaviCache:
    """NaviCache steps used by :class:`CacheDiT`."""

    def _reset_navi_state(self) -> None:
        if self._navi is None:
            return
        self._navi.update(
            {
                "forward_count": 0,
                "accumulated_error": 0.0,
                "should_compute_pair": True,
                "prediction_ratio": None,
                "state_ratio": None,
                "uncertainty": 1.0,
                "previous_raw_cond_input": None,
                "previous_raw_cond_output": None,
                "previous_raw_uncond_output": None,
                "prior_raw_cond_input": None,
                "cond_residual": None,
                "uncond_residual": None,
                "computed_forwards": 0,
                "skipped_forwards": 0,
                "last_generation_stats": None,
            }
        )

    def _navi_advance(self) -> None:
        st = self._navi
        assert st is not None
        st["forward_count"] += 2 if st["no_cfg"] else 1
        if st["forward_count"] >= st["num_forwards"]:
            st["last_generation_stats"] = {
                "computed_forwards": st["computed_forwards"],
                "skipped_forwards": st["skipped_forwards"],
                "total_forwards": st["computed_forwards"] + st["skipped_forwards"],
            }
            self._reset_navi_state()

    def _navicache_forward(
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
        if x_video is None or x_audio is None:
            raise RuntimeError("NaviCache only supports the multimodal (video+audio) input path.")

        dit = self.module
        st = self._navi
        assert st is not None

        raw_input = [
            x_video[..., : dit.in_visual_dim].clone(),
            x_audio.clone(),
        ]
        is_cond = st["forward_count"] % 2 == 0

        if is_cond:
            if st["forward_count"] < st["align_forwards"] or st["forward_count"] >= st["cutoff_forwards"]:
                should_compute = True
                st["accumulated_error"] = 0.0
            elif st["previous_raw_cond_input"] is not None and st["previous_raw_cond_output"] is not None:
                raw_input_change = (
                    torch.cat(
                        [(u - v).flatten() for u, v in zip(raw_input, st["previous_raw_cond_input"], strict=True)]
                    )
                    .abs()
                    .mean()
                )

                if st["state_ratio"] is not None:
                    st["prediction_ratio"] = st["state_ratio"]

                if st["prediction_ratio"] is not None:
                    output_norm = torch.cat([u.flatten() for u in st["previous_raw_cond_output"]]).abs().mean()
                    pred_change = st["prediction_ratio"] * (raw_input_change / (output_norm + 1e-8))
                    st["accumulated_error"] += pred_change
                    if st["accumulated_error"] < st["thresh"]:
                        should_compute = False
                    else:
                        should_compute = True
                        st["accumulated_error"] = 0.0
                else:
                    should_compute = True
            else:
                should_compute = True

            st["should_compute_pair"] = should_compute
            st["previous_raw_cond_input"] = [u.clone() for u in raw_input]

        if is_cond and not st["should_compute_pair"] and st["cond_residual"] is not None:
            st["skipped_forwards"] += 1
            self._navi_advance()
            return (
                raw_input[0] + st["cond_residual"][0],
                raw_input[1] + st["cond_residual"][1],
            )

        if (not is_cond) and (not st["should_compute_pair"]) and st["uncond_residual"] is not None:
            st["skipped_forwards"] += 1
            self._navi_advance()
            return (
                raw_input[0] + st["uncond_residual"][0],
                raw_input[1] + st["uncond_residual"][1],
            )

        st["computed_forwards"] += 1
        output = dit(
            x_video=x_video,
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
        # Normalize to tuple of tensors
        if not isinstance(output, tuple):
            raise RuntimeError("NaviCache expects multimodal (video, audio) DiT output")
        out_list = list(output)

        if is_cond:
            if st["previous_raw_cond_output"] is not None:
                output_change = (
                    torch.cat(
                        [(u - v).flatten() for u, v in zip(out_list, st["previous_raw_cond_output"], strict=True)]
                    )
                    .abs()
                    .mean()
                )
                if st["prior_raw_cond_input"] is not None:
                    input_change = (
                        torch.cat(
                            [
                                (u - v).flatten()
                                for u, v in zip(st["previous_raw_cond_input"], st["prior_raw_cond_input"], strict=True)
                            ]
                        )
                        .abs()
                        .mean()
                    )
                    z = output_change / (input_change + 1e-8)
                    is_warmup = st["forward_count"] < st["align_forwards"]
                    if st["state_ratio"] is None or is_warmup:
                        st["state_ratio"] = z
                        st["uncertainty"] = 1.0
                    else:
                        st["uncertainty"] = st["uncertainty"] + st["process_noise"]
                        kalman_gain = st["uncertainty"] / (st["uncertainty"] + st["measurement_noise"] + 1e-8)
                        st["state_ratio"] = st["state_ratio"] + kalman_gain * (z - st["state_ratio"])
                        st["uncertainty"] = (1 - kalman_gain) * st["uncertainty"]
                    st["prediction_ratio"] = st["state_ratio"]

            st["prior_raw_cond_input"] = st["previous_raw_cond_input"]
            st["previous_raw_cond_output"] = [u.clone() for u in out_list]
            st["cond_residual"] = [u - v for u, v in zip(out_list, raw_input, strict=True)]
        else:
            st["previous_raw_uncond_output"] = [u.clone() for u in out_list]
            st["uncond_residual"] = [u - v for u, v in zip(out_list, raw_input, strict=True)]

        self._navi_advance()
        return output
