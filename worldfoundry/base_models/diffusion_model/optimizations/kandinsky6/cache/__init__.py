"""DiT cache policy. Each strategy lives in its own module.

Never monkey-patches ``DiffusionTransformer3D.forward``. The bare module stays
exportable and compilable via ``CacheDiT.module``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ConfigModel
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.cache.magcache import MagCache, MagCacheConfig, prepare_mag_ratios, select_mag_ratios
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.cache.navicache import NaviCache, NaviCacheConfig

CacheMode = Literal["none", "magcache", "navicache"]
CacheModeName = CacheMode


class CacheConfig(ConfigModel):
    mode: CacheModeName = "none"
    magcache: MagCacheConfig | None = None
    navicache: NaviCacheConfig | None = None


class CacheDiT(MagCache, NaviCache, nn.Module):
    """Facade around a bare DiT with switchable MagCache / NaviCache / none."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.module = dit
        self.mode: CacheMode = "none"
        self._mag: dict | None = None
        self._navi: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_cache(
        self,
        mode: CacheMode | None,
        *,
        num_steps: int | None = None,
        no_cfg: bool = False,
        # MagCache
        mag_ratios: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        mode_name: str | None = None,
        thresh: float | None = None,
        K: int | None = None,  # noqa: N803  # MagCache paper hyperparameter name
        retention_ratio: float | None = None,
        # NaviCache
        align_steps: int | None = None,
        process_noise: float | None = None,
        measurement_noise: float | None = None,
    ) -> None:
        if mode is None:
            mode = "none"
        if mode not in ("none", "magcache", "navicache"):
            raise ValueError(f"Unknown cache mode: {mode!r}")

        if mode == "none":
            self.mode = "none"
            self._mag = None
            self._navi = None
            return

        if num_steps is None:
            raise ValueError(f"set_cache({mode!r}) requires num_steps")

        if mode == "magcache":
            if mag_ratios is None:
                raise ValueError("set_cache('magcache') requires mag_ratios")
            mag_ratios, mode_name = select_mag_ratios(mag_ratios, mode_name)
            self._navi = None
            self._mag = {
                "num_steps": num_steps * 2,
                "gen_mode": mode_name,
                "no_cfg": no_cfg,
                "thresh": 0.12 if thresh is None else thresh,
                "K": 2 if K is None else K,
                "retention_ratio": 0.2 if retention_ratio is None else retention_ratio,
                "mag_ratios": prepare_mag_ratios(mag_ratios, num_steps),
            }
            self.mode = "magcache"
            self._reset_mag_state()
            return

        # navicache
        if not getattr(self.module, "is_multimodal", False):
            raise NotImplementedError("NaviCache is only implemented for the multimodal (video+audio) T2VA DiT.")
        self._mag = None
        self._navi = {
            "num_forwards": num_steps * 2,
            "num_steps": num_steps,
            "no_cfg": no_cfg,
            "thresh": 0.05 if thresh is None else thresh,
            "align_forwards": (10 if align_steps is None else align_steps) * 2,
            "cutoff_forwards": num_steps * 2 - 2,
            "process_noise": 0.05 if process_noise is None else process_noise,
            "measurement_noise": 0.05 if measurement_noise is None else measurement_noise,
        }
        self.mode = "navicache"
        self._reset_navi_state()

    def reset_cache_state(self) -> None:
        if self.mode == "magcache":
            self._reset_mag_state()
        elif self.mode == "navicache":
            self._reset_navi_state()

    @property
    def last_generation_stats(self) -> dict | None:
        if self.mode == "magcache" and self._mag is not None:
            return self._mag.get("last_generation_stats")
        if self.mode == "navicache" and self._navi is not None:
            return self._navi.get("last_generation_stats")
        return None

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        if self.mode == "none":
            return self.module(*args, **kwargs)
        if self.mode == "magcache":
            return self._magcache_forward(*args, **kwargs)
        return self._navicache_forward(*args, **kwargs)

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ raises AttributeError for missing attrs;
        # forward attribute lookups to the wrapped DiT (visual_cond, patch_size, …).
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self._modules.get("module") if "_modules" in self.__dict__ else None
            if module is None:
                raise
            return getattr(module, name)


# CLAUDE.md naming alias
MagCacheDiT = CacheDiT

__all__ = (
    "CacheConfig",
    "CacheDiT",
    "CacheMode",
    "CacheModeName",
    "MagCacheConfig",
    "MagCacheDiT",
    "NaviCacheConfig",
    "prepare_mag_ratios",
    "select_mag_ratios",
)
