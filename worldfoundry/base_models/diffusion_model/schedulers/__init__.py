"""WorldFoundry canonical numerical schedulers.

Each export implements (or adapts to) the :class:`~..contracts.DiffusionScheduler`
Protocol used by :class:`~..runners.base.NativeDiffusionRunner`:

- ``schedule(SamplingConfig) -> Sequence[SchedulerStep]`` must return exactly
  ``num_inference_steps`` steps or the runner rejects the run.
- ``scale_model_input`` / ``step`` consume ``DenoiserOutput.sample`` and
  produce the next latents.

Family modules live here so recipes do not import Diffusers.  Recipes
bind a ``build_*_scheduler`` factory; they do not construct scheduler
objects.  Symbols are lazy-imported via :data:`_EXPORTS`; an unknown
name raises :exc:`AttributeError`.

Typical recipe pairing
----------------------
- Wan / SkyReels / Echo / Cosmos 2.5 / Gamma-World: UniPC or Euler
  flow-match (optional Karras sigmas, family ``shift``).
- LTX: fixed-sigma Euler — one instance per refinement stage.
- Sana: flow-match, Flow-DPM, SCM (Sprint), or streaming Euler.
- HunyuanVideo: flow-match Euler with family shift.
- Cosmos 1 / 2: Karras x0 Euler / AB2 (EDM-style, not flow).
- Cosmos 3: Flow UniPC from the Hub scheduler config.
- Step-Video: family flow scheduler (``time_shift``).
- T2V-Turbo: LCM (few-step distilled guidance).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Cosmos3FlowUniPCScheduler": ".cosmos3",
    "FlowDPMSolverMultistepScheduler": ".flow_dpm",
    "FlowMatchEulerScheduler": ".flow_match",
    "FlowUniPCMultistepScheduler": ".flow_unipc",
    "FastVideoCausalWanSelfForcingScheduler": ".wan",
    "HunyuanVideoFlowMatchEulerScheduler": ".hunyuan_video",
    "InferenceFlowMatchScheduler": ".wan",
    "KarrasX0AB2Scheduler": ".karras_x0",
    "KarrasX0EulerScheduler": ".karras_x0",
    "LTXFixedEulerScheduler": ".ltx",
    "SanaSCMScheduler": ".sana",
    "SanaFlowDPMScheduler": ".sana",
    "SanaLongLiveScheduler": ".sana",
    "SanaStreamingEulerScheduler": ".sana",
    "SanaWMStreamingEulerScheduler": ".sana",
    "StepVideoFlowScheduler": ".step_video",
    "T2VTurboLCMScheduler": ".t2v_turbo",
    "WanFlowMatchEulerScheduler": ".wan",
    "WanFlowUniPCScheduler": ".wan",
    "build_cosmos3_flow_unipc_scheduler": ".cosmos3",
    "build_fastvideo_causal_wan_self_forcing_scheduler": ".wan",
    "build_hunyuan_video_flow_match_scheduler": ".hunyuan_video",
    "build_karras_x0_ab2_scheduler": ".karras_x0",
    "build_karras_x0_euler_scheduler": ".karras_x0",
    "build_ltx_fixed_euler_scheduler": ".ltx",
    "build_sana_flow_match_scheduler": ".sana",
    "build_sana_flow_dpm_scheduler": ".sana",
    "build_sana_longlive_scheduler": ".sana",
    "build_sana_scm_scheduler": ".sana",
    "build_sana_streaming_euler_scheduler": ".sana",
    "build_sana_wm_streaming_euler_scheduler": ".sana",
    "build_step_video_flow_scheduler": ".step_video",
    "build_t2v_turbo_lcm_scheduler": ".t2v_turbo",
    "build_wan_flow_match_euler_scheduler": ".wan",
    "build_wan_flow_unipc_scheduler": ".wan",
    "build_wan_sigmas": ".wan",
    "get_sampling_sigmas": ".flow_dpm",
    "retrieve_timesteps": ".flow_dpm",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one scheduler symbol from :data:`_EXPORTS` and cache it in ``globals()``."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
