"""Dataset-indexed SolarWM inference through its official runtime."""

from worldfoundry.pipelines.official_world import OfficialWorldPipeline
from worldfoundry.synthesis.visual_generation.solarwm import SolarWMRuntime


class SolarWMPipeline(OfficialWorldPipeline):
    MODEL_ID = "solarwm"
    RUNTIME_CLS = SolarWMRuntime

    VARIANT_ALIASES = {
        "solarwm-wan2.2-5b": "wan-5b-dmd",
        "solarwm-wan22-5b": "wan-5b-dmd",
        "solarwm-wan2.2-14b": "wan-14b",
        "solarwm-ltx-22b": "ltx",
        "solarwm-h3-33b": "h3-bidirectional",
    }

    @classmethod
    def from_pretrained(cls, model_path=None, required_components=None, device="cuda", **kwargs):
        from collections.abc import Mapping

        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        selected = options.get("variant_id") or options.get("model_id")
        if selected in cls.VARIANT_ALIASES and "variant" not in options:
            kwargs["variant"] = cls.VARIANT_ALIASES[selected]
        return super().from_pretrained(model_path, required_components, device=device, **kwargs)
