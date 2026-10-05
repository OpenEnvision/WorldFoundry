"""Standard video synthesis adapter for native Zing inference."""

from worldfoundry.synthesis.visual_generation.runtime_video_synthesis import RuntimeVideoSynthesis

from .runtime import ZingRuntime


class ZingSynthesis(RuntimeVideoSynthesis):
    MODEL_NAME = "zing"
    GENERATION_TYPE = "ti2v"
    RUNTIME_CLS = ZingRuntime
    PRIMARY_PATH_KEY = "checkpoint_dir"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/zing/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = "zing"

    def _prediction_runtime_overrides(self, kwargs, *, fps):
        aliases = {"frames": "num_frames", "frame_num": "num_frames"}
        allowed = {"num_frames", "height", "width", "seed", "controls", "local_attn_size", "sink_size"}
        unknown = {aliases.get(key, key) for key in kwargs} - allowed - {"save_path", "execute"}
        if unknown:
            raise TypeError(f"Unknown Zing inference options: {sorted(unknown)}")
        result = {
            aliases.get(key, key): value
            for key, value in kwargs.items()
            if aliases.get(key, key) in allowed and value is not None
        }
        if fps is not None:
            result["fps"] = fps
        return result

    def close(self):
        if self.generator is not None:
            self.generator.close()
        self.generator = None
