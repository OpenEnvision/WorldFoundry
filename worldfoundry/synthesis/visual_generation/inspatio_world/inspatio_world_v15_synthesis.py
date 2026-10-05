"""InSpatio World 1.5 synthesis using shared native Wan and DA3 models."""

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .v15_runtime import DEFAULT_CHECKPOINT_REPO, InspatioWorldV15Runtime


class InspatioWorldV15Synthesis(BaseSynthesis):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, pretrained_model_path=DEFAULT_CHECKPOINT_REPO, args=None, device=None, **kwargs):
        return cls(InspatioWorldV15Runtime(pretrained_model_path, device=device or "cuda", **kwargs))

    def plan(self):
        return self.runtime.plan()

    def predict(self, **kwargs):
        return self.runtime.predict(**kwargs)
