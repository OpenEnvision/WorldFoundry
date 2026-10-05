import inspect
import sys
from types import SimpleNamespace

from worldfoundry.pipelines.dynamicrafter.pipeline_dynamicrafter_512_i2v import (
    DynamiCrafter512I2VPipeline,
)
from worldfoundry.synthesis.visual_generation.dynamicrafter.worldfoundry_runtime import (
    DynamiCrafter,
    REQUIRED_IMPORTS,
    _seed_everything,
    load_model_checkpoint,
)
from worldfoundry.synthesis.visual_generation.runtime_video_synthesis import (
    RuntimeVideoSynthesis,
)


class _Model:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"weight": object()}

    def load_state_dict(self, state_dict, strict=True):
        self.loaded = (state_dict, strict)


def test_runtime_seeding_does_not_require_pytorch_lightning(monkeypatch):
    calls = []
    fake_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: calls.append(("numpy", value)))
    )
    fake_torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(("torch", value))
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "worldfoundry.synthesis.visual_generation.dynamicrafter.worldfoundry_runtime.random.seed",
        lambda value: calls.append(("python", value)),
    )

    assert "pytorch_lightning" not in REQUIRED_IMPORTS
    assert _seed_everything(123) == 123
    assert calls == [("python", 123), ("numpy", 123), ("torch", 123)]


def test_runtime_uses_workspace_device_instead_of_default_cuda_device():
    parameters = inspect.signature(DynamiCrafter).parameters
    source = inspect.getsource(DynamiCrafter)

    assert parameters["device"].default == "cuda"
    assert "model.cuda()" not in source
    assert '.to("cuda")' not in source


def test_checkpoint_load_is_weights_only_and_memory_mapped(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"state_dict": {"weight": object()}}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))
    model = _Model()

    assert load_model_checkpoint(model, "/checkpoints/model.ckpt") is model
    assert calls == [
        (
            "/checkpoints/model.ckpt",
            {"map_location": "cpu", "mmap": True, "weights_only": True},
        )
    ]
    assert model.loaded is not None
    assert model.loaded[1] is True


def test_studio_overrides_reach_dynamicrafter_constructor_names():
    class Runtime:
        def __init__(
            self,
            video_length=50,
            ddim_steps=50,
            unconditional_guidance_scale=7.5,
        ):
            pass

    wrapper = RuntimeVideoSynthesis(
        model_name="dynamicrafter_512_i2v",
        generation_type="i2v",
        runtime_cls=Runtime,
        runtime_kwargs={},
    )

    assert wrapper._prediction_runtime_overrides(
        {
            "num_frames": 9,
            "num_inference_steps": 4,
            "guidance_scale": 6.0,
        },
        fps=None,
    ) == {
        "video_length": 9,
        "ddim_steps": 4,
        "unconditional_guidance_scale": 6.0,
    }


def test_pipeline_forwards_quality_overrides_to_synthesis():
    class Operator:
        def get_interaction(self, prompt):
            self.prompt = prompt

        def process_interaction(self):
            return {"processed_prompt": self.prompt}

        def delete_last_interaction(self):
            pass

        def process_perception(self, *, images):
            return {"images": images}

    class Synthesis:
        generation_type = "i2v"
        model_name = "dynamicrafter_512_i2v"

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return {"video": "video"}

    synthesis = Synthesis()
    pipeline = DynamiCrafter512I2VPipeline(
        operator=Operator(),
        synthesis_model=synthesis,
        memory_module=object(),
    )

    assert pipeline(
        prompt="sparkler",
        images="image",
        num_frames=17,
        num_inference_steps=16,
        guidance_scale=7.5,
        seed=42,
    ) == "video"
    assert synthesis.kwargs["num_frames"] == 17
    assert synthesis.kwargs["num_inference_steps"] == 16
    assert synthesis.kwargs["guidance_scale"] == 7.5
    assert synthesis.kwargs["seed"] == 42
