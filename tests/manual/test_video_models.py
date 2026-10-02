from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.pipelines.allegro.pipeline_allegro_ti2v import AllegroTi2VPipeline
from worldfoundry.pipelines.cogvideox.pipeline_cogvideox_5b_t2v import CogVideoX5bT2VPipeline
from worldfoundry.pipelines.dynamicrafter.pipeline_dynamicrafter_512_i2v import DynamiCrafter512I2VPipeline
from worldfoundry.pipelines.ltx2.pipeline_ltx2_3_i2v import LTX23I2VPipeline
from worldfoundry.pipelines.ltx2.pipeline_ltx2_i2v import LTX2I2VPipeline
from worldfoundry.pipelines.runway.pipeline_gen_3_i2v import Gen3I2VPipeline
from worldfoundry.pipelines.videocrafter.pipeline_videocrafter1_i2v import VideoCrafter1I2VPipeline
from worldfoundry.pipelines.wan.pipeline_wan_2p1_i2v import Wan2p1I2VPipeline
from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline
from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_i2v_synthesis import EasyAnimateI2VSynthesis
from worldfoundry.synthesis.visual_generation.ltx2.ltx2_3_i2v_synthesis import LTX23I2VSynthesis
from worldfoundry.synthesis.visual_generation.videocrafter.videocrafter1_i2v_synthesis import VideoCrafter1I2VSynthesis
from worldfoundry.synthesis.visual_generation.wan.wan_2p1_t2v_synthesis import Wan2p1T2VSynthesis

hy_world_entry = load_model_zoo_registry().get("hy-world-2.0")
assert hy_world_entry.pipeline_target == "worldfoundry.pipelines.hunyuan_world.pipeline_hy_world_2p0:HYWorld2Pipeline"
assert "hyworld2.0" in hy_world_entry.aliases
assert VideoCrafter1I2VSynthesis.GENERATION_TYPE == "i2v"
assert Wan2p1T2VSynthesis.GENERATION_TYPE == "t2v"
assert EasyAnimateI2VSynthesis.RUNTIME_CLS.__name__ == "EasyAnimate"
assert LTX23I2VSynthesis.default_runtime_kwargs()["version_hint"] == "2.3"

assert AllegroTi2VPipeline is not None
assert DynamiCrafter512I2VPipeline is not None
assert Gen3I2VPipeline is not None
assert LTX2I2VPipeline is not None
assert LTX23I2VPipeline is not None
assert Wan2p1I2VPipeline is not None
assert Wan2p1T2VPipeline is not None

pipe = CogVideoX5bT2VPipeline.from_pretrained(
    lazy=True,
)
assert pipe.generation_type == "t2v"
cog_overrides = pipe.synthesis_model._prediction_runtime_overrides(
    {
        "num_frames": 17,
        "num_inference_steps": 10,
        "guidance_scale": 5.0,
        "seed": 123,
        "height": 480,
        "width": 720,
    },
    fps=8,
)
assert cog_overrides["num_frames"] == 17
assert cog_overrides["num_inference_steps"] == 10
assert cog_overrides["guidance_scale"] == 5.0
assert cog_overrides["seed"] == 123
assert cog_overrides["height"] == 480
assert cog_overrides["width"] == 720
assert "fps" not in cog_overrides

pipe = VideoCrafter1I2VPipeline.from_pretrained(
    lazy=True,
)
assert pipe.generation_type == "i2v"

pipe = LTX2I2VPipeline.from_pretrained(
    lazy=True,
)
assert pipe.generation_type == "i2v"

pipe = LTX23I2VPipeline.from_pretrained(
    lazy=True,
)
assert pipe.generation_type == "i2v"

pipe = Wan2p1T2VPipeline.from_pretrained(
    model_path="cache/hfd/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a",
    lazy=True,
)
assert pipe.generation_type == "t2v"

print("individual video model pipelines and lazy synthesis smoke passed")
