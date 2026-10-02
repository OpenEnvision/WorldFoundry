from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.pipelines.hunyuan_world.pipeline_hy_world_2p0 import (
    HYWorld2PanoPipeline,
    HYWorld2Pipeline,
)


assert HYWorld2Pipeline.__name__ == "HYWorld2Pipeline"
assert HYWorld2PanoPipeline.__name__ == "HYWorld2PanoPipeline"

entry = load_model_zoo_registry().get("hy-world-2.0")
assert entry.pipeline_target == "worldfoundry.pipelines.hunyuan_world.pipeline_hy_world_2p0:HYWorld2Pipeline"
assert {
    "hyworld-2.0",
    "hy-world2.0",
    "hyworld2.0",
}.issubset(set(entry.aliases))

print("HY-World-2.0 loader registry validation passed")
