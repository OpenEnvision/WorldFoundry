import importlib
import sys


def test_renderer_model_import_does_not_load_planner_flash_attention():
    planner_module = (
        "worldfoundry.synthesis.visual_generation.bernini.inference.models."
        "modeling_qwen2_5_vl"
    )
    sys.modules.pop(planner_module, None)

    models = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.bernini.inference.models"
    )

    assert models.BerniniRendererModel is not None
    assert planner_module not in sys.modules
