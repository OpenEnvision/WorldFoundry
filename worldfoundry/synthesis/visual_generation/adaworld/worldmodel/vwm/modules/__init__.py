from .encoders.modules import GeneralConditioner as GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
