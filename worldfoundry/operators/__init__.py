"""Public operator namespace.

Operator implementations can depend on optional model stacks, so this module
keeps the public surface explicit while loading concrete classes on demand.
"""

from importlib import import_module

_OPERATOR_MODULES = {
    "AC3DOperator": "ac3d_operator",
    "ACTOperator": "act_operator",
    "AnimateDiffOperator": "animatediff_operator",
    "AstraOperator": "astra_operator",
    "BaseOperator": "base_operator",
    "BeingH05Operator": "being_h05_operator",
    "CausalForcingOperator": "forcing_operator",
    "CUT3ROperator": "cut3r_operator",
    "CameraCtrlOperator": "cameractrl_operator",
    "CosmosPredict2p5Operator": "cosmos_predict2p5_operator",
    "CogACTOperator": "vla_native_operator",
    "DepthAnything3Operator": "depth_anything_v3_operator",
    "DepthAnythingOperator": "depth_anything_operator",
    "DiffusionPolicyOperator": "diffusion_policy_operator",
    "DBCogACTOperator": "vla_native_operator",
    "DreamDojoOperator": "dreamdojo_operator",
    "DreamZeroOperator": "dreamzero_operator",
    "DualCamCtrlOperator": "dualcamctrl_operator",
    "DVLTOperator": "dvlt_operator",
    "EmbodiedActionOperator": "embodied_action_operator",
    "FantasyWorldOperator": "fantasy_world_operator",
    "FlashWorldOperator": "flash_world_operator",
    "GigaBrain0Operator": "giga_brain_0_operator",
    "GR00TOperator": "gr00t_operator",
    "Gen3COperator": "gen3c_operator",
    "GeometryPriorOperator": "geometry_prior_operator",
    "Hailuo2p3Operator": "hailuo_2p3_operator",
    "HunyuanGameCraftOperator": "hunyuan_game_craft_operator",
    "HunyuanMirrorOperator": "hunyuan_mirror_operator",
    "HunyuanWorldPlayOperator": "hunyuan_worldplay_operator",
    "HunyuanWorldVoyagerOperator": "hunyuan_world_voyager_operator",
    "IRASimOperator": "irasim_operator",
    "InfiniteVGGTOperator": "infinite_vggt_operator",
    "InfiniteWorldOperator": "infinite_world_operator",
    "KairosOperator": "kairos_operator",
    "KlingApiOperator": "kling_api_operator",
    "LAPAOperator": "lapa_operator",
    "LingBotOperator": "lingbot_world_operator",
    "LingBotMapOperator": "lingbot_map_operator",
    "LingBotVAOperator": "lingbot_va_operator",
    "LongVieOperator": "longvie_operator",
    "LumaRay2Operator": "luma_ray2_operator",
    "Lyra1Operator": "lyra1_operator",
    "LyraOperator": "lyra_operator",
    "MatrixGame2Operator": "matrix_game_2_operator",
    "MatrixGame3Operator": "matrix_game_3_operator",
    "MMEVLAOperator": "vla_native_operator",
    "MoVerseOperator": "moverse_operator",
    "MolmoAct2Operator": "molmoact2_operator",
    "MolmoBotOperator": "vla_native_operator",
    "MotionCtrlOperator": "motionctrl_operator",
    "NeoVerseOperator": "neoverse_operator",
    "OctoOperator": "octo_operator",
    "OfficialPolicyOperator": "official_policy_operator",
    "OpenMAGVIT2Operator": "open_magvit2_operator",
    "OpenPIOperator": "openpi_operator",
    "OpenVLAOFTOperator": "vla_native_operator",
    "OpenVLAOperator": "openvla_operator",
    "PandoraOperator": "pandora_operator",
    "Pi3Operator": "pi3_operator",
    "PixelSplatOperator": "pixelsplat_operator",
    "RT1Operator": "rt1_operator",
    "ReCamMasterOperator": "recammaster_operator",
    "RoboFlamingoOperator": "roboflamingo_operator",
    "RollingForcingOperator": "forcing_operator",
    "RunwayGen4p5Operator": "runway_gen4p5_operator",
    "RuntimeVideoOperator": "runtime_video_operator",
    "ShowOOperator": "show_o_operator",
    "Sora2Operator": "sora2_operator",
    "SelfForcingOperator": "forcing_operator",
    "Splatt3ROperator": "splatt3r_operator",
    "StarVLAOperator": "starvla_operator",
    "StepVideoT2VOperator": "step_video_t2v_operator",
    "ThreeDFourDRuntimeOperator": "three_d_four_d_runtime_operator",
    "Uni3COperator": "uni3c_operator",
    "VGGTOperator": "vggt_operator",
    "VGGTOmegaOperator": "vggt_omega_operator",
    "VLANeXtOperator": "vla_native_operator",
    "VMemOperator": "vmem_operator",
    "Veo3Operator": "veo3_operator",
    "Wan2p2Operator": "wan_2p2_operator",
    "Wan2p5Operator": "wan_2p5_operator",
    "Wan2p6Operator": "wan_2p6_operator",
    "Wan2p7Operator": "wan_2p7_operator",
    "WoWOperator": "wow_operator",
    "WorldCamOperator": "worldcam_operator",
    "WorldFMOperator": "worldfm_operator",
    "WorldLabsOperator": "worldlabs_operator",
    "WorldModelRuntimeOperator": "world_model_runtime_operator",
    "XVLAOperator": "xvla_operator",
    "Yume1p5Operator": "yume_1p5_operator",
    "YumeOperator": "yume_operator",
    "ZeroScopeOperator": "zeroscope_operator",
}

__all__ = sorted(_OPERATOR_MODULES)


def __getattr__(name):
    """Getattr   implementation."""
    if name not in _OPERATOR_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{_OPERATOR_MODULES[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    """Dir   implementation."""
    return sorted([*globals(), *__all__])
