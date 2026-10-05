"""Lazy target and component contract pipeline definitions."""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

from .pipeline_utils import PipelineABC


def _load_target(target: str) -> type[Any]:
    """Load target helper function."""
    module_name, class_name = target.split(":", 1)
    # Dynamically load target module to prevent eager-import overhead
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class _ComponentPipelineMeta(type):
    """Metaclass for ComponentPipeline that dynamically resolves and loads target components on access."""
    _COMPONENT_ATTRIBUTES = {"OPERATOR_CLS", "MEMORY_CLS", "SYNTHESIS_CLS"}
    _TARGET_ATTRIBUTES = {
        "OPERATOR_CLS": "OPERATOR_TARGET",
        "MEMORY_CLS": "MEMORY_TARGET",
        "SYNTHESIS_CLS": "SYNTHESIS_TARGET",
    }

    def __getattribute__(cls, name: str) -> Any:
        """Getattribute for _ComponentPipelineMeta."""
        value = type.__getattribute__(cls, name)
        if name in _ComponentPipelineMeta._COMPONENT_ATTRIBUTES and value is None:
            target_attribute = _ComponentPipelineMeta._TARGET_ATTRIBUTES[name]
            target = type.__getattribute__(cls, target_attribute)
            if target is not None:
                value = _load_target(target)
                setattr(cls, name, value)
        return value


class ComponentPipeline(PipelineABC, metaclass=_ComponentPipelineMeta):
    """PipelineABC component-contract implementation backed by lazy targets."""

    OPERATOR_TARGET: ClassVar[str | None] = None
    MEMORY_TARGET: ClassVar[str | None] = None
    SYNTHESIS_TARGET: ClassVar[str | None] = None
    _TARGET_ATTRIBUTES: ClassVar[dict[str, str]] = {
        "OPERATOR_CLS": "OPERATOR_TARGET",
        "MEMORY_CLS": "MEMORY_TARGET",
        "SYNTHESIS_CLS": "SYNTHESIS_TARGET",
    }

    @classmethod
    def _uses_component_contract(cls) -> bool:
        """Determine whether the pipeline class implements the component contract."""
        return all(
            cls.__dict__.get(attribute_name) is not None
            or cls.__dict__.get(cls._TARGET_ATTRIBUTES[attribute_name]) is not None
            for attribute_name in cls._TARGET_ATTRIBUTES
        )

    @classmethod
    def _component(cls, attribute_name: str) -> type[Any]:
        """Retrieve a pipeline component class by attribute name."""
        component = cls.__dict__.get(attribute_name)
        if component is not None:
            return component
        target_attribute = cls._TARGET_ATTRIBUTES.get(attribute_name)
        target = cls.__dict__.get(target_attribute) if target_attribute is not None else None
        if target is None:
            raise RuntimeError(f"{cls.__name__} must define {attribute_name} or {target_attribute}.")
        component = _load_target(target)
        setattr(cls, attribute_name, component)
        return component

    @classmethod
    def _create_memory_module_if_available(cls) -> Any:
        """Instantiate and return the memory module if specified."""
        if cls.__dict__.get("MEMORY_CLS") is None and cls.__dict__.get("MEMORY_TARGET") is None:
            return None
        return cls._component("MEMORY_CLS")(model_id=cls.MODEL_ID)


def _component_pipeline_class(
    name: str,
    *,
    doc: str,
    model_id: str,
    operator_target: str,
    memory_target: str,
    synthesis_target: str,
    generation_type: str | None = None,
    model_path_option: str = "repo_root",
) -> type[ComponentPipeline]:
    """Component pipeline class helper function."""
    attributes: dict[str, Any] = {
        "__doc__": doc,
        "__module__": __name__,
        "MODEL_ID": model_id,
        "OPERATOR_TARGET": operator_target,
        "MEMORY_TARGET": memory_target,
        "SYNTHESIS_TARGET": synthesis_target,
        "MODEL_PATH_OPTION": model_path_option,
    }
    if generation_type is not None:
        attributes["generation_type"] = generation_type
    return type(name, (ComponentPipeline,), attributes)


ACTPipeline = _component_pipeline_class(
    "ACTPipeline",
    doc="WorldFoundry action-chunking policy pipeline for ACT controllers.",
    model_id="act",
    operator_target="worldfoundry.operators.act_operator:ACTOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:ACTMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.act.act_synthesis:ACTSynthesis",
    generation_type="action_chunking_policy",
    model_path_option="checkpoint_path",
)
BeingH05Pipeline = _component_pipeline_class(
    "BeingH05Pipeline",
    doc="WorldFoundry VLA policy pipeline for Being-H0.5.",
    model_id="being-h05",
    operator_target="worldfoundry.operators.being_h05_operator:BeingH05Operator",
    memory_target="worldfoundry.synthesis.action_generation.memory:BeingH05Memory",
    synthesis_target="worldfoundry.synthesis.action_generation.being_h05.being_h05_synthesis:BeingH05Synthesis",
    generation_type="vla_policy",
)
BeingH05Pipeline.MODEL_PATH_OPTION = "checkpoint_dir"
DiffusionPolicyPipeline = _component_pipeline_class(
    "DiffusionPolicyPipeline",
    doc="WorldFoundry visuomotor policy pipeline for action-diffusion controllers.",
    model_id="diffusion-policy",
    operator_target="worldfoundry.operators.diffusion_policy_operator:DiffusionPolicyOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:DiffusionPolicyMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.diffusion_policy.diffusion_policy_synthesis:DiffusionPolicySynthesis",
    generation_type="visuomotor_policy",
)
DiffusionPolicyPipeline.MODEL_PATH_OPTION = "checkpoint_path"
DreamZeroPipeline = _component_pipeline_class(
    "DreamZeroPipeline",
    doc="WorldFoundry WAM/embodied-action pipeline for NVIDIA DreamZero.",
    model_id="dreamzero",
    operator_target="worldfoundry.operators.dreamzero_operator:DreamZeroOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:DreamZeroMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.dreamzero.dreamzero_synthesis:DreamZeroSynthesis",
    generation_type="world_action_model",
)
DreamZeroPipeline.MODEL_PATH_OPTION = "checkpoint_dir"
GigaBrain0Pipeline = _component_pipeline_class(
    "GigaBrain0Pipeline",
    doc="WorldFoundry VLA policy pipeline for GigaBrain-0 action generation.",
    model_id="giga-brain-0",
    operator_target="worldfoundry.operators.giga_brain_0_operator:GigaBrain0Operator",
    memory_target="worldfoundry.synthesis.action_generation.memory:GigaBrain0Memory",
    synthesis_target="worldfoundry.synthesis.action_generation.giga_brain_0.giga_brain_0_synthesis:GigaBrain0Synthesis",
    generation_type="vla_policy",
)
GigaBrain0Pipeline.MODEL_PATH_OPTION = "model_path"
GR00TPipeline = _component_pipeline_class(
    "GR00TPipeline",
    doc="WorldFoundry VLA policy pipeline for NVIDIA GR00T-style action generation.",
    model_id="gr00t",
    operator_target="worldfoundry.operators.gr00t_operator:GR00TOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:GR00TMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.gr00t.gr00t_synthesis:GR00TSynthesis",
    generation_type="vla_policy",
)
GR00TPipeline.MODEL_PATH_OPTION = "checkpoint_dir"
LAPAPipeline = _component_pipeline_class(
    "LAPAPipeline",
    doc="WorldFoundry VA/VAM pipeline for latent-action generation.",
    model_id="lapa",
    operator_target="worldfoundry.operators.lapa_operator:LAPAOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:LAPAMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.lapa.lapa_synthesis:LAPASynthesis",
    generation_type="visual_action_model",
)
LAPAPipeline.MODEL_PATH_OPTION = "checkpoint_dir"
LingBotVAPipeline = _component_pipeline_class(
    "LingBotVAPipeline",
    doc="WorldFoundry embodied-action pipeline for LingBot-VA video-action models.",
    model_id="lingbot-va",
    operator_target="worldfoundry.operators.lingbot_va_operator:LingBotVAOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:LingBotVAMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.lingbot_va.lingbot_va_synthesis:LingBotVASynthesis",
    generation_type="embodied_action",
)
LingBotVAPipeline.MODEL_PATH_OPTION = "checkpoint_dir"
MolmoAct2Pipeline = _component_pipeline_class(
    "MolmoAct2Pipeline",
    doc="WorldFoundry VLA policy pipeline for MolmoAct2 action generation.",
    model_id="molmoact2",
    operator_target="worldfoundry.operators.molmoact2_operator:MolmoAct2Operator",
    memory_target="worldfoundry.synthesis.action_generation.memory:MolmoAct2Memory",
    synthesis_target="worldfoundry.synthesis.action_generation.molmoact2.molmoact2_synthesis:MolmoAct2Synthesis",
    generation_type="vla_policy",
)
MolmoAct2Pipeline.MODEL_PATH_OPTION = "checkpoint_dir"
OctoPipeline = _component_pipeline_class(
    "OctoPipeline",
    doc="WorldFoundry VLA/generalist policy pipeline for Octo action generation.",
    model_id="octo",
    operator_target="worldfoundry.operators.octo_operator:OctoOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:OctoMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.octo.octo_synthesis:OctoSynthesis",
    generation_type="vla_policy",
)
OctoPipeline.MODEL_PATH_OPTION = "checkpoint_dir"
OpenPIPipeline = _component_pipeline_class(
    "OpenPIPipeline",
    doc="WorldFoundry VLA policy pipeline for OpenPI/pi0-style action generation.",
    model_id="openpi",
    operator_target="worldfoundry.operators.openpi_operator:OpenPIOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:OpenPIMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.openpi.openpi_synthesis:OpenPISynthesis",
    generation_type="vla_policy",
)
OpenPIPipeline.MODEL_PATH_OPTION = "checkpoint_dir"
OpenVLAPipeline = _component_pipeline_class(
    "OpenVLAPipeline",
    doc="WorldFoundry VLA policy pipeline for OpenVLA-style action generation.",
    model_id="openvla",
    operator_target="worldfoundry.operators.openvla_operator:OpenVLAOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:OpenVLAMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.openvla.openvla_synthesis:OpenVLASynthesis",
    generation_type="vla_policy",
)
OpenVLAPipeline.MODEL_PATH_OPTION = "checkpoint_dir"
RoboFlamingoPipeline = _component_pipeline_class(
    "RoboFlamingoPipeline",
    doc="WorldFoundry VLA/VLM-policy pipeline for RoboFlamingo action generation.",
    model_id="roboflamingo",
    operator_target="worldfoundry.operators.roboflamingo_operator:RoboFlamingoOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:RoboFlamingoMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.roboflamingo.roboflamingo_synthesis:RoboFlamingoSynthesis",
    generation_type="vla_policy",
)
RoboFlamingoPipeline.MODEL_PATH_OPTION = "policy_checkpoint_path"
StarVLAPipeline = _component_pipeline_class(
    "StarVLAPipeline",
    doc="WorldFoundry embodied-action pipeline for StarVLA VLA and WAM variants.",
    model_id="starvla",
    operator_target="worldfoundry.operators.starvla_operator:StarVLAOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:StarVLAMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.starvla.starvla_synthesis:StarVLASynthesis",
    generation_type="embodied_action",
)
StarVLAPipeline.MODEL_PATH_OPTION = "checkpoint_dir"


def _official_policy_pipeline(name: str, *, model_id: str, doc: str, generation_type: str = "vla_policy") -> type[ComponentPipeline]:
    """Official policy pipeline helper function."""
    return _component_pipeline_class(
        name,
        doc=doc,
        model_id=model_id,
        operator_target="worldfoundry.operators.official_policy_operator:OfficialPolicyOperator",
        memory_target="worldfoundry.synthesis.action_generation.memory:ActionTraceMemory",
        synthesis_target="worldfoundry.synthesis.action_generation.official_policy:OfficialPolicySynthesis",
        generation_type=generation_type,
        model_path_option="checkpoint_path",
    )


def _independent_action_pipeline(
    name: str,
    *,
    model_id: str,
    doc: str,
    synthesis_target: str,
    operator_target: str = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator",
    generation_type: str = "vla_policy",
) -> type[ComponentPipeline]:
    """Action pipeline helper for model-specific in-tree synthesis modules."""
    return _component_pipeline_class(
        name,
        doc=doc,
        model_id=model_id,
        operator_target=operator_target,
        memory_target="worldfoundry.synthesis.action_generation.memory:ActionTraceMemory",
        synthesis_target=synthesis_target,
        generation_type=generation_type,
        model_path_option="checkpoint_path",
    )


BeingH07Pipeline = _official_policy_pipeline(
    "BeingH07Pipeline",
    model_id="being-h07",
    doc="WorldFoundry WAM/action pipeline for Being-H0.7.",
    generation_type="world_action_model",
)
CogACTPipeline = _independent_action_pipeline(
    "CogACTPipeline",
    model_id="cogact",
    doc="WorldFoundry VLA policy pipeline for CogACT action generation.",
    operator_target="worldfoundry.operators.vla_native_operator:CogACTOperator",
    synthesis_target="worldfoundry.synthesis.action_generation.cogact:CogACTSynthesis",
)
DBCogACTPipeline = _independent_action_pipeline(
    "DBCogACTPipeline",
    model_id="db-cogact",
    doc="WorldFoundry VLA policy pipeline for DB-CogACT/Dexbotic action generation.",
    operator_target="worldfoundry.operators.vla_native_operator:DBCogACTOperator",
    synthesis_target="worldfoundry.synthesis.action_generation.db_cogact:DBCogACTSynthesis",
)
FastWAMPipeline = _official_policy_pipeline(
    "FastWAMPipeline",
    model_id="fastwam",
    doc="WorldFoundry WAM policy pipeline for FastWAM.",
    generation_type="world_action_model",
)
LingBotVLAPipeline = _independent_action_pipeline(
    "LingBotVLAPipeline",
    model_id="lingbot-vla",
    doc="WorldFoundry VLA policy pipeline for LingBot-VLA v1 action generation.",
    synthesis_target="worldfoundry.synthesis.action_generation.lingbot_vla:LingBotVLASynthesis",
)
LingBotVLAV2Pipeline = _independent_action_pipeline(
    "LingBotVLAV2Pipeline",
    model_id="lingbot-vla-v2",
    doc="WorldFoundry VLA policy pipeline for LingBot-VLA v2 batched action generation.",
    synthesis_target="worldfoundry.synthesis.action_generation.lingbot_vla_v2:LingBotVLAV2Synthesis",
)
LastR1Pipeline = _component_pipeline_class(
    "LastR1Pipeline",
    model_id="last-r1",
    doc="WorldFoundry offline action-inference pipeline for LaST-R1.",
    operator_target="worldfoundry.operators.official_policy_operator:OfficialPolicyOperator",
    memory_target="worldfoundry.synthesis.action_generation.memory:ActionTraceMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.last_r1:LastR1Synthesis",
    generation_type="vla_policy",
    model_path_option="checkpoint_path",
)
MMEVLAPipeline = _independent_action_pipeline(
    "MMEVLAPipeline",
    model_id="mme-vla",
    doc="WorldFoundry VLA policy pipeline for MME-VLA RoboMME policies.",
    operator_target="worldfoundry.operators.vla_native_operator:MMEVLAOperator",
    synthesis_target="worldfoundry.synthesis.action_generation.mme_vla:MMEVLASynthesis",
)
MolmoBotPipeline = _independent_action_pipeline(
    "MolmoBotPipeline",
    model_id="molmobot",
    doc="WorldFoundry VLA policy pipeline for MolmoBot action generation.",
    operator_target="worldfoundry.operators.vla_native_operator:MolmoBotOperator",
    synthesis_target="worldfoundry.synthesis.action_generation.molmobot:MolmoBotSynthesis",
)
OpenVLAOFTPipeline = _independent_action_pipeline(
    "OpenVLAOFTPipeline",
    model_id="openvla-oft",
    doc="WorldFoundry VLA policy pipeline for OpenVLA-OFT action generation.",
    operator_target="worldfoundry.operators.vla_native_operator:OpenVLAOFTOperator",
    synthesis_target="worldfoundry.synthesis.action_generation.openvla_oft:OpenVLAOFTSynthesis",
)
OpenVLAOFTPipeline.MODEL_PATH_OPTION = "checkpoint_path"
SmolVLAPipeline = _official_policy_pipeline(
    "SmolVLAPipeline",
    model_id="smolvla",
    doc="WorldFoundry VLA policy pipeline for SmolVLA.",
)
SpiritV15Pipeline = _official_policy_pipeline(
    "SpiritV15Pipeline",
    model_id="spirit-v1.5",
    doc="WorldFoundry VLA policy pipeline for Spirit-VLA v1.5.",
)
VLANeXtPipeline = _independent_action_pipeline(
    "VLANeXtPipeline",
    model_id="vlanext",
    doc="WorldFoundry VLA policy pipeline for VLANeXt action generation.",
    operator_target="worldfoundry.operators.vla_native_operator:VLANeXtOperator",
    synthesis_target="worldfoundry.synthesis.action_generation.vlanext:VLANeXtSynthesis",
)
XVLAPipeline = _independent_action_pipeline(
    "XVLAPipeline",
    model_id="xvla",
    doc="WorldFoundry cross-embodiment VLA policy pipeline for X-VLA.",
    operator_target="worldfoundry.operators.xvla_operator:XVLAOperator",
    synthesis_target="worldfoundry.synthesis.action_generation.xvla:XVLASynthesis",
)
XVLAPipeline.MODEL_PATH_OPTION = "checkpoint_path"

AnimateDiffPipeline = _component_pipeline_class(
    "AnimateDiffPipeline",
    doc="WorldFoundry video-generation pipeline for AnimateDiff.",
    model_id="animatediff",
    operator_target="worldfoundry.operators.animatediff_operator:AnimateDiffOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.animatediff:AnimateDiffSynthesis",
)
CameraCtrlPipeline = _component_pipeline_class(
    "CameraCtrlPipeline",
    doc="WorldFoundry camera-control video pipeline for CameraCtrl.",
    model_id="cameractrl",
    operator_target="worldfoundry.operators.cameractrl_operator:CameraCtrlOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.cameractrl.synthesis:CameraCtrlSynthesis",
)
DualCamCtrlPipeline = _component_pipeline_class(
    "DualCamCtrlPipeline",
    doc="WorldFoundry image/depth/camera-control video pipeline for DualCamCtrl.",
    model_id="dualcamctrl",
    operator_target="worldfoundry.operators.dualcamctrl_operator:DualCamCtrlOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.dualcamctrl.synthesis:DualCamCtrlSynthesis",
)
MotionCtrlPipeline = _component_pipeline_class(
    "MotionCtrlPipeline",
    doc="WorldFoundry camera-control video pipeline for MotionCtrl.",
    model_id="motionctrl",
    operator_target="worldfoundry.operators.motionctrl_operator:MotionCtrlOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.motionctrl.synthesis:MotionCtrlSynthesis",
)
MotionCtrlPipeline.MODEL_PATH_OPTION = "ckpt_path"
DreamDojoPipeline = _component_pipeline_class(
    "DreamDojoPipeline",
    doc="WorldFoundry robot-world pipeline for NVIDIA DreamDojo.",
    model_id="dreamdojo",
    operator_target="worldfoundry.operators.dreamdojo_operator:DreamDojoOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.action_generation.dreamdojo:DreamDojoSynthesis",
)
DVLTPipeline = _component_pipeline_class(
    "DVLTPipeline",
    doc="WorldFoundry pipeline for the in-tree DVLT multi-view 3D reconstruction runtime.",
    model_id="dvlt",
    operator_target="worldfoundry.operators.dvlt_operator:DVLTOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.base_models.three_dimensions.depth.dvlt:DVLTSynthesis",
)
DVLTPipeline.MODEL_PATH_OPTION = "checkpoint_path"
IRASimPipeline = _component_pipeline_class(
    "IRASimPipeline",
    doc="WorldFoundry robotics world-model pipeline for IRASim.",
    model_id="irasim",
    operator_target="worldfoundry.operators.irasim_operator:IRASimOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.irasim:IRASimSynthesis",
)
OpenMAGVIT2Pipeline = _component_pipeline_class(
    "OpenMAGVIT2Pipeline",
    doc="WorldFoundry video/token-generation pipeline for Open-MAGVIT2.",
    model_id="open-magvit2",
    operator_target="worldfoundry.operators.open_magvit2_operator:OpenMAGVIT2Operator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.open_magvit2:OpenMAGVIT2Synthesis",
)
PandoraPipeline = _component_pipeline_class(
    "PandoraPipeline",
    doc="WorldFoundry diffusion world-model pipeline for Pandora.",
    model_id="pandora",
    operator_target="worldfoundry.operators.pandora_operator:PandoraOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.pandora:PandoraSynthesis",
)
PixelSplatPipeline = _component_pipeline_class(
    "PixelSplatPipeline",
    doc="WorldFoundry 3D Gaussian splatting pipeline for pixelSplat.",
    model_id="pixelsplat",
    operator_target="worldfoundry.operators.pixelsplat_operator:PixelSplatOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.pixelsplat:PixelSplatSynthesis",
    model_path_option="checkpoint_path",
)
ShowOPipeline = _component_pipeline_class(
    "ShowOPipeline",
    doc="WorldFoundry multimodal video-generation pipeline for Show-o.",
    model_id="show-o",
    operator_target="worldfoundry.operators.show_o_operator:ShowOOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.show_o:ShowOSynthesis",
)
Splatt3RPipeline = _component_pipeline_class(
    "Splatt3RPipeline",
    doc="WorldFoundry 3D Gaussian splatting pipeline for Splatt3R.",
    model_id="splatt3r",
    operator_target="worldfoundry.operators.splatt3r_operator:Splatt3ROperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.base_models.three_dimensions.general_3d.splatt3r:Splatt3RSynthesis",
)
Splatt3RPipeline.MODEL_PATH_OPTION = "checkpoint_path"
ZeroScopePipeline = _component_pipeline_class(
    "ZeroScopePipeline",
    doc="WorldFoundry video-generation pipeline for ZeroScope.",
    model_id="zeroscope",
    operator_target="worldfoundry.operators.zeroscope_operator:ZeroScopeOperator",
    memory_target="worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory",
    synthesis_target="worldfoundry.synthesis.visual_generation.zeroscope:ZeroScopeSynthesis",
)
ZeroScopePipeline.MODEL_PATH_OPTION = "model_path"

__all__ = [
    "ACTPipeline",
    "AnimateDiffPipeline",
    "BeingH07Pipeline",
    "BeingH05Pipeline",
    "CameraCtrlPipeline",
    "CogACTPipeline",
    "ComponentPipeline",
    "DBCogACTPipeline",
    "DVLTPipeline",
    "DiffusionPolicyPipeline",
    "DreamDojoPipeline",
    "DreamZeroPipeline",
    "DualCamCtrlPipeline",
    "FastWAMPipeline",
    "GaussianActorPipeline",
    "GigaBrain0Pipeline",
    "GR00TPipeline",
    "IRASimPipeline",
    "LAPAPipeline",
    "LastR1Pipeline",
    "LingBotVAPipeline",
    "LingBotVLAPipeline",
    "LingBotVLAV2Pipeline",
    "MMEVLAPipeline",
    "MolmoAct2Pipeline",
    "MolmoBotPipeline",
    "MotionCtrlPipeline",
    "MultiTaskDiTPipeline",
    "OctoPipeline",
    "OpenMAGVIT2Pipeline",
    "OpenPIPipeline",
    "OpenVLAOFTPipeline",
    "OpenVLAPipeline",
    "PandoraPipeline",
    "PixelSplatPipeline",
    "RoboFlamingoPipeline",
    "ShowOPipeline",
    "SmolVLAPipeline",
    "Splatt3RPipeline",
    "SpiritV15Pipeline",
    "StarVLAPipeline",
    "TDMPCPipeline",
    "VQBeTPipeline",
    "VLANeXtPipeline",
    "XVLAPipeline",
    "ZeroScopePipeline",
]
