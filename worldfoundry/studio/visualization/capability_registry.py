"""Auditable visualization support for in-tree 3D and perception models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_MODELS_ROOT = REPO_ROOT / "worldfoundry" / "base_models"

RENDER_BACKENDS = {
    "detection": "media",
    "mask": "media",
    "flow": "media",
    "tracks": "media",
    "feature-pca": "media",
    "depth": "media",
    "normal": "media",
    "keypoints": "media",
    "text": "media",
    "media": "media",
    "point-cloud": "points",
    "mesh": "points",
    "camera": "points",
    "gaussian-splat": "spark",
    "rerun": "rerun",
    "viser": "viser",
}


@dataclass(frozen=True)
class ModelVisualizationCapability:
    model_id: str
    package: str
    renderers: tuple[str, ...]
    evidence: str

    @property
    def family(self) -> str:
        return self.package.split("/", 1)[0]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["family"] = self.family
        payload["backends"] = sorted({RENDER_BACKENDS[item] for item in self.renderers})
        payload["package_ready"] = (BASE_MODELS_ROOT / self.package).is_dir()
        payload["evidence_ready"] = (BASE_MODELS_ROOT / self.evidence).exists()
        return payload


def _cap(model_id: str, package: str, renderers: tuple[str, ...], evidence: str) -> ModelVisualizationCapability:
    return ModelVisualizationCapability(model_id, package, renderers, evidence)


MODEL_VISUALIZATION_CAPABILITIES = (
    # Depth and geometric perception.
    _cap(
        "depth-anything-v1",
        "three_dimensions/depth/depth_anything/depth_anything_v1",
        ("depth",),
        "three_dimensions/depth/depth_anything/depth_anything_v1/dap_model.py",
    ),
    _cap(
        "depth-anything-v2",
        "three_dimensions/depth/depth_anything/depth_anything_v2",
        ("depth",),
        "three_dimensions/depth/depth_anything/depth_anything_v2/dpt.py",
    ),
    _cap(
        "depth-anything-v3",
        "three_dimensions/depth/depth_anything/depth_anything_v3",
        ("depth", "point-cloud", "mesh"),
        "three_dimensions/depth/depth_anything/depth_anything_v3/api.py",
    ),
    _cap(
        "dvlt",
        "three_dimensions/depth/dvlt",
        ("depth", "normal", "camera", "point-cloud"),
        "three_dimensions/depth/dvlt/viz/camera_overlay.py",
    ),
    _cap(
        "metric3d",
        "three_dimensions/depth/metric3d",
        ("depth", "normal"),
        "three_dimensions/depth/metric3d/model/monodepth_model.py",
    ),
    _cap("midas", "three_dimensions/depth/midas", ("depth",), "three_dimensions/depth/midas/model_loader.py"),
    _cap(
        "moge",
        "three_dimensions/depth/moge",
        ("depth", "normal", "point-cloud", "mesh"),
        "three_dimensions/depth/moge/utils/io.py",
    ),
    _cap(
        "prior-depth-anything",
        "three_dimensions/depth/priorda",
        ("depth",),
        "three_dimensions/depth/priorda/depth_completion.py",
    ),
    _cap(
        "unidepth",
        "three_dimensions/depth/unidepth",
        ("depth", "point-cloud"),
        "three_dimensions/depth/unidepth/models/unidepthv2/unidepthv2.py",
    ),
    _cap(
        "unik3d",
        "three_dimensions/depth/unik3d",
        ("depth", "normal", "point-cloud"),
        "three_dimensions/depth/unik3d/unik3d.py",
    ),
    _cap(
        "longvie-video-depth",
        "three_dimensions/depth/video_depth_anything_longvie",
        ("depth", "media"),
        "three_dimensions/depth/video_depth_anything_longvie/video_depth_anything/dpt.py",
    ),
    _cap(
        "video-depth-anything",
        "three_dimensions/depth/videodepthanything",
        ("depth", "media"),
        "three_dimensions/depth/videodepthanything/video_depth.py",
    ),
    # Reconstruction, dynamic scenes, splats, and SLAM.
    _cap(
        "dust3r",
        "three_dimensions/general_3d/dust3r",
        ("point-cloud", "mesh", "viser"),
        "three_dimensions/general_3d/dust3r/dust3r/viz/scene3d_dust.py",
    ),
    _cap(
        "4d-gaussians",
        "three_dimensions/general_3d/four_d_gaussians",
        ("gaussian-splat", "media"),
        "three_dimensions/general_3d/four_d_gaussians/four_d_gaussians_runtime/render.py",
    ),
    _cap(
        "geocalib",
        "three_dimensions/general_3d/geocalib",
        ("camera", "media"),
        "three_dimensions/general_3d/geocalib/geocalib.py",
    ),
    _cap(
        "lagernvs",
        "three_dimensions/general_3d/lagernvs",
        ("media",),
        "three_dimensions/general_3d/lagernvs/lagernvs_runtime/minimal_inference.py",
    ),
    _cap(
        "mast3r",
        "three_dimensions/general_3d/mast3r",
        ("point-cloud", "mesh", "viser"),
        "three_dimensions/general_3d/mast3r/mast3r/cloud_opt/sparse_ga.py",
    ),
    _cap(
        "monst3r",
        "three_dimensions/general_3d/monst3r",
        ("point-cloud", "mesh", "viser", "rerun"),
        "three_dimensions/general_3d/monst3r/demo.py",
    ),
    _cap(
        "mvdiffusion",
        "three_dimensions/general_3d/mvdiffusion",
        ("media",),
        "three_dimensions/general_3d/mvdiffusion/mvdiffusion_runtime/demo.py",
    ),
    _cap(
        "splatt3r",
        "three_dimensions/general_3d/splatt3r",
        ("gaussian-splat",),
        "three_dimensions/general_3d/splatt3r/splatt3r_runtime/utils/export.py",
    ),
    _cap(
        "stable-virtual-camera",
        "three_dimensions/general_3d/stable_virtual_camera",
        ("media", "camera"),
        "three_dimensions/general_3d/stable_virtual_camera/stable_virtual_camera_runtime/demo.py",
    ),
    _cap(
        "vipe",
        "three_dimensions/general_3d/vipe",
        ("point-cloud", "camera", "rerun"),
        "three_dimensions/general_3d/vipe/pipeline/default.py",
    ),
    _cap(
        "cut3r",
        "three_dimensions/point_clouds/cut3r",
        ("point-cloud", "viser", "rerun"),
        "three_dimensions/point_clouds/cut3r/model.py",
    ),
    _cap(
        "flash-world",
        "three_dimensions/point_clouds/flash_world",
        ("gaussian-splat", "media"),
        "three_dimensions/point_clouds/flash_world/render.py",
    ),
    _cap(
        "gaussian-splatting",
        "three_dimensions/point_clouds/gaussian_splatting",
        ("gaussian-splat", "media"),
        "three_dimensions/point_clouds/gaussian_splatting/gaussian_renderer/__init__.py",
    ),
    _cap(
        "hunyuan-world-mirror",
        "three_dimensions/point_clouds/hunyuan_mirror",
        ("gaussian-splat", "media"),
        "three_dimensions/point_clouds/hunyuan_mirror/models/heads/dense_head.py",
    ),
    _cap(
        "hy-world-2.0",
        "three_dimensions/point_clouds/hyworldmirror_2p0",
        ("gaussian-splat", "media"),
        "three_dimensions/point_clouds/hyworldmirror_2p0/utils/scene_render.py",
    ),
    _cap(
        "infinite-vggt",
        "three_dimensions/point_clouds/infinite_vggt",
        ("point-cloud", "mesh"),
        "three_dimensions/point_clouds/infinite_vggt/utils/geometry.py",
    ),
    _cap(
        "lingbot-map",
        "three_dimensions/point_clouds/lingbot_map",
        ("point-cloud", "viser"),
        "three_dimensions/point_clouds/lingbot_map/lingbot_map/models/gct_stream.py",
    ),
    _cap(
        "loger",
        "three_dimensions/point_clouds/loger",
        ("point-cloud", "mesh"),
        "three_dimensions/point_clouds/loger/pi3.py",
    ),
    _cap(
        "pi3",
        "three_dimensions/point_clouds/pi3",
        ("point-cloud", "mesh", "viser"),
        "three_dimensions/point_clouds/pi3/pi3/pipe/pi3x_vo.py",
    ),
    _cap(
        "pixelsplat",
        "three_dimensions/point_clouds/pixelsplat",
        ("media",),
        "three_dimensions/point_clouds/pixelsplat/decoder_splatting_cuda.py",
    ),
    _cap(
        "vggt",
        "three_dimensions/point_clouds/vggt",
        ("point-cloud", "mesh", "viser"),
        "three_dimensions/point_clouds/vggt/vggt/models/vggt.py",
    ),
    _cap(
        "vggt-omega",
        "three_dimensions/point_clouds/vggt_omega",
        ("point-cloud", "mesh", "viser"),
        "three_dimensions/point_clouds/vggt_omega/vggt_omega/models/aggregator.py",
    ),
    _cap(
        "droid-slam",
        "three_dimensions/slam/droid_slam",
        ("point-cloud", "camera", "viser"),
        "three_dimensions/slam/droid_slam/droid.py",
    ),
    _cap(
        "megasam",
        "three_dimensions/slam/mega_sam_runtime",
        ("depth", "point-cloud", "camera", "media"),
        "three_dimensions/slam/mega_sam_runtime/base/droid_slam/visualization.py",
    ),
    # Perception models with an official visual output or demo.
    _cap(
        "umt",
        "perception_core/action_recognition/umt",
        ("text", "media"),
        "perception_core/action_recognition/umt/models/modeling_finetune.py",
    ),
    _cap(
        "videomae-mmaction",
        "perception_core/action_recognition/videomae_mmaction",
        ("text", "media"),
        "perception_core/action_recognition/videomae_mmaction/configs/_base_/default_runtime.py",
    ),
    _cap("grit", "perception_core/captioning/grit", ("detection", "text"), "perception_core/captioning/grit/model.py"),
    _cap(
        "tag2text", "perception_core/captioning/tag2text", ("text",), "perception_core/captioning/tag2text/tag2text.py"
    ),
    _cap(
        "grounding-dino",
        "perception_core/detection/grounding_dino",
        ("detection",),
        "perception_core/detection/grounding_dino/util/inference.py",
    ),
    _cap(
        "yolo-world",
        "perception_core/detection/yolo_world",
        ("detection",),
        "perception_core/detection/yolo_world/mmyolo/demo/image_demo.py",
    ),
    _cap(
        "deepface",
        "perception_core/face/deepface",
        ("detection", "text", "media"),
        "perception_core/face/deepface/deepface/commons/realtime.py",
    ),
    _cap(
        "amt",
        "perception_core/frame_interpolation/amt",
        ("media", "flow"),
        "perception_core/frame_interpolation/amt/utils/flow_utils.py",
    ),
    _cap(
        "vfimamba",
        "perception_core/frame_interpolation/vfimamba",
        ("media",),
        "perception_core/frame_interpolation/vfimamba/model/flow_estimation.py",
    ),
    _cap(
        "dino",
        "perception_core/general_perception/dino",
        ("feature-pca",),
        "perception_core/general_perception/dino/models/vision_transformer.py",
    ),
    _cap(
        "dinov2",
        "perception_core/general_perception/dinov2",
        ("feature-pca",),
        "perception_core/general_perception/dinov2/models/vision_transformer.py",
    ),
    _cap(
        "dinov3",
        "perception_core/general_perception/dinov3",
        ("feature-pca",),
        "perception_core/general_perception/dinov3/models/vision_transformer.py",
    ),
    _cap(
        "vit-human-detector",
        "perception_core/human_anatomy/vit_detector",
        ("detection", "text"),
        "perception_core/human_anatomy/vit_detector/inference.py",
    ),
    _cap(
        "flowformer++",
        "perception_core/optical_flow/flowformerplusplus",
        ("flow",),
        "perception_core/optical_flow/flowformerplusplus/core/utils/flow_viz.py",
    ),
    _cap("raft", "perception_core/optical_flow/raft", ("flow",), "perception_core/optical_flow/raft/raft.py"),
    _cap(
        "sea-raft",
        "perception_core/optical_flow/sea_raft",
        ("flow",),
        "perception_core/optical_flow/sea_raft/custom.py",
    ),
    _cap(
        "nudenet", "perception_core/safety/nudenet", ("detection", "text"), "perception_core/safety/nudenet/nudenet.py"
    ),
    _cap(
        "efficient-sam",
        "perception_core/segment/efficient_sam",
        ("mask",),
        "perception_core/segment/efficient_sam/efficient_sam.py",
    ),
    _cap(
        "grounded-sam",
        "perception_core/segment/grounded_segment_anything",
        ("detection", "mask"),
        "perception_core/segment/grounded_segment_anything/pipeline.py",
    ),
    _cap(
        "mobile-sam",
        "perception_core/segment/mobile_sam",
        ("mask",),
        "perception_core/segment/mobile_sam/predictor.py",
    ),
    _cap(
        "repvit-sam",
        "perception_core/segment/repvit_sam",
        ("mask",),
        "perception_core/segment/repvit_sam/predictor.py",
    ),
    _cap("sam2", "perception_core/segment/sam2", ("mask", "media"), "perception_core/segment/sam2/utils/misc.py"),
    _cap("sam3", "perception_core/segment/sam3", ("mask", "media"), "perception_core/segment/sam3/model/sam3_image.py"),
    _cap(
        "sam-v1",
        "perception_core/segment/sam_v1",
        ("mask",),
        "perception_core/segment/sam_v1/automatic_mask_generator.py",
    ),
    _cap(
        "aot",
        "perception_core/tracking/aot",
        ("mask", "media"),
        "perception_core/tracking/track_anything/aot/utils/image.py",
    ),
    _cap(
        "cotracker",
        "perception_core/tracking/cotracker",
        ("tracks", "media"),
        "perception_core/tracking/cotracker/visualizer.py",
    ),
    _cap("dot", "perception_core/tracking/dot", ("tracks", "media"), "perception_core/tracking/dot/utils/io.py"),
    _cap(
        "track-anything",
        "perception_core/tracking/track_anything",
        ("mask", "tracks", "media"),
        "perception_core/tracking/track_anything/seg_tracker.py",
    ),
)


def visualization_capabilities(
    *, family: str | None = None, model_id: str | None = None
) -> tuple[ModelVisualizationCapability, ...]:
    """Return registered official visualization capabilities, optionally filtered."""

    return tuple(
        item
        for item in MODEL_VISUALIZATION_CAPABILITIES
        if (family is None or item.family == family) and (model_id is None or item.model_id == model_id)
    )


def visualization_inventory(*, family: str | None = None, model_id: str | None = None) -> dict[str, object]:
    entries = visualization_capabilities(family=family, model_id=model_id)
    return {
        "schema_version": 1,
        "model_count": len(entries),
        "renderer_backends": dict(RENDER_BACKENDS),
        "models": [item.to_dict() for item in entries],
    }
