_base_ = ["../../../configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py"]

custom_imports = {
    "imports": ["projects.assigner_visualization.detectors", "projects.assigner_visualization.dense_heads"]
}

model = {"type": "YOLODetectorAssigner", "bbox_head": {"type": "RTMHeadAssigner"}}
