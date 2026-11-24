_base_ = ["../../../configs/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py"]

custom_imports = {
    "imports": ["projects.assigner_visualization.detectors", "projects.assigner_visualization.dense_heads"]
}

model = {"type": "YOLODetectorAssigner", "bbox_head": {"type": "YOLOv7HeadAssigner"}}
