_base_ = ["../../../configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"]

custom_imports = {
    "imports": ["projects.assigner_visualization.detectors", "projects.assigner_visualization.dense_heads"]
}

model = {"type": "YOLODetectorAssigner", "bbox_head": {"type": "YOLOv8HeadAssigner"}}
