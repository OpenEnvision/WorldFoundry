_base_ = ["../../../configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py"]

custom_imports = {
    "imports": ["projects.assigner_visualization.detectors", "projects.assigner_visualization.dense_heads"]
}

model = {"type": "YOLODetectorAssigner", "bbox_head": {"type": "YOLOv5HeadAssigner"}}
