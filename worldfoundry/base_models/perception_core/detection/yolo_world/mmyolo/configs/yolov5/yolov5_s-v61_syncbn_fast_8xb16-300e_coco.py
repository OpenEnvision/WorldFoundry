_base_ = "yolov5_s-v61_syncbn_8xb16-300e_coco.py"

# fast means faster training speed,
# but less flexibility for multitasking
model = {
    "data_preprocessor": {
        "type": "YOLOv5DetDataPreprocessor",
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "bgr_to_rgb": True,
    }
}

train_dataloader = {"collate_fn": {"type": "yolov5_collate"}}
