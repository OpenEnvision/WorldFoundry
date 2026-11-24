_base_ = "mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py"

checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth"

model = {
    "backbone": {"init_cfg": {"type": "Pretrained", "prefix": "backbone.", "checkpoint": checkpoint}},
    "bbox_head": {"num_classes": 1},
    "test_cfg": {
        "nms_pre": 1000,
        "min_bbox_size": 0,
        "score_thr": 0.05,
        "nms": {"type": "nms", "iou_threshold": 0.6},
        "max_per_img": 100,
    },
}

train_dataloader = {"dataset": {"metainfo": {"classes": ("person",)}}}

val_dataloader = {"dataset": {"metainfo": {"classes": ("person",)}}}
test_dataloader = val_dataloader
