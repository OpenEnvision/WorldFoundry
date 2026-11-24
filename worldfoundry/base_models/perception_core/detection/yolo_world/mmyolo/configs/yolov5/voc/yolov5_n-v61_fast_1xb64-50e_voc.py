_base_ = "./yolov5_s-v61_fast_1xb64-50e_voc.py"

deepen_factor = 0.33
widen_factor = 0.25

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth"

model = {
    "backbone": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "neck": {
        "deepen_factor": deepen_factor,
        "widen_factor": widen_factor,
    },
    "bbox_head": {"head_module": {"widen_factor": widen_factor}},
}
