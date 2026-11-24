default_scope = "mmyolo"

default_hooks = {
    "timer": {"type": "IterTimerHook"},
    "logger": {"type": "LoggerHook", "interval": 50},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "checkpoint": {"type": "CheckpointHook", "interval": 1},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
    "visualization": {"type": "mmdet.DetVisualizationHook"},
}

env_cfg = {
    "cudnn_benchmark": False,
    "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
    "dist_cfg": {"backend": "nccl"},
}

vis_backends = [{"type": "LocalVisBackend"}]
visualizer = {"type": "mmdet.DetLocalVisualizer", "vis_backends": vis_backends, "name": "visualizer"}
log_processor = {"type": "LogProcessor", "window_size": 50, "by_epoch": True}

log_level = "INFO"
load_from = None
resume = False

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions
# before MMDet 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

backend_args = None
