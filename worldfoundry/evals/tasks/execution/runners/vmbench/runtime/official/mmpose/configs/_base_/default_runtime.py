default_scope = "mmpose"

# hooks
default_hooks = {
    "timer": {"type": "IterTimerHook"},
    "logger": {"type": "LoggerHook", "interval": 50},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "checkpoint": {"type": "CheckpointHook", "interval": 10},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
    "visualization": {"type": "PoseVisualizationHook", "enable": False},
    "badcase": {
        "type": "BadCaseAnalysisHook",
        "enable": False,
        "out_dir": "badcase",
        "metric_type": "loss",
        "badcase_thr": 5,
    },
}

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    {"type": "SyncBuffersHook"}
]

# multi-processing backend
env_cfg = {
    "cudnn_benchmark": False,
    "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
    "dist_cfg": {"backend": "nccl"},
}

# visualizer
vis_backends = [
    {"type": "LocalVisBackend"},
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = {"type": "PoseLocalVisualizer", "vis_backends": vis_backends, "name": "visualizer"}

# logger
log_processor = {"type": "LogProcessor", "window_size": 50, "by_epoch": True, "num_digits": 6}
log_level = "INFO"
load_from = None
resume = False

# file I/O backend
backend_args = {"backend": "local"}

# training/validation/testing progress
train_cfg = {"by_epoch": True}
val_cfg = {}
test_cfg = {}
