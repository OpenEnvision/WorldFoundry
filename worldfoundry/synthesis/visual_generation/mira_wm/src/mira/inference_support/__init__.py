"""Shared training infrastructure for the codec and world-model trainers.

The submodules are single-GPU safe: every ``torch.distributed`` call is guarded by
``is_initialized()`` so the trainers (and tests) run without ``torchrun``.

    distributed        — DistributedSettings, get_distributed_settings, set_up_distributed
    ema                — ModelEMA (parameter EMA + swap/restore), DistributedEMA (scalar EMA)
    lr_schedule        — WarmupConstantCosineDecayLR
    checkpoint_manager — CheckpointManager (save / continue_from / finetune_from)
    tracker            — TrainingTracker, periodic_event, display_execution_time
    metrics            — DistributedMetric and image-quality metrics
    visualization      — video helpers for W&B logging
"""
