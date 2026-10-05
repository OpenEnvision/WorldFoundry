"""Accelerate-backed inference runner for Matrix-Game 3.5."""

from .rollout import run_mosaic_inference


def run_mosaic_inference_task(
    accelerator,
    model,
    *,
    log_dir,
    args,
    inference_dataset,
):
    if inference_dataset is None:
        raise ValueError("Matrix-Game inference requires an input dataset")
    model = accelerator.prepare(model)
    run_mosaic_inference(
        accelerator,
        inference_dataset,
        model,
        log_dir,
        args=args,
        run_id=0,
    )


__all__ = ["run_mosaic_inference_task"]
