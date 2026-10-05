from worldfoundry.base_models.diffusion_model import (
    DiffusionRequest,
    SamplingConfig,
)

from .prompting import _require_nonempty_prompt


def run_mosaic_segment_inference(
    pipeline,
    prompt,
    negative_prompt,
    input_image,
    height,
    width,
    num_frames,
    seed,
    *,
    num_inference_steps=25,
    cfg_scale=5.0,
    negative_no_prope=False,
    negative_no_context=False,
    enable_mosaic=True,
    only_prope=False,
    mosaic_latent=None,
    mosaic_revgrid=None,
    mosaic_use_revgrid_rope=False,
    mosaic_view_change=None,
    mosaic_view_change_prope=False,
    mosaic_mask_holes=True,
    mosaic_drop_holes=False,
    mosaic_frame_indices=None,
    tiled=True,
    prope_camera_kwargs=None,
    return_latent=True,
    first_frame_latents=None,
    latent_rope_time_indices=None,
    subject_ref_latents=None,
    subject_ref_slot_ratio=0.5,
    subject_ref_time_gap=1,
    subject_ref_prope_mode="identity",
    allow_empty_prompt=False,
):
    """Run one TI2V mosaic inference pass.

    Normalizes Mosaic and PRoPE inputs into the same ``DiffusionRequest`` used
    by every native diffusion model. Multi-segment rollout remains the
    synthesis caller's responsibility.

    `first_frame_latents` and `input_image` are mutually exclusive condition
    sources for the first-frame slot. Passing `first_frame_latents` directly
    skips the VAE decode->encode round-trip that happens when an `input_image`
    is provided.
    """
    prompt = _require_nonempty_prompt(
        prompt,
        phase="inference",
        allow_empty=allow_empty_prompt,
    )
    if input_image is not None and first_frame_latents is None:
        raise ValueError("native Matrix inference requires first_frame_latents instead of an implicit image encode")
    use_mosaic = enable_mosaic and not only_prope
    inputs = {
        "tiled": tiled,
        "first_frame_latents": first_frame_latents,
        "negative_no_prope": negative_no_prope,
        "negative_no_context": negative_no_context,
        "return_latent": return_latent,
        "latent_rope_time_indices": latent_rope_time_indices,
        "subject_ref_latents": subject_ref_latents,
        "subject_ref_slot_ratio": subject_ref_slot_ratio,
        "subject_ref_time_gap": subject_ref_time_gap,
        "subject_ref_prope_mode": subject_ref_prope_mode,
    }
    if use_mosaic:
        inputs.update(
            {
                "mosaic_latent": mosaic_latent,
                "mosaic_timestep_zero": True,
                "mosaic_revgrid": mosaic_revgrid,
                "mosaic_use_revgrid_rope": mosaic_use_revgrid_rope,
                "mosaic_view_change": mosaic_view_change,
                "mosaic_view_change_prope": mosaic_view_change_prope,
                "mosaic_mask_holes": mosaic_mask_holes,
                "mosaic_drop_holes": mosaic_drop_holes,
                "mosaic_frame_indices": mosaic_frame_indices,
            }
        )
    if prope_camera_kwargs:
        inputs.update(prope_camera_kwargs)

    request = DiffusionRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        sampling=SamplingConfig(
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            seed=seed,
        ),
        inputs=inputs,
    )
    return pipeline(request).sample
