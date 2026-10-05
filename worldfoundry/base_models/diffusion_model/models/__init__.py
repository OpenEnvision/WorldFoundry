"""Canonical diffusion model components grouped by inference role.

Packages in this directory implement runner Protocols from
:mod:`~worldfoundry.base_models.diffusion_model.contracts`:

- ``autoencoders``: :class:`LatentEncoder` / :class:`LatentDecoder`
- ``encoders``: :class:`ConditionEncoder` (``DiffusionRequest`` → ``Conditioning``)
- ``initializers``: :class:`LatentInitializer` / :class:`EncodedLatentInitializer`
- ``denoisers``: :class:`Denoiser` (``DenoiserInput`` → ``DenoiserOutput``)
- ``upsamplers``: :class:`LatentProcessor` between multi-stage passes
- ``representations``: packing / patchify helpers (not Protocols)
- ``guardrails``: optional safety tables and classifiers
- ``networks``: checkpoint-compatible DiT / UNet math (commented separately)
"""

__all__: list[str] = []
