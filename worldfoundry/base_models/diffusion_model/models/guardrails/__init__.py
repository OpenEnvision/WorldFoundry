"""Optional safety / guardrail helpers used by diffusion recipes.

Not a runner Protocol.  Cosmos recipes may call these modules to
classify prompts (LlamaGuard / Aegis category tables) or blur faces
in decoded frames.  They do not implement :class:`~...contracts.Denoiser`
or :class:`~...contracts.ConditionEncoder`.
"""
