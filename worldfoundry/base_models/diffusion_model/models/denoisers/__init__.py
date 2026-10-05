"""Family adapters that implement the shared native :class:`~...contracts.Denoiser` contract.

Network modules under ``models/networks/*`` contain checkpoint-compatible
math only.  This package owns the small amount of family-specific
state-dict conversion, weight remapping, and contract adaptation needed
by the common loader and runners.  Recipes select a ``build_*_denoiser``
factory via :class:`~...components.ComponentSpec`; they never import
network internals.

This ``__init__`` is a documentation surface only.  Factories live in
sibling modules and are imported by recipes, not re-exported here.

Families
--------
- Wan 2.1 / 2.2 / VACE / SkyReels: single or dual-expert DiT adapters.
- LTX-2 / LTX-Video: joint audio-video or video-only DiT.
- Sana / Sprint / ControlNet / video / streaming / world-model.
- HunyuanVideo / HunyuanVideo 1.5 (including I2V concat condition).
- Cosmos Predict1 GEN3C, Predict2, Predict 2.5 / Transfer 2.5, Cosmos3.
- Gamma-World (causal, few-step, bidirectional).
- Echo-Memory, Step-Video, Matrix-Game 3.5, Vchitect, T2V-Turbo.

Dual-expert routing (Wan 2.2 A14B) is expressed here as two weight
roles on one denoiser object; the recipe still selects the
``wan22-dual-expert-guidance`` execution strategy for CFG scales.
"""
