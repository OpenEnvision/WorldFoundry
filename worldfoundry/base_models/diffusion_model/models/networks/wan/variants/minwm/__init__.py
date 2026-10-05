"""MinWM: Wan 2.1 causal + action + ProPE pose encoding.

Submodules
----------
- :mod:`.causal` — causal Wan 2.1 transformer (chunk KV cache).
- :mod:`.action` — action-conditioned causal graph.
- :mod:`.prope` — projective positional encoding for camera / pose.

Adds action and causal hooks plus ProPE.  No VACE or linear attention.
"""
