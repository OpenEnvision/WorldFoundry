"""DreamZero: Wan 2.1 action-conditioned DiT (Pi0-style action encoder).

Submodules
----------
- :mod:`.model` — official-layout Wan 2.1 stem used by DreamZero.
- :mod:`.action` — action-token injection on that stem.
- :mod:`.action_encoder` — sinusoidal-time action MLP (Pi0 recipe).
- :mod:`.attention` — DreamZero attention kernels / processors.

Adds action conditioning.  Causal / VACE / camera / linear-attn paths
are not defined here.
"""
