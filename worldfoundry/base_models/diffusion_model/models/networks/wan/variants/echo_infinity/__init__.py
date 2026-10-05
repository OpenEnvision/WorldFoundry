"""Echo-1 Infinity: causal Wan 2.x with sink / query memory banks.

Submodules
----------
- :mod:`.model` — bidirectional official-layout stem used as a building block.
- :mod:`.causal` — causal self-attention + KV cache for streaming chunks.
- :mod:`.infinity` — long-horizon Infinity graph (query memory + sink).
- :mod:`.memory` / :mod:`.sink_memory` / :mod:`.query_memory` — memory adapters.
- :mod:`.attention` — re-export of shared varlen flash attention.

Adds causal attention and memory hooks.  No VACE, camera ControlNet, or
linear-attention processors live in this package.
"""
