WorldPlay2 CPU reference tensors for upstream revision
`c5d83e32099116ff3a1437a8a05c764d579f704b`.

`upstream_cpu.json` covers bidirectional and cached autoregressive forwards,
memory features, initial and incremental KV prefill, both experts' compact PDD
heads, and a three-chunk Fast rollout. The transformer uses FP32, two layers,
four heads, and a hidden width of 32. Parameters follow sorted checkpoint keys
with uniform `[-0.2, 0.2]` values from seeds 123 (BI/AR), 456 (high), and 789
(low). Input tensors are stored alongside their reference outputs.

The rollout uses the official `WorldPlay2Pipeline.generate_chunked` method with
shared deterministic text and codec test doubles, 12 latent frames, 45 pixel
frames, three chunks, four evaluations per chunk, and seed 42. It covers the
transformer, sampling, actions, prompt switches, compressed memory, and KV
updates. Real text/VAE checkpoint inference and GPU generation are separate
validation stages.

The JSON records retain each array's dtype, shape and numeric values. Complex
RoPE entries store real/imaginary pairs as float64 values.
