"""Kandinsky-6 transformer component factory."""


def build_kandinsky6_denoiser(context):
    from ...loaders.kandinsky6 import component_config, create_bare_dit
    from ...optimizations.kandinsky6.kernels import bind_attention

    cfg, device = component_config(context)
    requested = context.policy.attention.value
    engine = {"torch": "sdpa", "flash": "auto"}.get(requested, requested)
    return create_bare_dit(cfg, device, bind_attention(engine))
