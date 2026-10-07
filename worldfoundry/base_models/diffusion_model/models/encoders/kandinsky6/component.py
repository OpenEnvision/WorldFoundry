"""Native prompt-conditioning boundary for Kandinsky-6."""

from ....contracts import Conditioning


class Kandinsky6Conditioner:
    def __init__(self, model):
        self.model = model

    def encode(self, request, *, device, dtype):
        positive_embeds, positive_lengths, positive_mask = self.model.encode(list(request.prompts))
        positive = {**positive_embeds, "cu_seqlens": positive_lengths, "attention_mask": positive_mask}
        negative = {}
        if request.negative_prompts:
            embeds, lengths, mask = self.model.encode(list(request.negative_prompts))
            negative = {**embeds, "cu_seqlens": lengths, "attention_mask": mask}
        return Conditioning(positive=positive, negative=negative)


def build_kandinsky6_conditioner(context):
    from ....loaders.kandinsky6 import component_config
    from .text_embedder import Kandinsky6TextEmbedder

    cfg, device = component_config(context)
    return Kandinsky6Conditioner(
        Kandinsky6TextEmbedder(
            qwen_path=cfg.paths.qwen,
            clip_path=cfg.paths.clip,
            max_length=cfg.text_embedder.max_length,
            device=device,
            quantized_qwen=False,
            text_token_padding=cfg.dit.text_token_padding,
        )
    )
