"""
Gemma Text Encoder Wrapper for DMD distillation.

Provides a simple interface for text encoding without prompt enhancement.
Just pure text -> context embedding conversion.
"""

import torch
import torch.nn as nn

from ...ltx_core.loader.registry import Registry


class GemmaTextEncoderWrapper(nn.Module):
    """
    Wrapper for Gemma text encoder to provide DMD-compatible interface.

    This wrapper:
    - Takes raw text prompts (no enhancement needed)
    - Returns conditional_dict with video_context and audio_context
    - Handles batched encoding
    """

    def __init__(
        self,
        text_encoder,
        embeddings_processor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            text_encoder: GemmaTextEncoder instance
            embeddings_processor: EmbeddingsProcessor instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def forward(
        self,
        text_prompts: list[str],
        padding_side: str = "left",
    ) -> dict[str, torch.Tensor | None]:
        """
        Encode text prompts to conditioning embeddings.

        Args:
            text_prompts: List of text prompts (already processed, no enhancement)
            padding_side: Padding side for tokenizer

        Returns:
            Dictionary containing:
                - video_context: [B, seq_len, dim] video conditioning
                - audio_context: [B, seq_len, dim] audio conditioning
                - attention_mask: [B, seq_len] attention mask
        """
        if not text_prompts:
            return {
                "video_context": None,
                "audio_context": None,
                "attention_mask": None,
            }

        hidden_states, attention_mask = self.text_encoder.encode_batch(
            text_prompts,
            padding_side=padding_side,
        )
        output = self.embeddings_processor.process_hidden_states(
            hidden_states,
            attention_mask,
            padding_side=padding_side,
        )

        return {
            "video_context": output.video_encoding,
            "audio_context": output.audio_encoding,
            "attention_mask": output.attention_mask,
        }

    def encode_batch(
        self,
        text_prompts: list[str],
    ) -> dict[str, torch.Tensor]:
        """Alias for forward() with default padding."""
        return self.forward(text_prompts)


def create_text_encoder_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Registry | None = None,
) -> GemmaTextEncoderWrapper:
    """
    Factory function to create GemmaTextEncoderWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype

    Returns:
        Configured GemmaTextEncoderWrapper
    """
    from ...ltx_pipelines.utils.model_ledger import ModelLedger

    # Load to CPU first to avoid safetensors device issues
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        registry=registry,
    )

    text_encoder = ledger.text_encoder().to(device=device, dtype=dtype)
    embeddings_processor = ledger.gemma_embeddings_processor().to(device=device, dtype=dtype)

    wrapper = GemmaTextEncoderWrapper(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        dtype=dtype,
    )

    return wrapper


def create_language_only_text_encoder(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Registry | None = None,
):
    """Load only the Gemma language backbone and tokenizer for DMD encoding."""
    from ...ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        registry=registry,
    )
    return ledger.language_only_text_encoder().to(device=device, dtype=dtype).eval()


def create_text_embeddings_processor(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Registry | None = None,
):
    """Load only Echo's feature extractor and video/audio text connectors."""
    from ...ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        registry=registry,
    )
    return ledger.gemma_embeddings_processor().to(device=device, dtype=dtype).eval()
