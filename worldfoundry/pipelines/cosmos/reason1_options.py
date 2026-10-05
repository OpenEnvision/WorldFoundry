"""Public loading options for pipelines sharing the Reason1 conditioner."""

from __future__ import annotations

from collections.abc import Mapping

from worldfoundry.core.model_loading.text_embeddings import (
    DEFAULT_EMBEDDING_CACHE_MAX_BYTES,
    validate_prompt_encoder_options,
)


def reason1_component_options(options: Mapping[str, object]) -> dict[str, bool | int]:
    """Resolve explicit CPU placement and bounded whole-batch embedding reuse.

    Validation runs before any native components or checkpoints are loaded.
    CPU encoding and caching remain disabled unless explicitly requested.
    """

    return validate_prompt_encoder_options(
        run_on_cpu=options.get("text_encoder_run_on_cpu", False),
        embedding_cache_size=options.get("text_embedding_cache_size", 0),
        embedding_cache_max_bytes=options.get("text_embedding_cache_max_bytes", DEFAULT_EMBEDDING_CACHE_MAX_BYTES),
    )
