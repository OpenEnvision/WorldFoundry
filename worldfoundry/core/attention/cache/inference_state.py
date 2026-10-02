"""Inference state containers shared by streaming generation runtimes.

Causal / incremental decoders need a place to stash per-layer KV tensors
and the current sequence offset without depending on a specific
transformer implementation. :class:`InferenceParams` is that bag of
state: layers write into ``key_value_memory_dict``, the decode loop
advances ``sequence_len_offset``, and batched generation can reorder
the cache with :meth:`InferenceParams.swap_key_value_dict` when the
active batch permutation changes.

This is *not* the CUDA-graph-friendly :class:`BlockKVCache` used by
in-tree causal video graphs. It is the lighter, Megatron-style
container that official and vendored text/video decoders already
expect. Capacity fields (``max_batch_size``, ``max_sequence_length``)
are recorded for callers that preallocate; this class does not allocate
the tensors itself.
"""

from __future__ import annotations


class InferenceParams:
    """Per-step KV and offset state for incremental decoding.

    Attributes:
        max_batch_size: Upper bound recorded at construction; used by
            callers that size static caches.
        max_sequence_length: Upper bound on the decode horizon.
        sequence_len_offset: Tokens already committed to the cache.
        key_value_memory_dict: ``layer_id -> (key, value)`` tensors.
            Layout is layer-defined; this class only stores and reorders.
        update_kv_cache: Hint for layers that optionally skip writes.
    """

    def __init__(self, max_batch_size: int, max_sequence_length: int):
        """Record cache capacity; tensors are allocated by the model."""
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.key_value_memory_dict = {}
        self.update_kv_cache = False

    def swap_key_value_dict(self, batch_idx) -> None:
        """Reorder the batch axis of every cached ``(key, value)`` pair.

        Used when a generate loop drops or permutes live sequences (for
        example after early EOS). ``batch_idx`` is gathered along axis 1,
        which is the batch dimension of the Megatron-style cache layout
        ``[seq, batch, ...]``.

        Args:
            batch_idx: Index tensor/list whose length matches the current
                cache batch size.

        Raises:
            ValueError: The cache is empty, so there is nothing to swap.
        """
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict is empty")

        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1]
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (new_inference_key_memory, new_inference_value_memory)


__all__ = ["InferenceParams"]
