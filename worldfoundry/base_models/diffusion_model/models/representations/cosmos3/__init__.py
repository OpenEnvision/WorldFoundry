"""Cosmos3 omni-sequence packing and embodiment-domain metadata.

:func:`build_cosmos3_sequence_layout` turns framework
:class:`~...contracts.ModalityState` objects plus tokenizer ``input_ids``
into the index / position tensors consumed by
:class:`~...denoisers.cosmos3.Cosmos3JointDenoiser`.  :data:`ACTION_DOMAIN_IDS`
and :data:`ACTION_RAW_DIMS` are the published embodiment catalog.
"""

from .actions import ACTION_DOMAIN_IDS, ACTION_RAW_DIMS
from .packing import Cosmos3SequenceLayout, build_cosmos3_sequence_layout

__all__ = [
    "ACTION_DOMAIN_IDS",
    "ACTION_RAW_DIMS",
    "Cosmos3SequenceLayout",
    "build_cosmos3_sequence_layout",
]
