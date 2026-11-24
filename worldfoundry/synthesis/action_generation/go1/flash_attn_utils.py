"""Resolve an already-installed Flash Attention implementation, if available.

Inference is deliberately offline: this module never asks a kernel service to
download code.  All model implementations retain their eager attention path.
"""

import logging
from functools import cache

logger = logging.getLogger(__name__)


@cache
def load_flash_attn():
    """Return local Flash Attention symbols, or an empty dict."""
    symbols = _load_from_source()
    if symbols is not None:
        return symbols

    return {}


def _load_from_source():
    try:
        from flash_attn import (
            flash_attn_func,
            flash_attn_varlen_func,
            flash_attn_varlen_qkvpacked_func,
        )
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

        return {
            "flash_attn_func": flash_attn_func,
            "flash_attn_varlen_func": flash_attn_varlen_func,
            "flash_attn_varlen_qkvpacked_func": flash_attn_varlen_qkvpacked_func,
            "pad_input": pad_input,
            "unpad_input": unpad_input,
            "index_first_axis": index_first_axis,
        }
    except ImportError:
        return None
