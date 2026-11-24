"""
Not in original Open-MAGVIT2.
"""

from dataclasses import dataclass


@dataclass
class VQConfig:
    # Model Arch
    in_channels: int = 3
    z_channels: int = 18
    out_channels: int = 3
    base_channels: int = 128  # Initial hidden width, all subsequent blocks are multiples (`ch_mult`) of this width
    ch_mult: tuple[int] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2

    num_codebooks: int = 1
    codebook_size: int = 262144
    sample_minimization_weight: float = 1.0
    batch_maximization_weight: float = 1.0
    token_factorization: bool = False
