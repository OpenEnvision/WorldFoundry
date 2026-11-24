from .clip import clip_b16 as clip_b16
from .clip import clip_l14 as clip_l14
from .clip import clip_l14_336 as clip_l14_336

# from .modeling_finetune import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224, vit_large_patch16_384
from .modeling_finetune import vit_large_patch16_224 as vit_large_patch16_224
from .modeling_pretrain import (
    pretrain_videomae_base_patch16_224 as pretrain_videomae_base_patch16_224,
)
from .modeling_pretrain import (
    pretrain_videomae_huge_patch16_224 as pretrain_videomae_huge_patch16_224,
)
from .modeling_pretrain import (
    pretrain_videomae_large_patch16_224 as pretrain_videomae_large_patch16_224,
)
from .modeling_pretrain_umt import pretrain_umt_base_patch16_224 as pretrain_umt_base_patch16_224
from .modeling_pretrain_umt import pretrain_umt_large_patch16_224 as pretrain_umt_large_patch16_224
