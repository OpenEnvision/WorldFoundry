import json
from dataclasses import dataclass

from .model.factorization_utils import nth_root


@dataclass
class GenieConfig:
    num_layers: int
    num_heads: int
    d_model: int
    T: int = 12  # temporal sequence length
    S: int = 256  # spatial sequence length, e.g. 256 for 16x16
    image_vocab_size: int = 262144  # image_vocab_size: number of distinct image tokens;
    # actual model vocab size is larger to include special (e.g. mask) tokens.
    use_mup: bool = False
    dataloader_apply_mask: bool = True  # apply mask in dataloader
    dataloader_apply_corruption: bool = True
    dataloader_mask_ratio_min: float = 0.2
    drop_action_ratio: float = 0.0  # for datasets
    arch: str = "STTransformerDecoder"
    random_dummy_action: bool = True  # for model

    # Factorization for large vocabs (e.g. Open-MAGVIT2)
    num_factored_vocabs: int = 1
    factored_vocab_size: int = None

    # MaskGIT training (all arbitrary numbers)
    max_corrupt_rate: float = 0.2  # Corrupt all tokens, uniform between [0, max_corrupt_rate]
    # Case 1: MLM training.
    # Case 2: Not standard MLM, `non_mlm`. Some earlier frames are left unmasked, as in Copilot4D.
    non_mlm_ratio: float = 0.2
    num_prompt_frames: int = 4

    # action related
    init_actions: bool = False
    d_action: int = 28  # action dimensions
    use_actions: bool = True
    action_domains: list[str] = None
    d_actions: list[int] = None
    action_stats: list[list[float]] = None  # TODO: is this actually three nested lists?
    action_network: str = "mlp"
    shared_action_mlps: bool = True
    action_contrastive_loss: bool = False
    jointly_predict_actions: bool = False  # jointly predict actions
    jointly_predict_states: bool = True  # jointly predict states
    action_token_size: int = 64  # images are 16x16
    label_drop_prob: float = 0.5  # the drop ratio for action tokens
    action_loss_weight: float = 0.5  # weight for action loss

    # Attention
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    qk_norm: bool = True

    # MLP
    mlp_ratio: float = 4.0
    mlp_drop: float = 0.0
    mlp_bias: bool = True

    @classmethod
    def from_pretrained(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)

        return cls(**config)

    def shallow_copy(self):
        return GenieConfig(**vars(self))

    def __post_init__(self):
        if self.image_vocab_size is None:
            self.factored_vocab_size = 64  # dummy
        else:
            self.factored_vocab_size = nth_root(self.image_vocab_size, self.num_factored_vocabs)


@dataclass
class DiffusionGenieConfig(GenieConfig):
    Diffusion: bool = True

    # Attention
    dim: int = 512
    dataloader_apply_mask: bool = True  # apply mask inside the model
    dataloader_apply_corruption: bool = False  # no need for random corruptions
    dataloader_mask_ratio_min: float = 0.1

    # MLP
    vae_stride: int = 1
    patch_size: int = 1
    vae_embed_dim: int = 4
    mask_ratio_min: float = 0.7
    # Upstream assigns this field twice (0.1, then 0.5); the later 0.5 is the effective default.
    label_drop_prob: float = 0.5  # the drop ratio for action tokens
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    buffer_size: int = 64
    diffloss_d: int = 4
    diffloss_w: int = 1024  # 1024
    num_sampling_steps: str = "100"
    diffusion_batch_mul: int = 1
    grad_checkpointing: bool = False
    use_actions: bool = True
    jointly_predict_actions: bool = False  # jointly predict actions
    jointly_predict_states: bool = True  # jointly predict states
    action_token_size: int = 64  # images are 16x16
    action_loss_weight: float = 1.0  # weight for action loss
    predict_unmask: bool = False  # also predict tokens in unmasked regions
    maskgit_steps: int = 16  # the mask iterations during inference

    def shallow_copy(self):
        return DiffusionGenieConfig(**vars(self))
