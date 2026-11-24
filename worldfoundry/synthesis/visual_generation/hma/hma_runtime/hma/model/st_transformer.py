import torch
from einops import rearrange
from torch import Tensor, nn

from .attention import SelfAttention


class Mlp(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.05,  # add dropout
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.05,
        # action relevant
        action_processing: str = "mlp",
        jointly_predict_actions: bool = False,
        mask_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # sequence dim is over each frame's 16x16 patch tokens
        self.spatial_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )

        # sequence dim is over time sequence (16)
        self.temporal_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )

        self.action_prediction = jointly_predict_actions
        self.action_processing = action_processing
        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.action_projectors = None  # set at run-time

    def forward(self, x_TSC: Tensor, action_ids: Tensor = None, domain=None) -> Tensor:
        """
        The main forward pass of the STBlock. It does action conditioning (with options),
        (bidrectional) spatial attention, (causal) temporal attention, and action masking.
        """
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, "B T S C -> (B T) S C")
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        # Process attention temporally
        x_TC = rearrange(x_SC, "(B T) S C -> (B S) T C", T=T)

        if action_ids is not None and domain is not None and self.action_projectors is not None:
            # action_ids: [B, T, D]. Only apply to video parts
            if "mlp" in self.action_processing:
                action_ids = self.action_projectors[domain](action_ids)  # does not depend on x_TC
                x_TC = rearrange(x_TC, "(B S) T C -> B S T C", S=S)
                x_TC = x_TC + action_ids[:, None, : x_TC.shape[2]]  # expand across spatial
                x_TC = rearrange(x_TC, "B S T C -> (B S) T C", S=S)

            elif "cross_attention" in self.action_processing:
                x_TC = x_TC + self.action_projectors[domain](x_TC, action_ids, action_ids)

            elif "modulate" in self.action_processing:
                x_TC = x_TC + self.action_projectors[domain](x_TC, action_ids)

        # Apply the Causal Transformer
        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)  # [256, 16, 256]
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        x_TSC = rearrange(x_TC, "(B S) T C -> B T S C", S=S)
        return x_TSC


class STTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
        # action relevant
        action_processing: str = "mlp",
        jointly_predict_actions: bool = False,
        random_dummy_action: bool = True,
        mask_token_id: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                STBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    qk_norm=qk_norm,
                    use_mup=use_mup,
                    attn_drop=attn_drop,
                    mlp_ratio=mlp_ratio,
                    mlp_bias=mlp_bias,
                    mlp_drop=mlp_drop,
                    action_processing=action_processing,
                    jointly_predict_actions=jointly_predict_actions,
                    mask_token_id=mask_token_id,
                )
                for _ in range(num_layers)
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Weight initialization for transformer
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, tgt, action_ids=None, domain=""):
        x = tgt
        for layer in self.layers:
            x = layer(x, action_ids=action_ids, domain=domain)

        return x
