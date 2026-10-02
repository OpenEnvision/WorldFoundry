"""
UniCeption Alternating-Attention Transformer for Information Sharing
"""

from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from worldfoundry.core.model_loading.checkpoints import load_tensor_state_dict
import torch.nn as nn

from uniception.models.info_sharing.base import (
    MultiViewTransformerInput,
    MultiViewTransformerOutput,
    UniCeptionInfoSharingBase,
)
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner, feature_take_indices
from uniception.models.utils.positional_encoding import PositionGetter
from uniception.models.utils.transformer_blocks import Mlp, SelfAttentionBlock


class MultiViewAlternatingAttentionTransformer(UniCeptionInfoSharingBase):
    "UniCeption Multi-View Alternating-Attention Transformer for information sharing across image features from different views."

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        distinguish_ref_and_non_ref_views: bool = True,
        use_pe_for_non_reference_views: bool = False,
        max_num_views_for_pe: int = 1000,
        use_rand_idx_pe_for_non_reference_views: bool = True,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: Type[nn.Module] = Mlp,
        custom_positional_encoding: Optional[Callable] = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Alternating-Attention Transformer for information sharing across image features from different views.
        Alternates between global and frame-level attention.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            distinguish_ref_and_non_ref_views (bool): Whether to identify/distinguish reference and non-reference views by using positional encoding. (default: True)
            use_pe_for_non_reference_views (bool): Whether to use view positional encoding for input non-reference views. Only used if distinguish_ref_and_non_ref_views flag is True. (default: False)
            max_num_views_for_pe (int): Maximum number of views for positional encoding. (default: 1000)
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views. (default: True)
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: True)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory efficiency. (default: False)
        """
        # Initialize the base class
        super().__init__(name=name, size=size, *args, **kwargs)

        # Initialize the specific attributes of the transformer
        self.input_embed_dim = input_embed_dim
        self.distinguish_ref_and_non_ref_views = distinguish_ref_and_non_ref_views
        self.use_pe_for_non_reference_views = use_pe_for_non_reference_views
        self.max_num_views_for_pe = max_num_views_for_pe
        self.use_rand_idx_pe_for_non_reference_views = use_rand_idx_pe_for_non_reference_views
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.init_values = init_values
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.mlp_layer = mlp_layer
        self.custom_positional_encoding = custom_positional_encoding
        self.use_scalable_softmax = use_scalable_softmax
        self.use_entropy_scaling = use_entropy_scaling
        self.base_token_count_for_entropy_scaling = base_token_count_for_entropy_scaling
        self.entropy_scaling_growth_factor = entropy_scaling_growth_factor
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize the projection layer for input embeddings
        if self.input_embed_dim != self.dim:
            self.proj_embed = nn.Linear(self.input_embed_dim, self.dim, bias=True)
        else:
            self.proj_embed = nn.Identity()

        # Initialize the self-attention blocks which ingest all views at once
        self.self_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    proj_drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    init_values=self.init_values,
                    drop_path=self.drop_path,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    mlp_layer=self.mlp_layer,
                    custom_positional_encoding=self.custom_positional_encoding,
                    use_scalable_softmax=self.use_scalable_softmax,
                    use_entropy_scaling=self.use_entropy_scaling,
                    base_token_count_for_entropy_scaling=self.base_token_count_for_entropy_scaling,
                    entropy_scaling_growth_factor=self.entropy_scaling_growth_factor,
                )
                for _ in range(self.depth)
            ]
        )

        # Initialize the final normalization layer
        self.norm = self.norm_layer(self.dim)

        # Initialize the position getter for patch positions if required
        if self.custom_positional_encoding is not None:
            self.position_getter = PositionGetter()

        # Initialize the positional encoding table for the views if required
        if self.distinguish_ref_and_non_ref_views:
            if self.use_pe_for_non_reference_views:
                # Initialize the positional encoding table for the different views
                self.register_buffer(
                    "view_pos_table",
                    self._get_sinusoid_encoding_table(self.max_num_views_for_pe, self.dim, 10000),
                )
            else:
                # Initialize the positional encoding table for the reference view
                self.register_buffer(
                    "view_pos_table",
                    self._get_sinusoid_encoding_table(1, self.dim, 10000),
                )

        # Initialize random weights
        self.initialize_weights()

        # Apply gradient checkpointing if enabled
        if self.gradient_checkpointing:
            for i, block in enumerate(self.self_attention_blocks):
                self.self_attention_blocks[i] = self.wrap_module_with_gradient_checkpointing(block)

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained multi-view Alternating-Attention transformer weights from {self.pretrained_checkpoint_path} ..."
            )
            state_dict = load_tensor_state_dict(self.pretrained_checkpoint_path, wrapper_keys=("model",))
            print(self.load_state_dict(state_dict))

    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        "Sinusoid position encoding table"

        def get_position_angle_vec(position):
            return [position / np.power(base, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)

    def initialize_weights(self):
        "Initialize weights of the transformer."
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        "Initialize the transformer linear and layer norm weights."
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> MultiViewTransformerOutput:
        """
        Forward interface for the Multi-View Alternating-Attention Transformer.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.
                Optionally, the input can also include:
                - additional_input_tokens: Global additional tokens (e.g., scale token)
                  which are appended to the token set from all multi-view features and only participate in global-level attention.
                  Shape: (batch, input_embed_dim, num_of_additional_tokens).
                - additional_input_tokens_per_view: Per-view additional tokens (e.g., view-specific tokens like per-view registers)
                  which are appended to each view's token set and participate in both frame-level and global-level attention.
                  List of tensors, each of shape (batch, input_embed_dim, num_of_additional_tokens_per_view).

        Returns:
            MultiViewTransformerOutput: Output of the model post information sharing.
                Returns view features and optionally additional_token_features (global) and additional_token_features_per_view (per-view).
        """
        # Check that the number of views matches the input and the features are of expected shape
        if self.use_pe_for_non_reference_views:
            assert (
                len(model_input.features) <= self.max_num_views_for_pe
            ), f"Expected less than {self.max_num_views_for_pe} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Process per-view additional tokens if provided
        num_of_additional_tokens_per_view = 0
        if model_input.additional_input_tokens_per_view is not None:
            additional_tokens_per_view = model_input.additional_input_tokens_per_view
            assert len(additional_tokens_per_view) == num_of_views, (
                f"Number of additional token tensors ({len(additional_tokens_per_view)}) "
                f"must match number of views ({num_of_views})"
            )
            assert all(
                tokens.ndim == 3 for tokens in additional_tokens_per_view
            ), "Additional tokens per view must have 3 dimensions (N, C, T)"
            assert all(
                tokens.shape[1] == self.input_embed_dim for tokens in additional_tokens_per_view
            ), f"Additional tokens per view must have input dimension {self.input_embed_dim}"
            assert all(
                tokens.shape[0] == batch_size for tokens in additional_tokens_per_view
            ), "Batch size mismatch for additional tokens per view"

            num_of_additional_tokens_per_view = additional_tokens_per_view[0].shape[2]

            # Concatenate per-view additional tokens to each view's features
            multi_view_features_with_tokens = []
            for view_idx, (view_features, view_tokens) in enumerate(
                zip(multi_view_features, additional_tokens_per_view)
            ):
                # view_features: (N, C, H, W)
                # view_tokens: (N, C, T)
                # Flatten spatial dimensions: (N, C, H, W) -> (N, C, H*W)
                view_features_flat = view_features.reshape(batch_size, self.input_embed_dim, height * width)
                # Concatenate tokens: (N, C, H*W + T)
                view_with_tokens = torch.cat([view_features_flat, view_tokens], dim=2)
                multi_view_features_with_tokens.append(view_with_tokens)

            # Stack all views: (N, V, C, H*W + T)
            multi_view_features = torch.stack(multi_view_features_with_tokens, dim=1)
            # Permute to (N, V, H*W + T, C)
            multi_view_features = multi_view_features.permute(0, 1, 3, 2)
            # Reshape to (N, V * (H*W + T), C)
            multi_view_features = multi_view_features.reshape(
                batch_size,
                num_of_views * (height * width + num_of_additional_tokens_per_view),
                self.input_embed_dim,
            ).contiguous()

            # Update tokens per view to include additional tokens
            num_of_tokens_per_view = height * width + num_of_additional_tokens_per_view
        else:
            # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
            multi_view_features = torch.stack(multi_view_features, dim=1)

            # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
            multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
            multi_view_features = multi_view_features.reshape(
                batch_size, num_of_views * height * width, self.input_embed_dim
            ).contiguous()

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:
            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Raise error if custom positional encoding is used with additional tokens
        if self.custom_positional_encoding is not None:
            if (
                model_input.additional_input_tokens is not None
                or model_input.additional_input_tokens_per_view is not None
            ):
                raise ValueError(
                    "Custom positional encoding is not supported when additional_input_tokens or "
                    "additional_input_tokens_per_view are provided. Please set custom_positional_encoding=None "
                    "or remove additional tokens from the input."
                )

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:
            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[1]
            multi_view_positions = multi_view_positions + additional_tokens_positions

        if self.distinguish_ref_and_non_ref_views:
            # Add positional encoding for reference view (idx 0)
            ref_view_pe = self.view_pos_table[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
        else:
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]

        if self.distinguish_ref_and_non_ref_views and self.use_pe_for_non_reference_views:
            # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
            if self.use_rand_idx_pe_for_non_reference_views:
                non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views_for_pe, size=(num_of_views - 1,))
            else:
                non_ref_view_pe_indices = torch.arange(1, num_of_views)
            non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
            non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
            non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
            non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            non_ref_view_features = non_ref_view_features + non_ref_view_pe
        else:
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]

        # Concatenate the reference and non-reference view features
        # Handle additional tokens (no view-based positional encoding for them)
        if model_input.additional_input_tokens is not None:
            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features, additional_features], dim=1)
        else:
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            if depth_idx % 2 == 0:
                # Apply the self-attention block and update the multi-view features
                # Global attention across all views
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)
            else:
                # Handle additional tokens separately for frame-level attention
                additional_features = None
                additional_positions = None
                if model_input.additional_input_tokens is not None:
                    # Extract additional token features
                    additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
                    # Keep only view features for frame-level attention
                    multi_view_features = multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

                    # Handle positions for additional tokens if custom positional encoding is used
                    if self.custom_positional_encoding is not None:
                        additional_positions = multi_view_positions[:, num_of_views * num_of_tokens_per_view :, :]
                        multi_view_positions = multi_view_positions[:, : num_of_views * num_of_tokens_per_view, :]

                # Reshape the multi-view features from (N, V * (H*W + T_per_view), C) to (N * V, (H*W + T_per_view), C)
                # Note: When T_per_view = 0 (no per-view tokens), this is (N, V * H*W, C) to (N * V, H*W, C)
                multi_view_features = multi_view_features.reshape(
                    batch_size * num_of_views, num_of_tokens_per_view, self.dim
                ).contiguous()  # (N * V, (H*W + T_per_view), C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size * num_of_views, num_of_tokens_per_view, 2
                    ).contiguous()  # (N * V, (H*W + T_per_view), 2)

                # Apply the self-attention block and update the multi-view features
                # Frame-level attention within each view (including per-view additional tokens if provided)
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)

                # Reshape the multi-view features from (N * V, (H*W + T_per_view), C) back to (N, V * (H*W + T_per_view), C)
                multi_view_features = multi_view_features.reshape(
                    batch_size, num_of_views * num_of_tokens_per_view, self.dim
                ).contiguous()  # (N, V * (H*W + T_per_view), C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size, num_of_views * num_of_tokens_per_view, 2
                    ).contiguous()  # (N, V * (H*W + T_per_view), 2)

                # Reattach additional tokens if they exist
                if additional_features is not None:
                    multi_view_features = torch.cat([multi_view_features, additional_features], dim=1)
                    # Reattach positions for additional tokens if they exist
                    if additional_positions is not None:
                        multi_view_positions = torch.cat([multi_view_positions, additional_positions], dim=1)

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Extract view features (excluding global additional tokens)
        view_features_flat = output_multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

        # Extract per-view additional tokens if they were provided
        additional_token_features_per_view = None
        if model_input.additional_input_tokens_per_view is not None:
            # Reshape to (N, V, H*W + T_per_view, C)
            view_features_with_tokens = view_features_flat.reshape(
                batch_size, num_of_views, num_of_tokens_per_view, self.dim
            )

            # Split into spatial features and per-view additional tokens
            spatial_tokens_per_view = height * width
            view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
            per_view_additional = view_features_with_tokens[:, :, spatial_tokens_per_view:, :]  # (N, V, T_per_view, C)

            # Reshape view features to (N, V, H, W, C)
            view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

            # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
            per_view_additional = per_view_additional.split(1, dim=1)
            additional_token_features_per_view = [
                tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                for tokens in per_view_additional
            ]
        else:
            # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
            view_features = view_features_flat.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split the output multi-view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

        # Extract and return additional token features (global) if provided
        additional_token_features = None
        if model_input.additional_input_tokens is not None:
            additional_token_features = output_multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

        return MultiViewTransformerOutput(
            features=view_features,
            additional_token_features=additional_token_features,
            additional_token_features_per_view=additional_token_features_per_view,
        )


class MultiViewAlternatingAttentionTransformerIFR(
    MultiViewAlternatingAttentionTransformer, IntermediateFeatureReturner
):
    "Intermediate Feature Returner for UniCeption Multi-View Alternating-Attention Transformer"

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        distinguish_ref_and_non_ref_views: bool = True,
        use_pe_for_non_reference_views: bool = False,
        max_num_views_for_pe: int = 1000,
        use_rand_idx_pe_for_non_reference_views: bool = True,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: str = None,
        indices: Optional[Union[int, List[int]]] = None,
        norm_intermediate: bool = True,
        intermediates_only: bool = False,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Alternating-Attention Transformer for information sharing across image features from different views.
        Extends the base class to return intermediate features.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            distinguish_ref_and_non_ref_views (bool): Whether to identify/distinguish reference and non-reference views by using positional encoding. (default: True)
            use_pe_for_non_reference_views (bool): Whether to use view positional encoding for input non-reference views. Only used if distinguish_ref_and_non_ref_views flag is True. (default: False)
            max_num_views_for_pe (int): Maximum number of views for positional encoding. (default: 1000)
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views. (default: True)
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. (default: None) Options:
            - None: Return all intermediate layers.
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. (default: True)
            intermediates_only (bool, optional): Whether to return only the intermediate features. (default: False)
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory efficiency. (default: False)
        """
        # Init the base classes
        MultiViewAlternatingAttentionTransformer.__init__(
            self,
            name=name,
            input_embed_dim=input_embed_dim,
            distinguish_ref_and_non_ref_views=distinguish_ref_and_non_ref_views,
            use_pe_for_non_reference_views=use_pe_for_non_reference_views,
            max_num_views_for_pe=max_num_views_for_pe,
            use_rand_idx_pe_for_non_reference_views=use_rand_idx_pe_for_non_reference_views,
            size=size,
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
            custom_positional_encoding=custom_positional_encoding,
            use_scalable_softmax=use_scalable_softmax,
            use_entropy_scaling=use_entropy_scaling,
            base_token_count_for_entropy_scaling=base_token_count_for_entropy_scaling,
            entropy_scaling_growth_factor=entropy_scaling_growth_factor,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
            intermediates_only=intermediates_only,
        )

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]],]:
        """
        Forward interface for the Multi-View Alternating-Attention Transformer with Intermediate Feature Return.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.
                Optionally, the input can also include:
                - additional_input_tokens: Global additional tokens (e.g., scale token)
                  which are appended to the token set from all multi-view features and only participate in global-level attention.
                  Shape: (batch, input_embed_dim, num_of_additional_tokens).
                - additional_input_tokens_per_view: Per-view additional tokens (e.g., view-specific tokens like per-view registers)
                  which are appended to each view's token set and participate in both frame-level and global-level attention.
                  List of tensors, each of shape (batch, input_embed_dim, num_of_additional_tokens_per_view).

        Returns:
            Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]]]:
                Output of the model post information sharing.
                If intermediates_only is True, returns a list of intermediate outputs containing:
                    - features: List of view features
                    - additional_token_features: Global additional token features (if provided)
                    - additional_token_features_per_view: Per-view additional token features (if provided)
                If intermediates_only is False, returns a tuple of:
                    - Final output (MultiViewTransformerOutput) with features, additional_token_features, and additional_token_features_per_view
                    - List of intermediate outputs (MultiViewTransformerOutput) from specified layers
        """
        # Check that the number of views matches the input and the features are of expected shape
        if self.use_pe_for_non_reference_views:
            assert (
                len(model_input.features) <= self.max_num_views_for_pe
            ), f"Expected less than {self.max_num_views_for_pe} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Get the indices of the intermediate features to return
        intermediate_multi_view_features = []
        take_indices, _ = feature_take_indices(self.depth, self.indices)

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Process per-view additional tokens if provided
        num_of_additional_tokens_per_view = 0
        if model_input.additional_input_tokens_per_view is not None:
            additional_tokens_per_view = model_input.additional_input_tokens_per_view
            assert len(additional_tokens_per_view) == num_of_views, (
                f"Number of additional token tensors ({len(additional_tokens_per_view)}) "
                f"must match number of views ({num_of_views})"
            )
            assert all(
                tokens.ndim == 3 for tokens in additional_tokens_per_view
            ), "Additional tokens per view must have 3 dimensions (N, C, T)"
            assert all(
                tokens.shape[1] == self.input_embed_dim for tokens in additional_tokens_per_view
            ), f"Additional tokens per view must have input dimension {self.input_embed_dim}"
            assert all(
                tokens.shape[0] == batch_size for tokens in additional_tokens_per_view
            ), "Batch size mismatch for additional tokens per view"

            num_of_additional_tokens_per_view = additional_tokens_per_view[0].shape[2]

            # Concatenate per-view additional tokens to each view's features
            multi_view_features_with_tokens = []
            for view_idx, (view_features, view_tokens) in enumerate(
                zip(multi_view_features, additional_tokens_per_view)
            ):
                # view_features: (N, C, H, W)
                # view_tokens: (N, C, T)
                # Flatten spatial dimensions: (N, C, H, W) -> (N, C, H*W)
                view_features_flat = view_features.reshape(batch_size, self.input_embed_dim, height * width)
                # Concatenate tokens: (N, C, H*W + T)
                view_with_tokens = torch.cat([view_features_flat, view_tokens], dim=2)
                multi_view_features_with_tokens.append(view_with_tokens)

            # Stack all views: (N, V, C, H*W + T)
            multi_view_features = torch.stack(multi_view_features_with_tokens, dim=1)
            # Permute to (N, V, H*W + T, C)
            multi_view_features = multi_view_features.permute(0, 1, 3, 2)
            # Reshape to (N, V * (H*W + T), C)
            multi_view_features = multi_view_features.reshape(
                batch_size,
                num_of_views * (height * width + num_of_additional_tokens_per_view),
                self.input_embed_dim,
            ).contiguous()

            # Update tokens per view to include additional tokens
            num_of_tokens_per_view = height * width + num_of_additional_tokens_per_view
        else:
            # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
            multi_view_features = torch.stack(multi_view_features, dim=1)

            # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
            multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
            multi_view_features = multi_view_features.reshape(
                batch_size, num_of_views * height * width, self.input_embed_dim
            ).contiguous()

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:
            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Raise error if custom positional encoding is used with additional tokens
        if self.custom_positional_encoding is not None:
            if (
                model_input.additional_input_tokens is not None
                or model_input.additional_input_tokens_per_view is not None
            ):
                raise ValueError(
                    "Custom positional encoding is not supported when additional_input_tokens or "
                    "additional_input_tokens_per_view are provided. Please set custom_positional_encoding=None "
                    "or remove additional tokens from the input."
                )

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:
            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[1]
            multi_view_positions = multi_view_positions + additional_tokens_positions

        if self.distinguish_ref_and_non_ref_views:
            # Add positional encoding for reference view (idx 0)
            ref_view_pe = self.view_pos_table[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
        else:
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]

        if self.distinguish_ref_and_non_ref_views and self.use_pe_for_non_reference_views:
            # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
            if self.use_rand_idx_pe_for_non_reference_views:
                non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views_for_pe, size=(num_of_views - 1,))
            else:
                non_ref_view_pe_indices = torch.arange(1, num_of_views)
            non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
            non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
            non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
            non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            non_ref_view_features = non_ref_view_features + non_ref_view_pe
        else:
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]

        # Concatenate the reference and non-reference view features
        # Handle additional tokens (no view-based positional encoding for them)
        if model_input.additional_input_tokens is not None:
            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features, additional_features], dim=1)
        else:
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            if depth_idx % 2 == 0:
                # Apply the self-attention block and update the multi-view features
                # Global attention across all views
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)
            else:
                # Handle additional tokens separately for frame-level attention
                additional_features = None
                additional_positions = None
                if model_input.additional_input_tokens is not None:
                    # Extract additional token features
                    additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
                    # Keep only view features for frame-level attention
                    multi_view_features = multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

                    # Handle positions for additional tokens if custom positional encoding is used
                    if self.custom_positional_encoding is not None:
                        additional_positions = multi_view_positions[:, num_of_views * num_of_tokens_per_view :, :]
                        multi_view_positions = multi_view_positions[:, : num_of_views * num_of_tokens_per_view, :]

                # Reshape the multi-view features from (N, V * (H*W + T_per_view), C) to (N * V, (H*W + T_per_view), C)
                # Note: When T_per_view = 0 (no per-view tokens), this is (N, V * H*W, C) to (N * V, H*W, C)
                multi_view_features = multi_view_features.reshape(
                    batch_size * num_of_views, num_of_tokens_per_view, self.dim
                ).contiguous()  # (N * V, (H*W + T_per_view), C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size * num_of_views, num_of_tokens_per_view, 2
                    ).contiguous()  # (N * V, (H*W + T_per_view), 2)

                # Apply the self-attention block and update the multi-view features
                # Frame-level attention within each view (including per-view additional tokens if provided)
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)

                # Reshape the multi-view features from (N * V, (H*W + T_per_view), C) back to (N, V * (H*W + T_per_view), C)
                multi_view_features = multi_view_features.reshape(
                    batch_size, num_of_views * num_of_tokens_per_view, self.dim
                ).contiguous()  # (N, V * (H*W + T_per_view), C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size, num_of_views * num_of_tokens_per_view, 2
                    ).contiguous()  # (N, V * (H*W + T_per_view), 2)

                # Reattach additional tokens if they exist
                if additional_features is not None:
                    multi_view_features = torch.cat([multi_view_features, additional_features], dim=1)
                    # Reattach positions for additional tokens if they exist
                    if additional_positions is not None:
                        multi_view_positions = torch.cat([multi_view_positions, additional_positions], dim=1)
            if depth_idx in take_indices:
                # Normalize the intermediate features with final norm layer if enabled
                intermediate_multi_view_features.append(
                    self.norm(multi_view_features) if self.norm_intermediate else multi_view_features
                )

        # Reshape the intermediate features and convert to MultiViewTransformerOutput class
        for idx in range(len(intermediate_multi_view_features)):
            # Get the current intermediate features
            current_features = intermediate_multi_view_features[idx]

            # Extract view features (excluding global additional tokens)
            view_features_flat = current_features[:, : num_of_views * num_of_tokens_per_view, :]

            # Extract per-view additional tokens if they were provided
            additional_token_features_per_view = None
            if model_input.additional_input_tokens_per_view is not None:
                # Reshape to (N, V, H*W + T_per_view, C)
                view_features_with_tokens = view_features_flat.reshape(
                    batch_size, num_of_views, num_of_tokens_per_view, self.dim
                )

                # Split into spatial features and per-view additional tokens
                spatial_tokens_per_view = height * width
                view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
                per_view_additional = view_features_with_tokens[
                    :, :, spatial_tokens_per_view:, :
                ]  # (N, V, T_per_view, C)

                # Reshape view features to (N, V, H, W, C)
                view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
                view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

                # Split view features into separate views
                view_features = view_features.split(1, dim=1)
                view_features = [
                    intermediate_view_features.squeeze(dim=1) for intermediate_view_features in view_features
                ]

                # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
                per_view_additional = per_view_additional.split(1, dim=1)
                additional_token_features_per_view = [
                    tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                    for tokens in per_view_additional
                ]
            else:
                # Reshape the intermediate multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
                view_features = view_features_flat.reshape(
                    batch_size, num_of_views, height, width, self.dim
                )  # (N, V, H, W, C)
                view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

                # Split the intermediate multi-view features into separate views
                view_features = view_features.split(1, dim=1)
                view_features = [
                    intermediate_view_features.squeeze(dim=1) for intermediate_view_features in view_features
                ]

            # Extract and return additional token features (global) if provided
            additional_token_features = None
            if model_input.additional_input_tokens is not None:
                additional_token_features = current_features[:, num_of_views * num_of_tokens_per_view :, :]
                additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

            intermediate_multi_view_features[idx] = MultiViewTransformerOutput(
                features=view_features,
                additional_token_features=additional_token_features,
                additional_token_features_per_view=additional_token_features_per_view,
            )

        # Return only the intermediate features if enabled
        if self.intermediates_only:
            return intermediate_multi_view_features

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Extract view features (excluding global additional tokens)
        view_features_flat = output_multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

        # Extract per-view additional tokens if they were provided
        additional_token_features_per_view = None
        if model_input.additional_input_tokens_per_view is not None:
            # Reshape to (N, V, H*W + T_per_view, C)
            view_features_with_tokens = view_features_flat.reshape(
                batch_size, num_of_views, num_of_tokens_per_view, self.dim
            )

            # Split into spatial features and per-view additional tokens
            spatial_tokens_per_view = height * width
            view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
            per_view_additional = view_features_with_tokens[:, :, spatial_tokens_per_view:, :]  # (N, V, T_per_view, C)

            # Reshape view features to (N, V, H, W, C)
            view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

            # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
            per_view_additional = per_view_additional.split(1, dim=1)
            additional_token_features_per_view = [
                tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                for tokens in per_view_additional
            ]
        else:
            # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
            view_features = view_features_flat.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split the output multi-view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

        # Extract and return additional token features (global) if provided
        additional_token_features = None
        if model_input.additional_input_tokens is not None:
            additional_token_features = output_multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

        output_multi_view_features = MultiViewTransformerOutput(
            features=view_features,
            additional_token_features=additional_token_features,
            additional_token_features_per_view=additional_token_features_per_view,
        )

        return output_multi_view_features, intermediate_multi_view_features


def dummy_positional_encoding(x, xpos):
    "Dummy function for positional encoding of tokens"
    x = x
    xpos = xpos
    return x


def test_reshape_for_frame_attention():
    "Test the reshape function for frame-level attention in the Alternating Attention Transformer"
    batch_size = 2
    num_of_views = 3
    height = width = 2
    dim = 4
    num_of_tokens_per_view = height * width

    # Create tensor with recognizable pattern
    x = torch.zeros(batch_size, num_of_views * num_of_tokens_per_view, dim)
    for b in range(batch_size):
        for v in range(num_of_views):
            for h in range(height):
                for w in range(width):
                    token_idx = v * num_of_tokens_per_view + h * width + w
                    x[b, token_idx] = torch.tensor([b, v, h, w])

    # Apply reshape
    reshaped = x.reshape(batch_size * num_of_views, num_of_tokens_per_view, dim).contiguous()

    # Verify shape
    assert reshaped.shape == (batch_size * num_of_views, num_of_tokens_per_view, dim)

    # Verify content (check a few values)
    for b in range(batch_size):
        for v in range(num_of_views):
            for h in range(height):
                for w in range(width):
                    batch_view_idx = b * num_of_views + v
                    token_idx = h * width + w
                    expected = torch.tensor([b, v, h, w])
                    assert torch.all(reshaped[batch_view_idx, token_idx] == expected)

    # Verify reshape back works
    back_to_original = reshaped.reshape(batch_size, num_of_views * num_of_tokens_per_view, dim)
    assert torch.all(x == back_to_original)

    print("Reshape test passed!")
