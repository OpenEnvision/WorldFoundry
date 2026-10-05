"""
UniCeption Cross-Attention Transformer for Information Sharing
"""

from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

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
from uniception.models.utils.transformer_blocks import CrossAttentionBlock, Mlp


class MultiViewCrossAttentionTransformer(UniCeptionInfoSharingBase):
    "UniCeption Multi-View Cross-Attention Transformer for information sharing across image features from different views."

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        num_views: int,
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
        norm_cross_tokens: bool = True,
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
        Initialize the Multi-View Cross-Attention Transformer for information sharing across image features from different views.
        Creates a cross-attention transformer with multiple branches for each view.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            num_views (int): Number of views (input feature sets).
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
            norm_cross_tokens (bool): Whether to normalize cross tokens (default: True)
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
        self.num_views = num_views
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
        self.norm_cross_tokens = norm_cross_tokens
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

        # Initialize the cross-attention blocks for a single view
        cross_attention_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
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
                    norm_cross_tokens=self.norm_cross_tokens,
                    use_scalable_softmax=self.use_scalable_softmax,
                    use_entropy_scaling=self.use_entropy_scaling,
                    base_token_count_for_entropy_scaling=self.base_token_count_for_entropy_scaling,
                    entropy_scaling_growth_factor=self.entropy_scaling_growth_factor,
                )
                for _ in range(self.depth)
            ]
        )

        # Copy the cross-attention blocks for all other views
        self.multi_view_branches = nn.ModuleList([cross_attention_blocks])
        for _ in range(1, self.num_views):
            self.multi_view_branches.append(deepcopy(cross_attention_blocks))

        # Initialize the final normalization layer
        self.norm = self.norm_layer(self.dim)

        # Initialize the position getter for patch positions if required
        if self.custom_positional_encoding is not None:
            self.position_getter = PositionGetter()

        # Initialize random weights
        self.initialize_weights()

        # Apply gradient checkpointing if enabled
        if self.gradient_checkpointing:
            for i, block in enumerate(self.cross_attention_blocks):
                self.cross_attention_blocks[i] = self.wrap_module_with_gradient_checkpointing(block)

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained multi-view cross-attention transformer weights from {self.pretrained_checkpoint_path} ..."
            )
            state_dict = load_tensor_state_dict(self.pretrained_checkpoint_path, wrapper_keys=("model",))
            print(self.load_state_dict(state_dict))

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
        Forward interface for the Multi-View Cross-Attention Transformer.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.

        Returns:
            MultiViewTransformerOutput: Output of the model post information sharing.
        """
        # Check that the number of views matches the input and the features are of expected shape
        assert (
            len(model_input.features) == self.num_views
        ), f"Expected {self.num_views} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Initialize the multi-view features from the model input
        multi_view_features = model_input.features

        # Resize the multi-view features from NCHW to NLC
        batch_size, _, height, width = multi_view_features[0].shape
        multi_view_features = [
            view_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.input_embed_dim).contiguous()
            for view_features in multi_view_features
        ]

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, view_features.device)
                for view_features in multi_view_features
            ]
        else:
            multi_view_positions = [None] * self.num_views

        # Project input features to the transformer dimension
        multi_view_features = [self.proj_embed(view_features) for view_features in multi_view_features]

        # Pass through each view's cross-attention blocks
        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            updated_multi_view_features = []
            # Loop over each view
            for view_idx, view_features in enumerate(multi_view_features):
                # Get all the other views
                other_views_features = [multi_view_features[i] for i in range(self.num_views) if i != view_idx]
                # Concatenate all the tokens from the other views
                other_views_features = torch.cat(other_views_features, dim=1)
                # Get the positions for the current view
                view_positions = multi_view_positions[view_idx]
                # Get the positions for all other views
                other_views_positions = (
                    torch.cat([multi_view_positions[i] for i in range(self.num_views) if i != view_idx], dim=1)
                    if view_positions is not None
                    else None
                )
                # Apply the cross-attention block and update the multi-view features
                updated_view_features = self.multi_view_branches[view_idx][depth_idx](
                    view_features, other_views_features, view_positions, other_views_positions
                )
                # Keep track of the updated view features
                updated_multi_view_features.append(updated_view_features)
            # Update the multi-view features for the next depth
            multi_view_features = updated_multi_view_features

        # Normalize the output features
        output_multi_view_features = [self.norm(view_features) for view_features in multi_view_features]

        # Resize the output multi-view features back to NCHW
        output_multi_view_features = [
            view_features.reshape(batch_size, height, width, self.dim).permute(0, 3, 1, 2).contiguous()
            for view_features in output_multi_view_features
        ]

        return MultiViewTransformerOutput(features=output_multi_view_features)


class MultiViewCrossAttentionTransformerIFR(MultiViewCrossAttentionTransformer, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption Multi-View Cross-Attention Transformer"

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        num_views: int,
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
        norm_cross_tokens: bool = True,
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
        Initialize the Multi-View Cross-Attention Transformer for information sharing across image features from different views.
        Creates a cross-attention transformer with multiple branches for each view.
        Extends the base class to return intermediate features.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            num_views (int): Number of views (input feature sets).
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
            norm_cross_tokens (bool): Whether to normalize cross tokens (default: True)
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
        MultiViewCrossAttentionTransformer.__init__(
            self,
            name=name,
            input_embed_dim=input_embed_dim,
            num_views=num_views,
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
            norm_cross_tokens=norm_cross_tokens,
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
        Forward interface for the Multi-View Cross-Attention Transformer with Intermediate Feature Return.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.

        Returns:
            Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]]]:
                Output of the model post information sharing.
                If intermediates_only is True, returns a list of intermediate outputs.
                If intermediates_only is False, returns a tuple of final output and a list of intermediate outputs.
        """
        # Check that the number of views matches the input and the features are of expected shape
        assert (
            len(model_input.features) == self.num_views
        ), f"Expected {self.num_views} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Get the indices of the intermediate features to return
        intermediate_multi_view_features = []
        take_indices, _ = feature_take_indices(self.depth, self.indices)

        # Initialize the multi-view features from the model input
        multi_view_features = model_input.features

        # Resize the multi-view features from NCHW to NLC
        batch_size, _, height, width = multi_view_features[0].shape
        multi_view_features = [
            view_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.input_embed_dim).contiguous()
            for view_features in multi_view_features
        ]

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, view_features.device)
                for view_features in multi_view_features
            ]
        else:
            multi_view_positions = [None] * self.num_views

        # Project input features to the transformer dimension
        multi_view_features = [self.proj_embed(view_features) for view_features in multi_view_features]

        # Pass through each view's cross-attention blocks
        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            updated_multi_view_features = []
            # Loop over each view
            for view_idx, view_features in enumerate(multi_view_features):
                # Get all the other views
                other_views_features = [multi_view_features[i] for i in range(self.num_views) if i != view_idx]
                # Concatenate all the tokens from the other views
                other_views_features = torch.cat(other_views_features, dim=1)
                # Get the positions for the current view
                view_positions = multi_view_positions[view_idx]
                # Get the positions for all other views
                other_views_positions = (
                    torch.cat([multi_view_positions[i] for i in range(self.num_views) if i != view_idx], dim=1)
                    if view_positions is not None
                    else None
                )
                # Apply the cross-attention block and update the multi-view features
                updated_view_features = self.multi_view_branches[view_idx][depth_idx](
                    view_features, other_views_features, view_positions, other_views_positions
                )
                # Keep track of the updated view features
                updated_multi_view_features.append(updated_view_features)
            # Update the multi-view features for the next depth
            multi_view_features = updated_multi_view_features
            # Append the intermediate features if required
            if depth_idx in take_indices:
                # Normalize the intermediate features with final norm layer if enabled
                intermediate_multi_view_features.append(
                    [self.norm(view_features) for view_features in multi_view_features]
                    if self.norm_intermediate
                    else multi_view_features
                )

        # Reshape the intermediate features and convert to MultiViewTransformerOutput class
        for idx in range(len(intermediate_multi_view_features)):
            intermediate_multi_view_features[idx] = [
                view_features.reshape(batch_size, height, width, self.dim).permute(0, 3, 1, 2).contiguous()
                for view_features in intermediate_multi_view_features[idx]
            ]
            intermediate_multi_view_features[idx] = MultiViewTransformerOutput(
                features=intermediate_multi_view_features[idx]
            )

        # Return only the intermediate features if enabled
        if self.intermediates_only:
            return intermediate_multi_view_features

        # Normalize the output features
        output_multi_view_features = [self.norm(view_features) for view_features in multi_view_features]

        # Resize the output multi-view features back to NCHW
        output_multi_view_features = [
            view_features.reshape(batch_size, height, width, self.dim).permute(0, 3, 1, 2).contiguous()
            for view_features in output_multi_view_features
        ]

        output_multi_view_features = MultiViewTransformerOutput(features=output_multi_view_features)

        return output_multi_view_features, intermediate_multi_view_features


def dummy_positional_encoding(x, xpos):
    "Dummy function for positional encoding of tokens"
    x = x
    xpos = xpos
    return x
