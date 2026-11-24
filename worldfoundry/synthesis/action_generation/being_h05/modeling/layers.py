# Inference-only Being-H0.5 runtime retained in-tree.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.lvdm.blocks import nonlinearity as swish


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """

    def __init__(self, llm_dim: int, max_state_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.max_state_dim = max_state_dim

        self.fc1 = nn.Linear(self.max_state_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, max_state_dim)
        # print(proprio.shape)
        # print(self.max_state_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class InternVLConnector(nn.Module):
    def __init__(self, llm_hidden_size, vit_hidden_size, downsample_ratio):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def init_weights(self, state_dict, logger):
        if state_dict is None:
            raise NotImplementedError
        else:
            logger.info("Initializing connector from MLLM weights")
            for name, param in self.named_parameters():
                param.data.copy_(state_dict.pop(name).data)

        return state_dict

    def forward(self, x):
        return self.mlp1(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T) or (N,) for packed 1D inputs
        # Works for both 2D (batched sequences) and 1D (packed per-token) formats
        timesteps = timesteps.float()  # Ensure float type

        # NOTE: Removed explicit shape unpacking to support both 1D and 2D inputs
        # For 1D: (N,) -> unsqueeze(-1) -> (N, 1) -> broadcast with freqs
        # For 2D: (B, T) -> unsqueeze(-1) -> (B, T, 1) -> broadcast with freqs
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)

        return enc


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        # Handle both 2D (B, D_in) and 3D (B, T, D_in) inputs uniformly
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        output = torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

        if is_2d:
            output = output.squeeze(1)

        return output


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) -- one timestep per batch item
        cat_ids:   shape (B,)
        """
        B, T, _ = actions.shape

        # 1. Expand timesteps from (B,) to (B, T) to align with action sequence
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,)")

        # 2. Encode actions
        a_emb = self.W1(actions, cat_ids)

        # 3. Encode timesteps
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4. Concatenate and fuse
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5. Final projection
        x = self.W3(x, cat_ids)
        return x


class ActionEncoder(nn.Module):
    """
    Action encoder for Flow Matching.
    """

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Use standard linear layers instead of CategorySpecificLinear
        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # Concatenated action and time encoding
        self.W3 = nn.Linear(hidden_size, hidden_size)

        # Positional encoding module remains unchanged
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        Encode action sequences with timestep conditioning.

        Supports both scalar and per-token timesteps for training-time RTC.

        Args:
            actions (torch.Tensor): Action sequence
                - 2D: (B*Chunk, action_dim) - packed actions (Training-Time RTC)
                - 3D: (B, T, action_dim) - batched sequences
            timesteps (torch.Tensor): Timestep information
                - Per-token mode (2D actions, RTC): (B*Chunk,) - one per action token
                - Scalar mode (3D actions): (B,) - broadcast to all actions

        Returns:
            torch.Tensor: Encoded action features
                - 2D input: (B*Chunk, hidden_size)
                - 3D input: (B, T, hidden_size)
        """

        if actions.dim() == 2:
            # === 2D Input: (B*Chunk, action_dim) - Packed Actions ===
            # Used for Training-Time RTC with per-token timesteps
            B_times_Chunk, A_dim = actions.shape

            # Per-token timesteps (Training-Time RTC mode)
            if timesteps.dim() == 1 and timesteps.shape[0] == B_times_Chunk:
                # timesteps: (B*Chunk,) - one per action token
                a_emb = self.W1(actions)  # (B*Chunk, hidden_size)
                tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)  # (B*Chunk, hidden_size)

            else:
                raise ValueError(
                    f"For 2D actions with shape {actions.shape}, expected per-token timesteps "
                    f"with shape ({B_times_Chunk},), got {timesteps.shape}. "
                    f"Training-time RTC requires per-token timesteps."
                )

            # Concatenate and process
            x = torch.cat([a_emb, tau_emb], dim=-1)  # (B*Chunk, 2*hidden_size)
            x = swish(self.W2(x))  # (B*Chunk, hidden_size)
            x = self.W3(x)  # (B*Chunk, hidden_size)
            return x

        elif actions.dim() == 3:
            # === 3D Input: (B, T, action_dim) - Batched Sequences ===
            # Standard flow matching mode
            B, T, _ = actions.shape

            # 1. Expand timesteps from (B,) to (B, T) to align with action sequence
            if timesteps.dim() == 1 and timesteps.shape[0] == B:
                timesteps = timesteps.unsqueeze(1).expand(-1, T)  # (B,) -> (B, T)
            else:
                raise ValueError(f"Expected timesteps shape (B,) for 3D actions, got {timesteps.shape}")

            # 2. Encode actions (using standard linear layer)
            a_emb = self.W1(actions)

            # 3. Encode timesteps
            tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

            # 4. Concatenate and fuse (using standard linear layer)
            x = torch.cat([a_emb, tau_emb], dim=-1)
            x = swish(self.W2(x))

            # 5. Final projection (using standard linear layer)
            x = self.W3(x)
            return x

        else:
            raise ValueError(f"Actions must be 2D or 3D, got {actions.dim()}D")


# =============================================================================
# MPG: Manifold-Preserving Gating Enhancement Module
# =============================================================================


class SlicedWassersteinDistance(nn.Module):
    """
    Compute Sliced Wasserstein-2 distance between two distributions.

    The Sliced Wasserstein distance approximates the optimal transport cost
    by projecting distributions onto random 1D directions and computing the
    1D Wasserstein distance (which is just the L2 distance between sorted samples).

    Args:
        num_projections: Number of random projections (M). Higher = more accurate but slower.
                        Default: 32 (proven effective in OpenPI-NLL)

    Mathematical Formulation:
        For distributions mu and nu over R^D:
        SW_2(mu, nu) ~ (1/M) * sum_k W_2(theta_k#mu, theta_k#nu)^2

        where theta_k are random unit directions and theta_k# is the pushforward measure.
    """

    def __init__(self, num_projections: int = 32):
        super().__init__()
        self.num_projections = num_projections

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Compute Sliced Wasserstein-2 distance.

        Args:
            x: First distribution samples, shape (B, N, D)
            y: Second distribution samples, shape (B, M, D)
            seed: Optional random seed for deterministic projections

        Returns:
            Scalar Wasserstein distance (averaged over batch and projections)
        """
        batch_size, n_samples_x, feature_dim = x.shape
        _, n_samples_y, _ = y.shape
        device = x.device
        dtype = x.dtype

        # Set random seed if provided (for deterministic inference)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        total_transport_cost = 0.0

        for _k in range(self.num_projections):
            # Generate random unit direction theta_k
            direction = torch.randn(feature_dim, device=device, dtype=dtype, generator=generator)
            direction = direction / torch.norm(direction)  # Normalize to unit vector

            # Project features onto direction: x*theta, y*theta
            # Shape: (B, N) for x_proj, (B, M) for y_proj
            x_proj_1d = torch.matmul(x, direction)  # (B, N, D) @ (D,) = (B, N)
            y_proj_1d = torch.matmul(y, direction)  # (B, M, D) @ (D,) = (B, M)

            # Flatten batch dimension for sorting
            # Shape: (B*N,) and (B*M,)
            x_proj_flat = x_proj_1d.reshape(-1)
            y_proj_flat = y_proj_1d.reshape(-1)

            # Sort projections (required for 1D Wasserstein distance)
            x_sorted = torch.sort(x_proj_flat)[0]
            y_sorted = torch.sort(y_proj_flat)[0]

            # Handle different sample sizes (pad shorter one)
            if x_sorted.shape[0] != y_sorted.shape[0]:
                min_len = min(x_sorted.shape[0], y_sorted.shape[0])
                x_sorted = x_sorted[:min_len]
                y_sorted = y_sorted[:min_len]

            # 1D Wasserstein-2 distance: mean of squared differences
            w2_distance = torch.mean((x_sorted - y_sorted) ** 2)
            total_transport_cost += w2_distance

        # Average over all projections
        avg_transport_cost = total_transport_cost / self.num_projections

        return avg_transport_cost


class MPGEnhancement(nn.Module):
    """
    MPG (Manifold-Preserving Gating) Feature Enhancement Module.

    Enhances observation features (VLM outputs) by measuring their alignment
    with action embeddings using Sliced Wasserstein distance. Features that are
    well-aligned with actions receive stronger enhancement.

    Architecture:
        obs_proj: Linear(D, D) - Project observation features
        action_proj: Linear(D, D) - Project action features
        obs_layer_norm: LayerNorm - Normalize obs features for scale invariance
        action_layer_norm: LayerNorm - Normalize action features for scale invariance
        enhancement_proj: Linear(D, D) - Generate enhancement residual
        sliced_wasserstein: Compute alignment score

    Forward Pass:
        1. Project obs and action features to common space
        2. Handle batch dimension mismatch (average action across batch if needed)
        3. Apply LayerNorm for scale-invariant Wasserstein computation
        4. Pool and align action features
        5. Compute Wasserstein discrepancy D
        6. Gate: g = exp(-D/tau) with temperature scaling
        7. Apply gate to projected obs: gated_obs = obs_proj(obs) * g
        8. Enhancement: enhancement = enhancement_proj(gated_obs)
        9. Residual: enhanced = obs + lambda * enhancement

    Args:
        obs_feature_dim: Dimension of observation/VLM features
        action_feature_dim: Dimension of action encoder features (can differ from obs_feature_dim)
        embedding_dim: Common embedding dimension for Wasserstein computation (default: use obs_feature_dim)
        num_projections: Number of Sliced Wasserstein projections (M=32 recommended)
        lambda_strength: Residual connection strength (lambda). 0.0 = disabled.
        use_stop_gradient: Apply stop-gradient to gate (recommended: True)
        gate_temperature: Temperature for gate scaling. Higher = softer gating. Default: 1.0
                         Calibrated for LayerNorm-normalized features where D ~ 1.0
    """

    def __init__(
        self,
        obs_feature_dim: int,
        action_feature_dim: int | None = None,
        embedding_dim: int | None = None,
        num_projections: int = 32,
        lambda_strength: float = 0.0,
        use_stop_gradient: bool = True,
        gate_temperature: float = 1.0,
    ):
        super().__init__()

        # Default: action_feature_dim = obs_feature_dim (backward compatibility)
        if action_feature_dim is None:
            action_feature_dim = obs_feature_dim

        # Default: embedding_dim = obs_feature_dim
        if embedding_dim is None:
            embedding_dim = obs_feature_dim

        self.obs_feature_dim = obs_feature_dim
        self.action_feature_dim = action_feature_dim
        self.embedding_dim = embedding_dim
        self.num_projections = num_projections
        self.lambda_strength = lambda_strength
        self.use_stop_gradient = use_stop_gradient
        self.gate_temperature = gate_temperature

        # Linear projections to common embedding space
        self.obs_proj = nn.Linear(obs_feature_dim, embedding_dim)
        self.action_proj = nn.Linear(action_feature_dim, embedding_dim)

        # LayerNorm for scale-invariant Wasserstein computation (no learnable params)
        # This ensures transport cost stays bounded (~1.0) regardless of feature magnitude
        self.obs_layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.action_layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

        # Enhancement projection (generates residual features)
        self.enhancement_proj = nn.Linear(embedding_dim, obs_feature_dim)

        # Sliced Wasserstein distance module
        self.sliced_wasserstein = SlicedWassersteinDistance(num_projections)

    def forward(
        self,
        obs_features: torch.Tensor,
        action_features: torch.Tensor,
        return_gate: bool = False,
        return_metrics: bool = False,
    ):
        """
        Apply MPG enhancement to observation features.

        Args:
            obs_features: Observation/VLM features, shape (B_obs, T_obs, D)
            action_features: CLEAN action embeddings (t=0), shape (B_act, T_act, D)
                            Note: B_obs and B_act can differ; action will be averaged if needed
            return_gate: If True, return (enhanced, gate, None). Otherwise return enhanced only.
            return_metrics: If True, return (enhanced, gate, transport_cost). Requires return_gate=True.

        Returns:
            enhanced: Enhanced observation features, shape (B_obs, T_obs, D)
            gate: (optional) Alignment gate value, scalar. Returned if return_gate=True.
            transport_cost: (optional) Sliced Wasserstein distance, scalar. Returned if return_metrics=True.

        Note:
            - action_features should be CLEAN (non-noisy) action embeddings
            - For flow matching, encode with t=0 (not the noisy t~Beta used for training)
            - lambda_strength=0 disables enhancement (returns obs_features unchanged)
            - return_metrics requires return_gate=True
            - Batch dimension mismatch is handled by averaging action_features across batch
        """
        # Validate input tensor dimensions
        if obs_features.dim() != 3 or action_features.dim() != 3:
            raise ValueError(
                f"Expected 3D tensors (B, T, D), got "
                f"obs_features.shape={obs_features.shape}, "
                f"action_features.shape={action_features.shape}"
            )

        # Validate dtype consistency
        if obs_features.dtype != action_features.dtype:
            raise ValueError(
                f"obs_features and action_features must have same dtype, "
                f"got {obs_features.dtype} vs {action_features.dtype}"
            )

        batch_size_obs, seq_len_obs, _ = obs_features.shape
        batch_size_act, seq_len_act, _ = action_features.shape
        device = obs_features.device

        # Quick path: if lambda=0, no enhancement needed
        if self.lambda_strength == 0.0:
            zero_cost = torch.tensor(0.0, device=device)
            if return_metrics:
                return obs_features, torch.tensor(1.0, device=device), zero_cost
            if return_gate:
                return obs_features, torch.tensor(1.0, device=device), None
            return obs_features

        # 1. Project features to common embedding space
        obs_emb = self.obs_proj(obs_features)  # (B_obs, T_obs, D)
        action_emb = self.action_proj(action_features)  # (B_act, T_act, D)

        # 2. Handle batch dimension mismatch
        # When obs comes from packed sequence (B=1) and action from batch (B=N),
        # average action features across batch to match
        if batch_size_obs != batch_size_act:
            action_emb = action_emb.mean(dim=0, keepdim=True)  # (1, T_act, D)

        # 3. Apply LayerNorm for scale-invariant Wasserstein computation
        # This ensures transport cost stays bounded (~1.0) regardless of feature magnitude
        obs_emb_norm = self.obs_layer_norm(obs_emb)  # (B, T_obs, D)
        action_emb_norm = self.action_layer_norm(action_emb)  # (B, T_act, D)

        # 4. Pool action features and align with observation sequence
        # Use mean pooling across action sequence
        action_pooled = action_emb_norm.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Broadcast to match observation sequence length
        action_aligned = action_pooled.expand(-1, seq_len_obs, -1)  # (B, T_obs, D)

        # 5. Compute Sliced Wasserstein discrepancy
        # Measures how well observation features align with action distribution
        D = self.sliced_wasserstein(obs_emb_norm, action_aligned)  # Scalar

        # 6. Compute alignment gate with temperature scaling
        # Lower discrepancy -> higher gate -> stronger enhancement
        # Temperature scaling ensures gate stays in meaningful range (~0.5-0.7) for D~1.0
        gate = torch.exp(-D / self.gate_temperature)  # Range: (0, 1]

        # 7. Apply gate to projected observation features (use unnormalized for enhancement)
        # CODE's formula: gate * projected_input (NOT gate * residual as in paper)
        if self.use_stop_gradient:  # noqa: SIM108 - Branches carry per-case comments.
            # Stop gradient through gate (prevents gate collapse)
            gated_obs = obs_emb * gate.detach()
        else:
            gated_obs = obs_emb * gate

        # 8. Generate enhancement via projection
        enhancement = self.enhancement_proj(gated_obs)  # (B, T_obs, D)

        # 9. Residual connection with lambda scaling
        enhanced = obs_features + self.lambda_strength * enhancement

        if return_metrics:
            return enhanced, gate, D
        if return_gate:
            return enhanced, gate, None
        return enhanced

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"obs_feature_dim={self.obs_feature_dim}, "
            f"action_feature_dim={self.action_feature_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_projections={self.num_projections}, "
            f"lambda_strength={self.lambda_strength}, "
            f"use_stop_gradient={self.use_stop_gradient}, "
            f"gate_temperature={self.gate_temperature}"
        )
