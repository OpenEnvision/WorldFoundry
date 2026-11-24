"""Action-policy modules required to load and run VLANeXt checkpoints."""

import math

import torch
import torch.nn as nn

from worldfoundry.core.nn.transformer.dit import modulate_sequence as modulate


# -----------------------------------------------------------------------------
# ----------------------------- Shared Components -----------------------------
# -----------------------------------------------------------------------------
class MetaQueryBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)

        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MoEBlock(nn.Module):
    def __init__(self, hidden_size, vlm_hidden_size, num_heads, mlp_ratio=4.0, gen_hidden_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

        if gen_hidden_size is not None:
            self.gen_proj = nn.Linear(gen_hidden_size, hidden_size)
        else:
            self.gen_proj = None

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, vlm_feat, gen_feat=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        v_feat = self.vlm_proj(vlm_feat)
        kv_list = [x_norm, v_feat]

        if gen_feat is not None and self.gen_proj is not None:
            g_feat = self.gen_proj(gen_feat)
            kv_list.append(g_feat)

        kv = torch.cat(kv_list, dim=1)

        attn_out, _ = self.attn(query=x_norm, key=kv, value=kv)

        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# -----------------------------------------------------------------------------
# ----------------------------- Diffusion Policies ----------------------------
# -----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class ActionDiffusionTransformerMetaquery(nn.Module):
    def __init__(self, action_dim, condition_dim, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_proj = nn.Linear(condition_dim, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))

        self.blocks = nn.ModuleList([MetaQueryBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, noisy_action, timestep, condition, history_actions=None):
        noisy_action = noisy_action.to(dtype=self.input_proj.weight.dtype)
        condition = condition.to(dtype=self.cond_proj.weight.dtype)

        if history_actions is not None:
            history_actions = history_actions.to(dtype=self.input_proj.weight.dtype)
            x_input = torch.cat([history_actions, noisy_action], dim=1)
        else:
            x_input = noisy_action

        x = self.input_proj(x_input)
        x = x + self.pos_embed[:, : x.shape[1], :]

        t = self.t_embedder(timestep)
        c = self.cond_proj(condition) + t

        for block in self.blocks:
            x = block(x, c)

        output = self.final_layer(x, c)

        if history_actions is not None:
            output = output[:, -noisy_action.shape[1] :, :]

        return output


class ActionDiffusionTransformerMoE(nn.Module):
    def __init__(
        self, action_dim, vlm_hidden_size, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0, gen_hidden_size=None
    ):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))

        self.blocks = nn.ModuleList(
            [
                MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, gen_hidden_size=gen_hidden_size)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            if block.gen_proj is not None:
                nn.init.xavier_uniform_(block.gen_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, noisy_action, timestep, vlm_hidden_states, history_actions=None, gen_hidden_states=None):
        noisy_action = noisy_action.to(dtype=self.input_proj.weight.dtype)

        if history_actions is not None:
            history_actions = history_actions.to(dtype=self.input_proj.weight.dtype)
            x_input = torch.cat([history_actions, noisy_action], dim=1)
        else:
            x_input = noisy_action

        x = self.input_proj(x_input)
        x = x + self.pos_embed[:, : x.shape[1], :]
        t = self.t_embedder(timestep)

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks) :]

        relevant_gen_states = [None] * len(self.blocks)
        if gen_hidden_states is not None:
            relevant_gen_states = gen_hidden_states[-len(self.blocks) :]

        for block, vlm_state, gen_state in zip(self.blocks, relevant_vlm_states, relevant_gen_states, strict=False):
            vlm_state = vlm_state.to(dtype=x.dtype)
            if gen_state is not None:
                gen_state = gen_state.to(dtype=x.dtype)
            x = block(x, t, vlm_state, gen_feat=gen_state)

        output = self.final_layer(x, t)

        if history_actions is not None:
            output = output[:, -noisy_action.shape[1] :, :]

        return output


# -----------------------------------------------------------------------------
# ----------------------------- Regression Policies ---------------------------
# -----------------------------------------------------------------------------
class ActionRegressionTransformerMetaquery(nn.Module):
    def __init__(self, action_dim, condition_dim, num_actions=1, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim

        self.input_proj = nn.Linear(action_dim, hidden_size)

        self.query_embed = nn.Parameter(torch.zeros(1, num_actions, hidden_size))

        self.cond_proj = nn.Linear(condition_dim, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))

        self.blocks = nn.ModuleList([MetaQueryBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None):
        B = condition.shape[0]
        dtype = self.input_proj.weight.dtype
        condition = condition.to(dtype=dtype)

        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)

        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([history_emb, queries], dim=1)
        else:
            x = queries

        x = x + self.pos_embed[:, : x.shape[1], :]
        c = self.cond_proj(condition)

        for block in self.blocks:
            x = block(x, c)

        output = self.final_layer(x, c)
        output = output[:, -self.num_actions :, :]

        return output


class ActionRegressionTransformerMoE(nn.Module):
    def __init__(
        self,
        action_dim,
        vlm_hidden_size,
        num_actions=1,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        gen_hidden_size=None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.zeros(1, num_actions, hidden_size))

        self.cond_proj = nn.Linear(vlm_hidden_size, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))

        self.blocks = nn.ModuleList(
            [
                MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, gen_hidden_size=gen_hidden_size)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer1D(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            if block.gen_proj is not None:
                nn.init.xavier_uniform_(block.gen_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None, gen_hidden_states=None):
        vlm_hidden_states = condition

        final_state = vlm_hidden_states[-1]
        dtype = self.input_proj.weight.dtype
        final_state = final_state.to(dtype=dtype)

        c_emb = final_state.mean(dim=1)
        c = self.cond_proj(c_emb)

        B = c.shape[0]

        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)

        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([history_emb, queries], dim=1)
        else:
            x = queries

        x = x + self.pos_embed[:, : x.shape[1], :]

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks) :]

        relevant_gen_states = [None] * len(self.blocks)
        if gen_hidden_states is not None:
            relevant_gen_states = gen_hidden_states[-len(self.blocks) :]

        for block, vlm_state, gen_state in zip(self.blocks, relevant_vlm_states, relevant_gen_states, strict=False):
            vlm_state = vlm_state.to(dtype=dtype)
            if gen_state is not None:
                gen_state = gen_state.to(dtype=dtype)
            x = block(x, c, vlm_state, gen_feat=gen_state)

        output = self.final_layer(x, c)
        output = output[:, -self.num_actions :, :]

        return output


# -----------------------------------------------------------------------------
# --------------------------- Classification Policies -------------------------
# -----------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """Codebook lookup retained for action VQ decoding."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_(0, 0.02)


class ActionVQVAE(nn.Module):
    def __init__(
        self, action_dim=7, latent_codes_per_step=3, codebook_size=1024, hidden_size=256, depth=2, num_heads=4
    ):
        super().__init__()
        self.latent_codes = latent_codes_per_step
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size

        self.vq = VectorQuantizer(codebook_size, hidden_size)

        self.from_latent = nn.Linear(latent_codes_per_step * hidden_size, hidden_size)
        self.dec_pos = nn.Parameter(torch.zeros(1, 1024, hidden_size))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=depth)
        self.out_proj = nn.Linear(hidden_size, action_dim)

    def decode_indices(self, indices):
        B, T, _ = indices.shape
        indices_flat = indices.view(-1)

        codes = self.vq.embedding(indices_flat)
        codes = codes.view(B, T, self.latent_codes, self.hidden_size)

        codes_flat = codes.view(B, T, -1)
        dec_in = self.from_latent(codes_flat)
        dec_in = dec_in + self.dec_pos[:, :T, :]
        dec_out = self.decoder(dec_in)
        action = self.out_proj(dec_out)
        return action

    def decode_probs(self, probs):
        B, T, L, C = probs.shape
        codes = torch.matmul(probs, self.vq.embedding.weight)

        codes_flat = codes.view(B, T, -1)
        dec_in = self.from_latent(codes_flat)
        dec_in = dec_in + self.dec_pos[:, :T, :]
        dec_out = self.decoder(dec_in)
        action = self.out_proj(dec_out)
        return action


class ActionClassificationTransformerMetaquery(nn.Module):
    def __init__(
        self,
        action_dim,
        condition_dim,
        num_actions=1,
        num_bins=256,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        vqvae_mode=False,
        vq_codebook_size=1024,
        vq_latent_codes=3,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.vqvae_mode = vqvae_mode
        self.vq_codebook_size = vq_codebook_size
        self.vq_latent_codes = vq_latent_codes
        self.pose_dim = action_dim - 1

        self.dim_per_action = vq_latent_codes if vqvae_mode else action_dim
        self.total_queries = num_actions * self.dim_per_action
        self.per_dim_classes = vq_codebook_size if vqvae_mode else num_bins

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.zeros(1, self.total_queries, hidden_size))
        self.cond_proj = nn.Linear(condition_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))

        self.blocks = nn.ModuleList([MetaQueryBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])

        self.final_layer = FinalLayer1D(hidden_size, self.per_dim_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None):
        B = condition.shape[0]
        dtype = self.input_proj.weight.dtype
        condition = condition.to(dtype=dtype)

        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)

        if history_actions is not None:
            hist_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([hist_emb, queries], dim=1)
        else:
            x = queries

        x = x + self.pos_embed[:, : x.shape[1], :]

        c = self.cond_proj(condition)

        for block in self.blocks:
            x = block(x, c)

        output = self.final_layer(x, c)
        output = output[:, -self.total_queries :, :]

        if self.vqvae_mode:
            return output.view(B, self.num_actions, self.vq_latent_codes, self.per_dim_classes)
        else:
            return output.view(B, self.num_actions, self.action_dim, self.per_dim_classes)


class ActionClassificationTransformerMoE(nn.Module):
    def __init__(
        self,
        action_dim,
        vlm_hidden_size,
        num_actions=1,
        num_bins=256,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        vqvae_mode=False,
        vq_codebook_size=1024,
        vq_latent_codes=3,
        gen_hidden_size=None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.vqvae_mode = vqvae_mode
        self.vq_codebook_size = vq_codebook_size
        self.vq_latent_codes = vq_latent_codes

        self.dim_per_action = vq_latent_codes if vqvae_mode else action_dim
        self.total_queries = num_actions * self.dim_per_action
        self.per_dim_classes = vq_codebook_size if vqvae_mode else num_bins

        self.input_proj = nn.Linear(action_dim, hidden_size)
        self.query_embed = nn.Parameter(torch.zeros(1, self.total_queries, hidden_size))

        self.cond_proj = nn.Linear(vlm_hidden_size, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size))

        self.blocks = nn.ModuleList(
            [
                MoEBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio, gen_hidden_size=gen_hidden_size)
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer1D(hidden_size, self.per_dim_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.vlm_proj.weight)
            if block.gen_proj is not None:
                nn.init.xavier_uniform_(block.gen_proj.weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, condition, history_actions=None, gen_hidden_states=None):
        vlm_hidden_states = condition

        final_state = vlm_hidden_states[-1]
        dtype = self.input_proj.weight.dtype
        final_state = final_state.to(dtype=dtype)

        c_emb = final_state.mean(dim=1)
        c = self.cond_proj(c_emb)

        B = c.shape[0]

        queries = self.query_embed.expand(B, -1, -1).to(dtype=dtype)

        if history_actions is not None:
            history_emb = self.input_proj(history_actions.to(dtype=dtype))
            x = torch.cat([history_emb, queries], dim=1)
        else:
            x = queries

        x = x + self.pos_embed[:, : x.shape[1], :]

        relevant_vlm_states = vlm_hidden_states[-len(self.blocks) :]

        relevant_gen_states = [None] * len(self.blocks)
        if gen_hidden_states is not None:
            relevant_gen_states = gen_hidden_states[-len(self.blocks) :]

        for block, vlm_state, gen_state in zip(self.blocks, relevant_vlm_states, relevant_gen_states, strict=False):
            vlm_state = vlm_state.to(dtype=dtype)
            if gen_state is not None:
                gen_state = gen_state.to(dtype=dtype)
            x = block(x, c, vlm_state, gen_feat=gen_state)

        output = self.final_layer(x, c)
        output = output[:, -self.total_queries :, :]

        if self.vqvae_mode:
            return output.view(B, self.num_actions, self.vq_latent_codes, self.per_dim_classes)
        else:
            return output.view(B, self.num_actions, self.action_dim, self.per_dim_classes)
